/*
 * cnn_benchmark.ino
 *
 * Benchmark de uma CNN 1D (diagnóstico de falhas em rolamentos, CWRU) com TensorFlow Lite
 * for Microcontrollers. Testado no Arduino Nano 33 BLE Sense Rev2. O branco para ESP32-S3
 * abaixo NÃO foi compilado/testado em hardware real -- ver README, seção "ESP32-S3", antes
 * de usar, e valide primeiro com um exemplo hello_world da biblioteca antes de usar este sketch.
 *
 * O sketch NÃO decide nada sozinho: fica esperando comandos pela porta serial
 * e o script run_benchmark.py conduz o experimento.
 *
 * Protocolo (texto, uma linha por comando):
 *   PING            -> PONG
 *   INFO            -> INFO,chave=valor,...
 *   RUN <n>         -> GO
 *                      (o host envia 1 janela: input_len floats little-endian,
 *                       em blocos de 512 bytes; a cada bloco a placa responde ".")
 *                   -> n linhas "T,<tempo do Invoke em us>"
 *                   -> "R,<classe>,<score0>,<score1>,..."
 *                   -> "END"
 *   Em caso de erro: "ERR,<motivo>"
 *
 * A janela recebida é quantizada aqui na placa (scale/zero_point lidos do próprio
 * modelo), do mesmo jeito que um sensor real alimentaria o modelo.
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "config.h"

// O include abaixo é obrigatório e NÃO pode ficar dentro de um #if __has_include: o arduino-cli só
// descobre qual biblioteca compilar pelos #include que falham. Esse header é vazio de propósito e
// serve para puxar a biblioteca TFLM inteira.
//
// A escolha de biblioteca depende da arquitetura: no ARM (Nano 33 BLE Sense) usa-se a
// Arduino_TensorFlowLite (CMSIS-NN); no ESP32 ela não compila (Xtensa, não ARM), então
// usa-se a esp-tflite-micro da Espressif. Rode apply_tflm_patch.py uma vez para restringir
// a biblioteca ARM à arquitetura certa -- sem isso, o arduino-cli tenta (e falha) usá-la
// também no ESP32.


#if defined(ARDUINO_ARCH_ESP32)
  #include "tensorflow/lite/micro/system_setup.h"
  SET_LOOP_TASK_STACK_SIZE(16 * 1024);  
#elif defined(USE_CHIRALE)
  #include <Chirale_TensorFlowLite.h>   // biblioteca Chirale (Library Manager): sem INT16
#else
  #include <TensorFlowLite.h>           // biblioteca oficial ARM (GitHub): usada por padrão no Nano 33 BLE Sense
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if defined(MODEL_FP32)
#include "modelo_fp32.h"
#define MODEL_DATA modelo_fp32
#define MODEL_LEN modelo_fp32_len
#define MODEL_NAME "FP32"
#define INPUT_ELEM_BYTES 4
#elif defined(MODEL_INT16)
#include "modelo_int16.h"
#define MODEL_DATA modelo_int16
#define MODEL_LEN modelo_int16_len
#define MODEL_NAME "INT16"
#define INPUT_ELEM_BYTES 2
#elif defined(MODEL_INT8)
#include "modelo_int8.h"
#define MODEL_DATA modelo_int8
#define MODEL_LEN modelo_int8_len
#define MODEL_NAME "INT8"
#define INPUT_ELEM_BYTES 1
#else
#error "Defina MODEL_FP32, MODEL_INT16 ou MODEL_INT8 em config.h"
#endif

#ifndef ARENA_KB
#error "Defina ARENA_KB em config.h"
#endif

#ifndef WINDOW_LEN
#define WINDOW_LEN 2048     // tamanho da janela de vibração (amostras)
#endif

namespace {

constexpr size_t kArenaBytes = (size_t)ARENA_KB * 1024;
constexpr size_t kChunkBytes = 512;   // bloco de recepção da janela
constexpr int kBatch = 100;           // tempos são impressos em lotes (fora da região medida)
constexpr int kMaxClasses = 16;

alignas(16) uint8_t tensor_arena[kArenaBytes];

// O TFLM pode reaproveitar a memória do tensor de entrada para tensores intermediários,
// então a entrada é regravada antes de CADA Invoke a partir desta cópia (fora da região cronometrada).
constexpr size_t kInputBackupBytes = (size_t)WINDOW_LEN * INPUT_ELEM_BYTES;
uint8_t input_backup[kInputBackupBytes];
constexpr size_t kHarnessBytes = kInputBackupBytes + kChunkBytes + kBatch * sizeof(uint32_t);

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool alloc_ok = false;
size_t input_len = 0;   // número de elementos da janela
int n_classes = 0;

const char* type_name(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt16:   return "int16";
    case kTfLiteInt8:    return "int8";
    default:             return "outro";
  }
}

long clamp_long(long v, long lo, long hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Quantiza (se preciso) e grava um elemento da janela no tensor de entrada.
void store_input(size_t i, float v) {
  switch (input->type) {
    case kTfLiteFloat32:
      input->data.f[i] = v;
      break;
    case kTfLiteInt8: {
      long q = lrintf(v / input->params.scale) + input->params.zero_point;
      input->data.int8[i] = (int8_t)clamp_long(q, -128, 127);
      break;
    }
    case kTfLiteInt16: {
      long q = lrintf(v / input->params.scale) + input->params.zero_point;
      input->data.i16[i] = (int16_t)clamp_long(q, -32768, 32767);
      break;
    }
    default:
      break;
  }
}

// Lê uma janela (input_len floats) da serial, em blocos com confirmação.
bool receive_window() {
  static uint8_t chunk[kChunkBytes];
  const size_t total = input_len * sizeof(float);
  size_t received = 0, idx = 0;
  while (received < total) {
    size_t want = total - received;
    if (want > kChunkBytes) want = kChunkBytes;
    if (Serial.readBytes((char*)chunk, want) != want) return false;
    for (size_t k = 0; k + 4 <= want; k += 4) {
      float v;
      memcpy(&v, chunk + k, 4);
      store_input(idx++, v);
    }
    received += want;
    Serial.write('.');
    Serial.flush();
  }
  return true;
}

float read_score(int c) {
  switch (output->type) {
    case kTfLiteFloat32: return output->data.f[c];
    case kTfLiteInt8:    return (output->data.int8[c] - output->params.zero_point) * output->params.scale;
    case kTfLiteInt16:   return (output->data.i16[c] - output->params.zero_point) * output->params.scale;
    default:             return 0.0f;
  }
}

void print_info() {
  Serial.print("INFO,variant="); Serial.print(MODEL_NAME);
  Serial.print(",alloc="); Serial.print(alloc_ok ? 1 : 0);
  Serial.print(",arena_bytes="); Serial.print((unsigned long)kArenaBytes);
  Serial.print(",arena_used="); Serial.print(alloc_ok ? (unsigned long)interpreter->arena_used_bytes() : 0UL);
  Serial.print(",model_bytes="); Serial.print((unsigned long)MODEL_LEN);
  Serial.print(",harness_bytes="); Serial.print((unsigned long)kHarnessBytes);
  if (alloc_ok) {
    Serial.print(",input_len="); Serial.print((unsigned long)input_len);
    Serial.print(",n_classes="); Serial.print(n_classes);
    Serial.print(",in_type="); Serial.print(type_name(input->type));
    Serial.print(",in_scale="); Serial.print(input->params.scale, 9);
    Serial.print(",in_zp="); Serial.print(input->params.zero_point);
    Serial.print(",out_type="); Serial.print(type_name(output->type));
    Serial.print(",out_scale="); Serial.print(output->params.scale, 9);
    Serial.print(",out_zp="); Serial.print(output->params.zero_point);
  }
  Serial.println();
}

void run_command(int n_runs) {
  if (!alloc_ok) { Serial.println("ERR,ALLOC_FAILED"); return; }
  if (n_runs < 1) { Serial.println("ERR,BAD_N"); return; }

  Serial.println("GO");
  if (!receive_window()) { Serial.println("ERR,RX_TIMEOUT"); return; }
  if (input->bytes > kInputBackupBytes) { Serial.println("ERR,WINDOW_TOO_LARGE"); return; }
  memcpy(input_backup, input->data.raw, input->bytes);

  static uint32_t batch[kBatch];
  int done = 0;
  while (done < n_runs) {
    int m = n_runs - done;
    if (m > kBatch) m = kBatch;
    for (int i = 0; i < m; i++) {
      memcpy(input->data.raw, input_backup, input->bytes);   // mesma entrada em todas as execuções
#ifdef TRIGGER_PIN
      digitalWrite(TRIGGER_PIN, HIGH);
#endif
      uint32_t t0 = micros();
      TfLiteStatus st = interpreter->Invoke();
      uint32_t t1 = micros();
#ifdef TRIGGER_PIN
      digitalWrite(TRIGGER_PIN, LOW);
#endif
      if (st != kTfLiteOk) { Serial.println("ERR,INVOKE_FAILED"); return; }
      batch[i] = t1 - t0;
    }
    for (int i = 0; i < m; i++) {
      Serial.print("T,"); Serial.println((unsigned long)batch[i]);
    }
    Serial.flush();
    done += m;
  }

  int best = 0;
  float scores[kMaxClasses];
  for (int c = 0; c < n_classes && c < kMaxClasses; c++) {
    scores[c] = read_score(c);
    if (scores[c] > scores[best]) best = c;
  }
  Serial.print("R,"); Serial.print(best);
  for (int c = 0; c < n_classes && c < kMaxClasses; c++) {
    Serial.print(","); Serial.print(scores[c], 6);
  }
  Serial.println();
  Serial.println("END");
}

}  // namespace

void setup() {
#ifdef TRIGGER_PIN
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);
#endif

#if defined(ARDUINO_ARCH_ESP32)
  Serial.setRxBufferSize(2048);       
#endif
  Serial.begin(115200);
  unsigned long t_start = millis();
  while (!Serial && (millis() - t_start < 3000)) {
    delay(10);
  }

  Serial.setTimeout(3000);

  model = tflite::GetModel(MODEL_DATA);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERR,SCHEMA_VERSION");
    return;
  }

  // Operadores usados pela CNN (ver saída da célula de avaliação do notebook).
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddExpandDims();
  resolver.AddReshape();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaBytes);
  interpreter = &static_interpreter;

  alloc_ok = (interpreter->AllocateTensors() == kTfLiteOk);
  if (alloc_ok) {
    input = interpreter->input(0);
    output = interpreter->output(0);
    input_len = 1;
    for (int d = 0; d < input->dims->size; d++) input_len *= input->dims->data[d];
    n_classes = output->dims->data[output->dims->size - 1];
  }

  Serial.print("READY,");
  print_info();
}

void loop() {
  if (!Serial.available()) return;

  char line[32];
  size_t len = Serial.readBytesUntil('\n', line, sizeof(line) - 1);
  line[len] = '\0';
  while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == ' ')) line[--len] = '\0';
  if (len == 0) return;

  if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(line, "INFO") == 0) {
    print_info();
  } else if (strncmp(line, "RUN ", 4) == 0) {
    run_command(atoi(line + 4));
  } else {
    Serial.println("ERR,UNKNOWN_CMD");
  }
}
