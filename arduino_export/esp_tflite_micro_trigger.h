// esp_tflite_micro_trigger.h
//
// Este arquivo é um placeholder: eu (Claude) não consegui compilar para ESP32-S3 no meu
// ambiente (não tenho a toolchain Xtensa aqui), então não posso garantir qual biblioteca de
// TensorFlow Lite Micro vai compilar certo no seu ESP32-S3, nem qual é o "include gatilho" dela.
// Antes de mexer neste sketch, valide numa pasta separada com um exemplo hello_world da
// biblioteca -- veja o README, seção "ESP32-S3".
//
// Depois de validar qual biblioteca funciona na sua máquina, troque o
// #include "esp_tflite_micro_trigger.h"
// no cnn_benchmark.ino por um dos dois caminhos abaixo (o que tiver funcionado no teste):
//
// OPÇÃO A - esp-tflite-micro (Espressif, com aceleração ESP-NN):
//   git clone --depth 1 https://github.com/espressif/esp-tflite-micro \
//       "%USERPROFILE%\Documents\Arduino\libraries\esp-tflite-micro"
//   Depois, troque este #include pelo include usado no hello_world dessa biblioteca
//   (normalmente os mesmos headers tensorflow/lite/micro/... já usados neste sketch --
//   nesse caso basta apagar este arquivo e a linha que o inclui).
//
// OPÇÃO B - MicroTFLite (wrapper em estilo Arduino, já pensado para múltiplas placas,
// incluindo ESP32 e Nano 33 BLE):
//   arduino-cli lib install --git-url https://github.com/johnosbb/MicroTFLite
//   Essa biblioteca tem uma API própria (mais simples, sem acesso direto ao
//   tflite::MicroInterpreter), então usá-la exige adaptar as partes de setup() e
//   run_command() que mexem em "interpreter", "input" e "output" diretamente -- não é só
//   trocar o include.
//
// Qualquer que seja a opção, se a biblioteca escolhida tiver uma restrição de
// "architectures=" no library.properties dela, ótimo: ela já evita conflito com a
// Arduino_TensorFlowLite (ARM). Senão, considere restringi-la à mão à arquitetura "esp32",
// do mesmo jeito que apply_tflm_patch.py faz com a biblioteca ARM.

#error "Resolva este include antes de compilar para ESP32 -- veja os comentários acima e o README."
