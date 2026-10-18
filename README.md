# Benchmark da CNN 1D no Arduino Nano 33 BLE Sense Rev2

Mede FP32, INT16 e INT8 da mesma CNN na placa e gera CSVs com latência, RAM, Flash e acurácia/F1 feitas **na placa**. Energia fica de fora por enquanto.

```
arduino_benchmark/
├── cnn_benchmark/cnn_benchmark.ino   sketch: recebe uma janela pela serial, roda o modelo e devolve tempos e scores
├── cnn_benchmark/config.h            reescrito pelo pipeline a cada build (versão do modelo e tamanho do arena)
├── run_benchmark.py                  pipeline: compila, grava, mede, salva CSVs
├── apply_tflm_patch.py               habilita int16 no EXPAND_DIMS da biblioteca TFLM (só para a versão INT16)
└── requirements.txt
```

## 1. Preparar o computador (uma vez)

```bash
arduino-cli core update-index
arduino-cli core install arduino:mbed_nano          # o Sense Rev2 usa a mesma placa: arduino:mbed_nano:nano33ble
arduino-cli core install esp32:esp32


# A biblioteca TFLM oficial saiu do Library Manager: instala-se direto do GitHub
arduino-cli config set library.enable_unsafe_install true
arduino-cli lib install --git-url https://github.com/tensorflow/tflite-micro-arduino-examples
arduino-cli lib install ArduinoBLE Arduino_LSM9DS1  # dependências declaradas pela biblioteca
```
```bash
# Para esp32
git clone --depth 1 https://github.com/espressif/esp-tflite-micro "%USERPROFILE%\Documents\Arduino\libraries\esp-tflite-micro"
```
ou 
```bash
arduino-cli lib install --git-url https://github.com/johnosbb/MicroTFLite
```
Depois, aplique o patch para rodar INT16

```bash
python apply_tflm_patch.py                           # necessário para a versão INT16
pip install -r requirements.txt
```

No Arduino IDE o equivalente é *Sketch > Incluir Biblioteca > Adicionar biblioteca .ZIP* com o ZIP do repositório acima.
No Linux, o usuário precisa estar no grupo `dialout` para abrir a porta serial.

**Por que a biblioteca oficial e não a Chirale (Library Manager)?** Na Chirale 2.0.0 o `MAX_POOL_2D` só aceita float32 e int8, então o INT16 falha. FP32 e INT8 rodam com as duas, e o sketch aceita ambas.

## 2. Levar os arquivos do notebook

Rode o notebook até a última célula. Ele grava em `CWRU_Data/arduino_export/` (Drive): `modelo_*.h`, `modelo_*.tflite`, `X_test_cnn.npy`, `y_test_cnn.npy`, `pred_tflite_*.npy`, `meta.json`. Copie a pasta inteira para o computador ligado à placa.

## 3. Teste manual rápido (recomendado antes da campanha)

Copie `modelo_int8.h` para a pasta `cnn_benchmark/`, abra o sketch no IDE (`config.h` já vem com `MODEL_INT8` e `ARENA_KB 80`), envie e abra o Serial Monitor em 115200 com "Nova linha". `PING` deve responder `PONG` e `INFO` deve mostrar `alloc=1` e o `arena_used`.

## 4. Rodar a campanha

```bash
py run_benchmark.py --export-dir CAMINHO/arduino_export --fqbn arduino:mbed_nano:nano33ble --sessions 3
```
ou

```bash
py run_benchmark.py --export-dir arduino_export --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,FlashSize=8M,PartitionScheme=default_8MB --variants INT8 --runs 100 --port COMx
```

Para cada sessão, as três versões são medidas em **ordem aleatória** (semente `--seed`). Para cada versão:

1. procura o menor arena que a placa aceita, começando pela estimativa do notebook, e recompila com esse arena;
2. mede a latência: `--warmup` execuções descartadas + `--runs` (padrão 1000) cronometradas na placa;
3. envia todo o conjunto de teste janela a janela e calcula acurácia/F1 da inferência feita na placa (só na primeira sessão em que a versão é medida).

Opções úteis: `--variants INT8`, `--port /dev/ttyACM0`, `--runs`, `--skip-accuracy`, `--latency-window`.
A primeira compilação é lenta, porque o core Mbed e o TFLM são compilados do zero.

## 5. O que sai em `resultados/`

| Arquivo | Conteúdo |
|---|---|
| `resumo_final.csv` | uma linha por versão: latência agregada das sessões, RAM, Flash, arena, acurácia/F1 |
| `resumo_por_sessao.csv` | uma linha por sessão e versão |
| `latencia_bruta.csv` | cada execução (µs), para histogramas e testes estatísticos |
| `acuracia_na_placa.csv` | predição e scores de cada janela de teste |
| `sessao_meta.json` | versões do arduino-cli, do core e das bibliotecas, data, argumentos |

| Variável do canvas | Colunas | Como é medida |
|---|---|---|
| Tempo de inferência (ms, média ± DP) | `lat_mean_ms`, `lat_sd_ms` | `micros()` ao redor do `Invoke()`. A mesma janela é regravada na entrada antes de cada execução, fora do cronômetro. |
| Uso de RAM | `ram_bytes`, `ram_pct`, `arena_used_bytes`, `harness_bytes` | `ram_bytes` é a RAM estática do compilador (inclui Mbed OS, TFLM e o arena). `arena_used_bytes` é o que o TFLM realmente usou para os tensores. `harness_bytes` são os buffers do sketch de teste. |
| Uso de Flash | `flash_bytes`, `flash_pct`, `tflite_bytes` | sketch inteiro (Mbed + TFLM + modelo). `tflite_bytes` é só o modelo. |
| Acurácia / F1-macro | `acc_placa`, `f1_placa`, `concordancia_tflite_pc` | inferência na placa. A concordância compara com as predições do TFLite no PC. |
| Energia / EDP | (ainda não) | Com o INA219/INA226: EDP = energia × `lat_mean_ms`. `--trigger-pin 2` sobe um pino durante o `Invoke()` para sincronizar a medição. |

## 6. Status possíveis de uma versão

- `ok`: mediu normalmente.
- `no_fit_arena`: compila e inicia, mas o arena não basta para os tensores. É o esperado do **FP32**: o TFLM pediu ~259 KB de arena na simulação, para 256 KB de RAM total.
- `no_fit`: nenhum tamanho de arena compilou e iniciou.

Uma versão que não cabe é resultado da comparação e entra no relatório como "não executável nesta plataforma".

## 7. Cuidados de método

- A latência cobre só o `Invoke()`. Não inclui quantizar a entrada nem a comunicação serial.
- A RAM estática inclui o Mbed OS e o TFLM. Para comparar com outras plataformas, use também `arena_used_bytes` e subtraia `harness_bytes` se quiser isolar o modelo.
- O sketch espera a porta serial abrir (`while (!Serial)`). Para medir energia depois, com a placa alimentada fora do USB, esse trecho precisa sair.
- Feche o Serial Monitor antes de rodar o pipeline (a porta só abre para um programa).
- Se o upload falhar, dê dois toques rápidos no botão de reset (modo bootloader) e rode de novo.

## 8. O que foi verificado

Verificado no PC (x86), sem a placa: o sketch compila contra o código-fonte da biblioteca oficial (0 avisos com `-Wall`); as três versões rodam pelo protocolo serial; FP32 bate com o TFLite do Python quase bit a bit e INT8/INT16 dão 100% de concordância de predições; o pipeline completo roda contra um `arduino-cli` e uma placa simulados, incluindo a busca de arena e o caso "não cabe".

**Não verificado:** compilação com o core `arduino:mbed_nano` real, upload, e os números de latência/RAM/Flash na placa. Os valores de tempo e RAM dos testes vieram do simulador e não significam nada. No dispositivo, o `arena_used` pode ser maior que o simulado, por causa dos buffers de scratch do CMSIS-NN no Cortex-M4.
