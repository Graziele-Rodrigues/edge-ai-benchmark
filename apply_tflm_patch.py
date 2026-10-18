#!/usr/bin/env python3
"""
Habilita EXPAND_DIMS em int16 na biblioteca Arduino_TensorFlowLite (necessário para a versão INT16).

A versão oficial da biblioteca (repositório tensorflow/tflite-micro-arduino-examples) suporta int16 em
Conv2D, MaxPool2D, Mean, FullyConnected e Softmax, mas o kernel EXPAND_DIMS só aceita int8 e float32.
O TFLite do Keras insere EXPAND_DIMS antes de cada Conv1D, então a versão INT16 falha no primeiro nó.
Este script adiciona o caso int16 (a mesma mudança que o TFLite Micro upstream já tem).

Uso:  python apply_tflm_patch.py [--library-dir CAMINHO/Arduino_TensorFlowLite]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ALVO = Path("src/tensorflow/lite/micro/kernels/expand_dims.cpp")

ANTES = """    case kTfLiteInt8: {
      memCopyN(tflite::micro::GetTensorData<int8_t>(output),
               tflite::micro::GetTensorData<int8_t>(input), flat_size);
    } break;
"""
DEPOIS = """    case kTfLiteInt16: {
      memCopyN(tflite::micro::GetTensorData<int16_t>(output),
               tflite::micro::GetTensorData<int16_t>(input), flat_size);
    } break;
""" + ANTES


def achar_biblioteca(explicita):
    if explicita:
        return Path(explicita)
    candidatos = []
    try:
        out = subprocess.run(["arduino-cli", "config", "get", "directories.user"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            candidatos.append(Path(out) / "libraries" / "Arduino_TensorFlowLite")
    except FileNotFoundError:
        pass
    home = Path.home()
    candidatos += [home / "Arduino/libraries/Arduino_TensorFlowLite",
                   home / "Documents/Arduino/libraries/Arduino_TensorFlowLite"]
    for c in candidatos:
        if (c / ALVO).exists():
            return c
    sys.exit("Não achei a biblioteca Arduino_TensorFlowLite. Use --library-dir.")


def restringir_arquitetura(lib):
    """Impede que o arduino-cli tente usar esta biblioteca (ARM/CMSIS-NN) ao compilar para outras
    placas, como um ESP32 -- sem isso, o arduino-cli a considera compatível com qualquer arquitetura
    e tenta compilá-la também lá, o que falha (CMSIS-NN não existe para Xtensa)."""
    arq = lib / "library.properties"
    if not arq.exists():
        sys.exit(f"library.properties não encontrado em: {lib}")
    texto = arq.read_text()
    if re.search(r"^architectures=", texto, re.M):
        print("library.properties já restringe arquitetura; nada a fazer em:", arq)
        return
    if not arq.with_suffix(".properties.orig").exists():
        arq.with_suffix(".properties.orig").write_text(texto)
    arq.write_text(texto.rstrip("\n") + "\narchitectures=mbed,mbed_nano,samd\n")
    print("Restrição de arquitetura adicionada em:", arq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library-dir")
    args = ap.parse_args()
    lib = achar_biblioteca(args.library_dir)
    restringir_arquitetura(lib)
    arq = lib / ALVO
    if not arq.exists():
        sys.exit(f"Arquivo não encontrado: {arq}")
    texto = arq.read_text()
    if "case kTfLiteInt16" in texto:
        print("Patch já aplicado (ou a biblioteca já suporta int16 no EXPAND_DIMS):", arq)
        return
    if ANTES not in texto:
        sys.exit("O trecho esperado não existe neste arquivo (versão diferente da biblioteca?). Aplique à mão: "
                 "adicione um 'case kTfLiteInt16' igual ao de int8, com int16_t, no switch da função Eval.")
    arq.with_suffix(".cpp.orig").write_text(texto)
    arq.write_text(texto.replace(ANTES, DEPOIS, 1).replace("supports int8 and float32", "supports int8, int16 and float32"))
    print("Patch aplicado em:", arq, "(cópia original em .cpp.orig)")


if __name__ == "__main__":
    main()
