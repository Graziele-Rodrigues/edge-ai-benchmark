#!/usr/bin/env python3
"""
Pipeline de medição da CNN 1D (CWRU) no Arduino Nano 33 BLE Sense Rev2.

Para cada versão do modelo (FP32, INT16, INT8), em ordem aleatória a cada sessão:
  1. compila o sketch com arduino-cli (config.h é reescrito com a versão e o tamanho do arena);
  2. grava na placa e espera o "READY" pela serial;
  3. mede a latência do Invoke() (N execuções da mesma janela, tempo medido na placa com micros());
  4. envia todo o conjunto de teste, janela a janela, e mede a acurácia/F1 da inferência feita NA PLACA;
  5. salva os dados brutos e um resumo em CSV.

Métricas coletadas (energia fica de fora por enquanto):
  latência (ms), RAM (bytes e % da capacidade), Flash (bytes e % da capacidade), arena usado pelo TFLM,
  acurácia e F1-macro na placa, concordância com o TFLite do PC.

Uso:
  python run_benchmark.py --export-dir CAMINHO/arduino_export
"""

import argparse
import datetime
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FQBN_PADRAO = "arduino:mbed_nano:nano33ble"   # o Sense Rev2 usa a mesma placa/core do Nano 33 BLE
ARDUINO_VID = 0x2341
CHUNK = 512                                    # deve ser igual a kChunkBytes no sketch
VARIANTES = ["FP32", "INT16", "INT8"]


class BenchError(Exception):
    pass


# ----------------------------------------------------------------------------
# arduino-cli
# ----------------------------------------------------------------------------

def run_cli(cli, args, check=True, heartbeat=30):
    """Roda o arduino-cli e, enquanto ele trabalha (a 1ª compilação leva vários minutos), avisa que está vivo."""
    proc = subprocess.Popen([cli, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding="utf-8", errors="replace")
    inicio = time.time()
    while True:
        try:
            saida, erro = proc.communicate(timeout=heartbeat)
            break
        except subprocess.TimeoutExpired:
            seg = int(time.time() - inicio)
            print(f"      ... arduino-cli {args[0]} em andamento ({seg // 60} min {seg % 60:02d} s)", flush=True)
    resultado = subprocess.CompletedProcess(proc.args, proc.returncode, saida, erro)
    if check and resultado.returncode != 0:
        raise BenchError(f"arduino-cli {' '.join(args[:2])} falhou:\n{saida[-1500:]}\n{erro[-1500:]}")
    return resultado


RE_FLASH = re.compile(r"Sketch uses (\d+) bytes \((\d+)%\) of program storage space\. Maximum is (\d+) bytes")
RE_RAM = re.compile(r"Global variables use (\d+) bytes \((\d+)%\) of dynamic memory.*?Maximum is (\d+) bytes", re.S)


def parse_sizes(text):
    f, r = RE_FLASH.search(text), RE_RAM.search(text)
    if not (f and r):
        raise BenchError("Não consegui ler os tamanhos de Flash/RAM na saída do arduino-cli:\n" + text[-800:])
    return {"flash_bytes": int(f[1]), "flash_max_bytes": int(f[3]),
            "ram_bytes": int(r[1]), "ram_max_bytes": int(r[3])}


def write_config(sketch_dir, variante, arena_kb, window_len, trigger_pin):
    linhas = ["// Gerado por run_benchmark.py - não edite.", "#pragma once",
              f"#define MODEL_{variante}", f"#define ARENA_KB {arena_kb}", f"#define WINDOW_LEN {window_len}"]
    if trigger_pin is not None:
        linhas.append(f"#define TRIGGER_PIN {trigger_pin}")
    (sketch_dir / "config.h").write_text("\n".join(linhas) + "\n")


def overflow_bytes(texto):
    """Quantos bytes a RAM estourou, quando o linker informa (dois formatos de mensagem)."""
    m = re.search(r"region .?RAM.? overflowed by (\d+) bytes", texto)
    if m:
        return int(m[1])
    m = re.search(r"cannot move location counter backwards \(from ([0-9a-fA-F]+) to ([0-9a-fA-F]+)\)", texto)
    if m:
        return int(m[1], 16) - int(m[2], 16)
    return None


def compile_sketch(cfg, variante, arena_kb):
    """Devolve dict com status ('ok' | 'no_fit_link') e os tamanhos de Flash/RAM."""
    shutil.copy(cfg.export_dir / f"modelo_{variante.lower()}.h", cfg.sketch_dir / f"modelo_{variante.lower()}.h")
    write_config(cfg.sketch_dir, variante, arena_kb, cfg.window_len, cfg.trigger_pin)
    for tentativa in range(3):
        proc = run_cli(cfg.cli, ["compile", "--fqbn", cfg.fqbn, "--build-path", str(cfg.build_dir),
                                 "--warnings", "none", str(cfg.sketch_dir)], check=False)
        texto = proc.stdout + "\n" + proc.stderr
        if proc.returncode == 0:
            return {"status": "ok", **parse_sizes(texto)}
        if "Permission denied" in texto and tentativa < 2:   # arquivo travado por antivírus/OneDrive/processo anterior
            print("      ... arquivo travado pelo Windows; tentando de novo em 10 s")
            time.sleep(10)
            continue
        break
    if re.search(r"region .?(RAM|FLASH).? overflowed|will not fit in region|cannot move location counter", texto):
        return {"status": "no_fit_link", "overflow_bytes": overflow_bytes(texto)}
    raise BenchError("Falha de compilação:\n" + texto[-2500:])


def upload_sketch(cfg, porta):
    run_cli(cfg.cli, ["upload", "-p", porta, "--fqbn", cfg.fqbn, "--input-dir", str(cfg.build_dir)])


# ----------------------------------------------------------------------------
# Serial
# ----------------------------------------------------------------------------

'''
class SerialBoard:
    def __init__(self, porta, timeout=10.0):
        import serial   # pyserial
        self.ser = serial.Serial(porta, 115200, timeout=timeout, write_timeout=10)
        self.ser.dtr = True   # o sketch espera a porta ser aberta (while (!Serial))

    def write(self, dados):
        self.ser.write(dados)
        self.ser.flush()

    def read(self, n):
        return self.ser.read(n)

    def readline(self):
        return self.ser.readline()

    def reset_input_buffer(self):
        self.ser.reset_input_buffer()

    def set_timeout(self, s):
        self.ser.timeout = s

    def close(self):
        self.ser.close()
'''
USA_JTAG = False
class SerialBoard:
    def __init__(self, porta, timeout=10.0):
        import serial
        self.ser = serial.Serial()
        self.ser.port = porta
        self.ser.baudrate = 115200
        self.ser.timeout = timeout
        self.ser.write_timeout = 10
        # No USB-Serial/JTAG do S3, DTR ligado derruba o GPIO0 (modo download) e
        # RTS ligado segura o EN (reset). No Nano, o DTR é necessário para o while(!Serial).
        self.ser.dtr = not USA_JTAG
        self.ser.rts = False
        self.ser.open()

    def write(self, dados):
        self.ser.write(dados)
        self.ser.flush()

    def read(self, n):
        return self.ser.read(n)

    def readline(self):
        return self.ser.readline()

    def reset_input_buffer(self):
        self.ser.reset_input_buffer()

    def set_timeout(self, s):
        self.ser.timeout = s

    def close(self):
        self.ser.close()

def find_port(explicita=None, espera=40):
    """A porta USB pode mudar de nome depois do upload; espera a placa reaparecer."""
    from serial.tools import list_ports
    limite = time.time() + espera
    while time.time() < limite:
        portas = list_ports.comports()
        if explicita and explicita in [p.device for p in portas]:
            return explicita
        arduinos = [p.device for p in portas if p.vid == ARDUINO_VID]
        if len(arduinos) == 1:
            return arduinos[0]
        if len(arduinos) > 1 and not explicita:
            raise BenchError(f"Mais de uma placa Arduino conectada ({arduinos}); use --port.")
        time.sleep(1)
    raise BenchError("Placa não encontrada na USB. Confira o cabo e o --port.")


def open_board(porta, timeout=10.0, espera=30):
    limite = time.time() + espera
    while True:
        try:
            return SerialBoard(porta, timeout)
        except Exception:
            if time.time() > limite:
                raise
            time.sleep(1)


def parse_kv(linha):
    info = {}
    for parte in linha.split(","):
        if "=" in parte:
            k, v = parte.split("=", 1)
            try:
                info[k] = int(v)
            except ValueError:
                try:
                    info[k] = float(v)
                except ValueError:
                    info[k] = v
    return info


'''
def wait_ready(board, timeout=25):
    """Espera o 'READY,INFO,...' impresso pelo sketch depois do boot. Devolve o dict ou None."""
    limite = time.time() + timeout
    board.set_timeout(2)
    while time.time() < limite:
        linha = board.readline().decode(errors="ignore").strip()
        if linha.startswith("READY,"):
            return parse_kv(linha)
        if linha.startswith("ERR"):
            return {"alloc": 0, "erro": linha}
    return None
'''

def wait_ready(board, timeout=30):
    limite = time.time() + timeout
    board.set_timeout(1)
    visto, prox = [], 0.0
    while time.time() < limite:
        if time.time() >= prox:
            try:
                board.write(b"INFO\n")
            except Exception:
                pass
            prox = time.time() + 1.5
        linha = board.readline().decode(errors="ignore").strip()
        if linha:
            visto.append(linha)
        if linha.startswith(("READY,", "INFO,")):
            return parse_kv(linha)
        if linha.startswith("ERR"):
            return {"alloc": 0, "erro": linha}
    print("      saída da placa antes do timeout:", visto[-15:] or "(nada recebido)")
    return None

def run_window(board, x, n_runs, timeout=180):
    """Envia 1 janela e pede n_runs execuções. Devolve (tempos_us, classe, scores)."""
    board.set_timeout(timeout)
    board.reset_input_buffer()
    board.write(f"RUN {n_runs}\n".encode())
    while True:
        linha = board.readline().decode(errors="ignore").strip()
        if linha == "GO":
            break
        if linha.startswith("ERR"):
            raise BenchError(linha)
        if not linha:
            raise BenchError("Sem resposta da placa ao comando RUN.")
    dados = np.ascontiguousarray(x, dtype="<f4").tobytes()
    for off in range(0, len(dados), CHUNK):
        board.write(dados[off:off + CHUNK])
        if board.read(1) != b".":
            raise BenchError("Placa não confirmou o bloco da janela (perda de dados na serial).")
    tempos, pred, scores = [], None, None
    while True:
        linha = board.readline().decode(errors="ignore").strip()
        if not linha:
            raise BenchError("Timeout esperando resultados da placa.")
        if linha.startswith("T,"):
            tempos.append(int(linha[2:]))
        elif linha.startswith("R,"):
            p = linha.split(",")
            pred, scores = int(p[1]), [float(v) for v in p[2:]]
        elif linha == "END":
            return tempos, pred, scores
        elif linha.startswith("ERR"):
            raise BenchError(linha)


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------

def f1_macro(y, p, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def estat_latencia(tempos_us):
    ms = np.asarray(tempos_us, dtype=float) / 1000.0
    return {"n_runs": len(ms), "lat_mean_ms": ms.mean(), "lat_sd_ms": ms.std(ddof=1) if len(ms) > 1 else 0.0,
            "lat_median_ms": float(np.median(ms)), "lat_p95_ms": float(np.percentile(ms, 95)),
            "lat_min_ms": ms.min(), "lat_max_ms": ms.max()}


# ----------------------------------------------------------------------------
# Build + boot de uma versão (com busca do menor arena que a placa aceita)
# ----------------------------------------------------------------------------

def build_and_boot(cfg, variante, arena_kb):
    """Compila, grava e espera o boot. Devolve (resultado, board|None)."""
    comp = compile_sketch(cfg, variante, arena_kb)
    if comp["status"] != "ok":
        return {**comp, "arena_kb": arena_kb}, None
    porta = find_port(cfg.port)
    upload_sketch(cfg, porta)
    time.sleep(4 if USA_JTAG else 0)
    porta = find_port(cfg.port)
    board = open_board(porta)
    info = wait_ready(board)
    if info is None:
        board.close()
        return {**comp, "status": "boot_failed", "arena_kb": arena_kb}, None
    if not info.get("alloc", 0):
        board.close()
        return {**comp, "status": "alloc_failed", "arena_kb": arena_kb, "info": info}, None
    return {**comp, "status": "ok", "arena_kb": arena_kb, "info": info}, board


def prepare_variant(cfg, variante, cache):
    """Descobre o menor arena que funciona (primeira sessão) ou reaproveita o valor já descoberto."""
    if variante in cache and cache[variante].get("falha"):
        return cache[variante]["falha"], None          # já se sabe que não cabe; não regrava a placa à toa
    if variante in cache:
        kb = cache[variante]["arena_kb"]
        res, board = build_and_boot(cfg, variante, kb)
        if res["status"] != "ok":
            raise BenchError(f"{variante}: a build que funcionou na 1ª sessão falhou agora ({res['status']}).")
        return res, board

    est = cfg.meta["variants"].get(variante, {})
    est_max = est.get("arena_max_KB", 128)
    est_min = est.get("arena_min_KB", 32)
    kb = int(math.ceil((est_max * 1.15 + 8) / 16) * 16)      # o arena real fica um pouco acima da estimativa
    if USA_JTAG:
        # No S3 o ESP-NN pede scratch extra além da estimativa do notebook:
        # começa folgado e deixa o "arena apertado" reduzir depois.
        kb = max(kb, 224)
    piso = max(16, int(est_min * 0.9))
    print(f"    [{variante}] procurando arena: começando em {kb} KB (estimativa do notebook: {est_min:.0f}-{est_max:.0f} KB)")
    while kb >= piso:
        res, board = build_and_boot(cfg, variante, kb)
        print(f"    [{variante}] arena {kb} KB -> {res['status']}")
        if res["status"] == "ok":
            usado = res["info"]["arena_used"]
            apertado = int(math.ceil((usado + 16) / 1024)) + 1   # arena_used + 16 B (recomendação do TFLM) + folga de 1 KB
            if apertado < kb:
                board.close()
                res2, board2 = build_and_boot(cfg, variante, apertado)
                print(f"    [{variante}] arena apertado {apertado} KB -> {res2['status']}")
                if res2["status"] == "ok":
                    res, board = res2, board2
                else:   # improvável; volta para a build folgada
                    res, board = build_and_boot(cfg, variante, kb)
            cache[variante] = {"arena_kb": res["arena_kb"]}
            return res, board
        if res["status"] == "no_fit_link" and res.get("overflow_bytes"):
            # O linker diz quanto estourou: a maior arena que ainda linka é kb - excesso.
            max_kb = (kb * 1024 - res["overflow_bytes"]) // 1024
            if est_max > max_kb:
                print(f"    [{variante}] só cabem ~{max_kb} KB de arena na RAM e o modelo precisa de pelo menos ~{est_max:.0f} KB")
                falha = {"status": "no_fit_ram", "arena_kb": kb, "arena_max_link_kb": max_kb,
                         "arena_min_est_kb": est_max}
                cache[variante] = {"falha": falha}
                return falha, None
            kb = min(kb - 16, max_kb - 16)   # deixa folga para heap/stack do Mbed OS
            continue
        if res["status"] == "alloc_failed":
            # Compila e boota, mas o arena não basta; arenas maiores não linkam/bootam: a versão não cabe.
            falha = {**res, "status": "no_fit_arena"}
            cache[variante] = {"falha": falha}
            return falha, None
        kb -= 16
    falha = {"status": "no_fit", "arena_kb": kb + 16}
    cache[variante] = {"falha": falha}
    return falha, None


# ----------------------------------------------------------------------------
# Medição
# ----------------------------------------------------------------------------

def measure_variant(cfg, variante, sessao, cache, X, y, host_pred, resultados):
    res, board = prepare_variant(cfg, variante, cache)
    linha = {"sessao": sessao, "variante": variante, "status": res["status"], "arena_kb": res.get("arena_kb")}
    tf_bytes = (cfg.export_dir / f"modelo_{variante.lower()}.tflite")
    linha["tflite_bytes"] = tf_bytes.stat().st_size if tf_bytes.exists() else None
    if board is None:
        linha.update({k: res[k] for k in ("arena_max_link_kb", "arena_min_est_kb") if k in res})
        print(f"    [{variante}] NÃO EXECUTÁVEL: {res['status']}")
        resultados["resumo"].append(linha)
        return

    try:
        info = res["info"]
        linha.update({k: res[k] for k in ("flash_bytes", "flash_max_bytes", "ram_bytes", "ram_max_bytes")})
        linha["flash_pct"] = 100 * res["flash_bytes"] / res["flash_max_bytes"]
        linha["ram_pct"] = 100 * res["ram_bytes"] / res["ram_max_bytes"]
        linha.update({"arena_used_bytes": info["arena_used"], "harness_bytes": info.get("harness_bytes"),
                      "in_type": info.get("in_type")})

        # --- latência: mesma janela, n execuções (as primeiras são aquecimento e são descartadas) ---
        #tempos, _, _ = run_window(board, X[cfg.latency_window], cfg.warmup + cfg.runs)
        tempos, _, _ = run_window(board, X[cfg.latency_window], cfg.warmup + cfg.runs,
                          timeout=1800 if USA_JTAG else 180)
        tempos = tempos[cfg.warmup:]
        linha.update(estat_latencia(tempos))
        for i, t in enumerate(tempos):
            resultados["latencia"].append({"sessao": sessao, "variante": variante, "run": i, "invoke_us": t})
        print(f"    [{variante}] latência {linha['lat_mean_ms']:.3f} ± {linha['lat_sd_ms']:.3f} ms "
              f"(n={linha['n_runs']}) | arena usado {info['arena_used'] / 1024:.1f} KB | "
              f"RAM {linha['ram_pct']:.1f}% | Flash {linha['flash_pct']:.1f}%")

        # --- acurácia na placa: só na primeira vez que a versão é medida ---
        if not cfg.skip_accuracy and variante not in resultados["acc_feita"]:
            preds, tempos_acc = [], []
            for i in range(len(X)):
                t, p, sc = run_window(board, X[i], 1)
                preds.append(p)
                tempos_acc.append(t[0])
                resultados["acuracia"].append({"variante": variante, "janela": i, "y_true": int(y[i]),
                                               "y_pred": p, "invoke_us": t[0],
                                               **{f"score_{c}": s for c, s in enumerate(sc)}})
                if (i + 1) % 50 == 0:
                    print(f"    [{variante}] acurácia na placa: {i + 1}/{len(X)} janelas")
            preds = np.array(preds)
            n_cls = len(cfg.meta["classes"])
            resultados["acc_feita"].add(variante)
            linha["acc_placa"] = float(np.mean(preds == y))
            linha["f1_placa"] = f1_macro(y, preds, n_cls)
            if host_pred is not None:
                linha["concordancia_tflite_pc"] = float(np.mean(preds == host_pred))
            hostinfo = cfg.meta["variants"].get(variante, {})
            linha["acc_tflite_pc"] = hostinfo.get("host_accuracy")
            print(f"    [{variante}] NA PLACA: acurácia {linha['acc_placa']:.4f} | F1-macro {linha['f1_placa']:.4f} "
                  f"| concordância com o TFLite do PC {linha.get('concordancia_tflite_pc', float('nan')):.4f}")
    finally:
        board.close()
    resultados["resumo"].append(linha)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def salvar(cfg, resultados):
    out = cfg.out
    pd.DataFrame(resultados["resumo"]).to_csv(out / "resumo_por_sessao.csv", index=False)
    if resultados["latencia"]:
        pd.DataFrame(resultados["latencia"]).to_csv(out / "latencia_bruta.csv", index=False)
    if resultados["acuracia"]:
        pd.DataFrame(resultados["acuracia"]).to_csv(out / "acuracia_na_placa.csv", index=False)


def resumo_final(cfg, resultados):
    resumo = pd.DataFrame(resultados["resumo"])
    lat = pd.DataFrame(resultados["latencia"])
    linhas = []
    for v in VARIANTES:
        sub = resumo[resumo["variante"] == v]
        if sub.empty:
            continue
        ok = sub[sub["status"] == "ok"]
        if ok.empty:
            linhas.append({"variante": v, "status": sub["status"].iloc[0], "arena_kb_tentado": sub["arena_kb"].iloc[0],
                           "arena_max_link_kb": sub["arena_max_link_kb"].iloc[0] if "arena_max_link_kb" in sub else np.nan,
                           "arena_min_est_kb": sub["arena_min_est_kb"].iloc[0] if "arena_min_est_kb" in sub else np.nan,
                           "tflite_bytes": sub["tflite_bytes"].iloc[0]})
            continue
        primeira = ok.iloc[0]
        ms = lat[lat["variante"] == v]["invoke_us"].to_numpy() / 1000.0
        linhas.append({
            "variante": v, "status": "ok", "sessoes": len(ok), "n_total": len(ms),
            "lat_mean_ms": ms.mean(), "lat_sd_ms": ms.std(ddof=1), "lat_median_ms": np.median(ms),
            "lat_p95_ms": np.percentile(ms, 95),
            "flash_bytes": primeira["flash_bytes"], "flash_pct": primeira["flash_pct"],
            "ram_bytes": primeira["ram_bytes"], "ram_pct": primeira["ram_pct"],
            "arena_used_bytes": primeira["arena_used_bytes"], "harness_bytes": primeira["harness_bytes"],
            "tflite_bytes": primeira["tflite_bytes"],
            "acc_placa": ok["acc_placa"].dropna().iloc[0] if "acc_placa" in ok and ok["acc_placa"].notna().any() else np.nan,
            "f1_placa": ok["f1_placa"].dropna().iloc[0] if "f1_placa" in ok and ok["f1_placa"].notna().any() else np.nan,
            "concordancia_tflite_pc": ok["concordancia_tflite_pc"].dropna().iloc[0]
            if "concordancia_tflite_pc" in ok and ok["concordancia_tflite_pc"].notna().any() else np.nan,
        })
    final = pd.DataFrame(linhas)
    final.to_csv(cfg.out / "resumo_final.csv", index=False)
    return final


def ambiente(cfg):
    def txt(args):
        p = run_cli(cfg.cli, args, check=False)
        return (p.stdout + p.stderr).strip()
    return {"data": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.platform(),
            "python": sys.version.split()[0], "arduino_cli": txt(["version"]),
            "cores": txt(["core", "list"]), "bibliotecas": txt(["lib", "list"]), "fqbn": cfg.fqbn,
            "args": vars(cfg.args)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True, help="pasta arduino_export gerada pelo notebook")
    ap.add_argument("--sketch-dir", default=str(Path(__file__).parent / "cnn_benchmark"))
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--variants", nargs="+", default=VARIANTES, choices=VARIANTES)
    ap.add_argument("--runs", type=int, default=1000, help="execuções cronometradas por configuração")
    ap.add_argument("--warmup", type=int, default=20, help="execuções iniciais descartadas")
    ap.add_argument("--sessions", type=int, default=1, help="repetições da campanha (ordem aleatória em cada uma)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latency-window", type=int, default=0, help="índice da janela de teste usada na latência")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--trigger-pin", type=int, default=None, help="pino em HIGH durante o Invoke (medição de energia futura)")
    ap.add_argument("--arduino-cli", default="arduino-cli")
    ap.add_argument("--fqbn", default=FQBN_PADRAO)
    ap.add_argument("--port", default=None)
    args = ap.parse_args(argv)

    class Cfg:
        pass
    cfg = Cfg()
    cfg.args = args
    cfg.export_dir, cfg.sketch_dir = Path(args.export_dir), Path(args.sketch_dir)
    cfg.out = Path(args.out)
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.build_dir = cfg.out / "build"
    cfg.cli, cfg.fqbn, cfg.port = args.arduino_cli, args.fqbn, args.port

    global USA_JTAG
    USA_JTAG = cfg.fqbn.startswith("esp32:")

    cfg.runs, cfg.warmup, cfg.latency_window = args.runs, args.warmup, args.latency_window
    cfg.skip_accuracy, cfg.trigger_pin = args.skip_accuracy, args.trigger_pin
    cfg.meta = json.loads((cfg.export_dir / "meta.json").read_text())
    cfg.window_len = cfg.meta["window_len"]

    X = np.load(cfg.export_dir / "X_test_cnn.npy")[..., 0]
    y = np.load(cfg.export_dir / "y_test_cnn.npy")
    resultados = {"resumo": [], "latencia": [], "acuracia": [], "acc_feita": set()}
    (cfg.out / "sessao_meta.json").write_text(json.dumps(ambiente(cfg), indent=2, ensure_ascii=False, default=str))

    cache = {}
    for s in range(1, args.sessions + 1):
        ordem = list(args.variants)
        random.Random(args.seed * 1000 + s).shuffle(ordem)
        print(f"\n=== Sessão {s}/{args.sessions} | ordem: {' -> '.join(ordem)} ===")
        for v in ordem:
            print(f"  -> {v}")
            host_pred_path = cfg.export_dir / f"pred_tflite_{v.lower()}.npy"
            host_pred = np.load(host_pred_path) if host_pred_path.exists() else None
            measure_variant(cfg, v, s, cache, X, y, host_pred, resultados)
            salvar(cfg, resultados)

    final = resumo_final(cfg, resultados)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Resumo final ===")
    print(final.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArquivos em: {cfg.out.resolve()}")
    return final


if __name__ == "__main__":
    main()
