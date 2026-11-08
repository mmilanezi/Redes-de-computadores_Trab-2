"""Cadeia completa: bits -> código de linha -> modulação -> AWGN -> demodulação -> decodificação.

Uso:
  python sistema.py "Redes de Computadores" --modulacao bpsk --codigo nrz --ebn0 4
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from canal import awgn, ebn0_para_snr_db
from codigo_linha import CODIGOS
from fonte import bits_para_texto, texto_para_bits
from metricas import ber, eficiencia_espectral
from modulacao import MODULACOES


@dataclass
class Resultado:
    bits_recebidos: np.ndarray
    ber: float
    snr_db: float
    eficiencia: float
    n_simbolos: int
    pares_invalidos: int | None


def transmitir(bits: np.ndarray, modulacao: str, codigo: str, ebn0_db: float | None,
               rng: np.random.Generator) -> Resultado:
    """Passa os bits pela cadeia inteira. ebn0_db=None transmite sem ruído."""
    mod, cod = MODULACOES[modulacao], CODIGOS[codigo]
    eficiencia = eficiencia_espectral(mod.chips_por_simbolo, cod.chips_por_bit)

    chips = cod.codificar(bits)
    simbolos = mod.modular(chips)
    if ebn0_db is None:
        snr_db, recebidos = float("inf"), simbolos
    else:
        snr_db = float(ebn0_para_snr_db(ebn0_db, eficiencia))
        recebidos = awgn(simbolos, snr_db, rng)
    suaves = mod.demodular(recebidos)[:len(chips)]      # descarta os chips de preenchimento
    bits_recebidos = cod.decodificar(suaves)

    invalidos = cod.pares_invalidos(suaves) if cod.pares_invalidos else None
    return Resultado(bits_recebidos, ber(bits, bits_recebidos), snr_db, eficiencia,
                     len(simbolos), invalidos)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Transmite uma mensagem ASCII por um canal ruidoso")
    ap.add_argument("mensagem")
    ap.add_argument("--modulacao", choices=MODULACOES, default="bpsk")
    ap.add_argument("--codigo", choices=CODIGOS, default="nrz")
    ap.add_argument("--ebn0", type=float, default=None, help="Eb/N0 em dB (omitido: sem ruído)")
    ap.add_argument("--semente", type=int, default=2026)
    args = ap.parse_args()

    try:
        bits = texto_para_bits(args.mensagem)
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    r = transmitir(bits, args.modulacao, args.codigo, args.ebn0, np.random.default_rng(args.semente))
    print(f"Esquema   : {MODULACOES[args.modulacao].nome} + {CODIGOS[args.codigo].nome} "
          f"({r.eficiencia:g} bit/s/Hz)")
    if args.ebn0 is None:
        print("Canal     : sem ruído")
    else:
        print(f"Canal     : AWGN, Eb/N0 = {args.ebn0:g} dB (SNR = {r.snr_db:.2f} dB)")
    print(f"Enviado   : {args.mensagem}")
    print(f"Recebido  : {bits_para_texto(r.bits_recebidos)}")
    print(f"Bits      : {len(bits)} em {r.n_simbolos} símbolos; "
          f"{np.count_nonzero(bits != r.bits_recebidos)} errados (BER = {r.ber:.2e})")
    if r.pares_invalidos is not None:
        print(f"Detecção  : {r.pares_invalidos} pares Manchester inválidos")


if __name__ == "__main__":
    main()
