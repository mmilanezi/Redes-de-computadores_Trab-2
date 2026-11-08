"""Confere cada bloco do núcleo e compara a BER simulada com a teoria. Uso: python validar.py"""

import math
import sys

import numpy as np

from canal import awgn
from codigo_linha import CODIGOS, manchester_codificar, manchester_pares_invalidos
from fonte import bits_aleatorios, bits_para_texto, texto_para_bits
from metricas import ber_teorica, ebn0_necessario
from modulacao import MODULACOES
from sistema import transmitir

N_BITS = 2_000_000
EBN0_TESTE = [0, 4, 8]
ESQUEMAS = [(m, c) for m in MODULACOES for c in CODIGOS]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(2026)
    resultados = []

    def caso(descricao: str, ok: bool, detalhe: str = "") -> None:
        resultados.append(bool(ok))
        print(f"  [{'OK' if ok else 'FALHOU'}] {descricao}" + (f"\n         {detalhe}" if detalhe else ""))

    print("BLOCOS DA CADEIA")
    a = texto_para_bits("A")
    caso("ASCII: 'A' = 01000001", a.tolist() == [0, 1, 0, 0, 0, 0, 0, 1])
    caso("Texto -> bits -> texto volta ao original",
         bits_para_texto(texto_para_bits("Redes de Computadores 2026!")) == "Redes de Computadores 2026!")
    try:
        texto_para_bits("Transmissão")
        caso("Mensagem com acento é recusada", False)
    except ValueError:
        caso("Mensagem com acento é recusada", True)

    caso("Manchester: bits 1 0 -> chips 01 10",
         manchester_codificar(np.array([1, 0], dtype=np.uint8)).tolist() == [0, 1, 1, 0])
    caso("Manchester: par inválido (11) é detectado",
         manchester_pares_invalidos(np.array([1.0, 1.0, -1.0, 1.0])) == 1)

    for nome, mod in MODULACOES.items():
        chips = bits_aleatorios(400_000, rng)
        es = np.mean(np.abs(mod.modular(chips)) ** 2)
        caso(f"{mod.nome}: energia média por símbolo = 1", abs(es - 1) < 0.01, f"medida: {es:.4f}")

    tabela = MODULACOES["16qam"].modular(np.array(
        [(i >> b) & 1 for i in range(16) for b in (3, 2, 1, 0)], dtype=np.uint8))
    caso("16-QAM: 16 pontos distintos", len(np.unique(np.round(tabela, 6))) == 16)
    vizinhos_ok = all(bin(i ^ j).count("1") == 1
                      for i in range(16) for j in range(16)
                      if abs(abs(tabela[i] - tabela[j]) * math.sqrt(10) - 2) < 1e-6)
    caso("16-QAM: vizinhos diferem em um único chip (Gray)", vizinhos_ok)

    simbolos = np.ones(1_000_000, dtype=complex)
    medido = -10 * np.log10(np.mean(np.abs(awgn(simbolos, 10, rng) - simbolos) ** 2))
    caso("AWGN: SNR medida = SNR pedida (10 dB)", abs(medido - 10) < 0.05, f"medida: {medido:.3f} dB")

    print("\nCADEIA COMPLETA SEM RUÍDO (BER tem de ser 0, também com bits que não fecham um símbolo)")
    for m, c in ESQUEMAS:
        erros = max(transmitir(bits_aleatorios(n, rng), m, c, None, rng).ber for n in (1, 7, 1001))
        caso(f"{MODULACOES[m].nome} + {CODIGOS[c].nome}", erros == 0)

    print(f"\nBER SIMULADA x TEORIA ({N_BITS:,} bits; tolerância de 4 desvios-padrão)".replace(",", "."))
    print(f"  {'Eb/N0':>12}{'simulada':>12}{'teórica':>12}")
    for m, c in ESQUEMAS:
        print(f"  {MODULACOES[m].nome} + {CODIGOS[c].nome}")
        for e in EBN0_TESTE:
            bits = bits_aleatorios(N_BITS, rng)
            sim = transmitir(bits, m, c, e, rng).ber
            teo = ber_teorica(m, c, e)
            ok = abs(sim - teo) <= 4 * math.sqrt(teo * (1 - teo) / N_BITS)
            resultados.append(ok)
            print(f"  {e:>9} dB{sim:>12.3e}{teo:>12.3e}  [{'OK' if ok else 'FALHOU'}]")

    print("\nCONSTATAÇÃO: 16-QAM + Manchester")
    chips = CODIGOS["manchester"].codificar(bits_aleatorios(100_000, rng))
    pontos = len(np.unique(np.round(MODULACOES["16qam"].modular(chips), 6)))
    caso("A cadeia usa só 4 dos 16 pontos da constelação", pontos == 4, f"pontos usados: {pontos}")

    print("\nEb/N0 teórico para BER = 10^-5")
    for m, c in ESQUEMAS:
        print(f"  {MODULACOES[m].nome + ' + ' + CODIGOS[c].nome:<24}{ebn0_necessario(m, c):6.2f} dB")

    print(f"\nResultado: {sum(resultados)}/{len(resultados)} verificações OK")
    sys.exit(0 if all(resultados) else 1)


if __name__ == "__main__":
    main()
