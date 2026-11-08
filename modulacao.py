"""Modulação digital em banda base equivalente: cada símbolo é um número complexo (I + jQ).

  BPSK    1 chip por símbolo:  0 -> -1    1 -> +1
  16-QAM  4 chips por símbolo: os 2 primeiros escolhem o nível de I, os 2 últimos o de Q.
          Código de Gray por eixo: 00 -> -3   01 -> -1   11 -> +1   10 -> +3
          (vizinhos diferem em um único chip, então o erro mais comum custa um só chip).

As constelações têm energia média por símbolo Es = 1, para que a SNR tenha o mesmo significado nas
duas. Os demoduladores devolvem um valor suave por chip (ver codigo_linha.py).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

ESCALA_16QAM = np.sqrt(10)   # a energia média da grade {±1, ±3} + j{±1, ±3} é 10
NIVEIS_GRAY = np.array([-3, -1, 3, 1])   # índice = 2·chip1 + chip2 (00, 01, 10, 11)


def completar(chips: np.ndarray, multiplo: int) -> np.ndarray:
    """Acrescenta zeros para fechar o último símbolo; o receptor descarta esse excesso."""
    return np.concatenate([chips, np.zeros(-len(chips) % multiplo, dtype=np.uint8)])


def bpsk_modular(chips: np.ndarray) -> np.ndarray:
    return (2.0 * chips - 1.0).astype(complex)


def bpsk_demodular(recebidos: np.ndarray) -> np.ndarray:
    return recebidos.real


def qam16_modular(chips: np.ndarray) -> np.ndarray:
    c = completar(chips, 4).reshape(-1, 4).astype(int)
    i = NIVEIS_GRAY[2 * c[:, 0] + c[:, 1]]
    q = NIVEIS_GRAY[2 * c[:, 2] + c[:, 3]]
    return (i + 1j * q) / ESCALA_16QAM


def qam16_demodular(recebidos: np.ndarray) -> np.ndarray:
    # Em cada eixo: o 1º chip diz o sinal do nível; o 2º diz se é interno (|y| < 2) ou externo.
    r = recebidos * ESCALA_16QAM
    suaves = np.empty((len(r), 4))
    suaves[:, 0] = r.real
    suaves[:, 1] = 2 - np.abs(r.real)
    suaves[:, 2] = r.imag
    suaves[:, 3] = 2 - np.abs(r.imag)
    return suaves.ravel()


@dataclass(frozen=True)
class Modulacao:
    nome: str
    chips_por_simbolo: int
    modular: Callable[[np.ndarray], np.ndarray]
    demodular: Callable[[np.ndarray], np.ndarray]


MODULACOES = {
    "bpsk": Modulacao("BPSK", 1, bpsk_modular, bpsk_demodular),
    "16qam": Modulacao("16-QAM", 4, qam16_modular, qam16_demodular),
}
