"""Codificação de canal (código de linha): transforma bits em "chips" binários.

  NRZ        (referência, sem redundância)  1 chip por bit:   0 -> 0      1 -> 1
  Manchester (convenção IEEE 802.3)         2 chips por bit:  0 -> 1 0    1 -> 0 1

Os decodificadores recebem valores suaves: número positivo = chip provavelmente 1, negativo =
provavelmente 0, e quanto maior o módulo, maior a confiança. Isso permite ao Manchester comparar as
duas metades do bit, aproveitando a energia dos dois chips.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def nrz_codificar(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint8)


def nrz_decodificar(suaves: np.ndarray) -> np.ndarray:
    return (suaves > 0).astype(np.uint8)


def manchester_codificar(bits: np.ndarray) -> np.ndarray:
    chips = np.empty(2 * len(bits), dtype=np.uint8)
    chips[0::2] = 1 - bits
    chips[1::2] = bits
    return chips


def manchester_decodificar(suaves: np.ndarray) -> np.ndarray:
    # bit 1 = (0, 1): a segunda metade é maior que a primeira
    return (suaves[1::2] > suaves[0::2]).astype(np.uint8)


def manchester_pares_invalidos(suaves: np.ndarray) -> int:
    """Pares 00 ou 11 não existem no código: sua presença denuncia um erro do canal."""
    duros = suaves > 0
    return int(np.count_nonzero(duros[0::2] == duros[1::2]))


@dataclass(frozen=True)
class CodigoLinha:
    nome: str
    chips_por_bit: int
    codificar: Callable[[np.ndarray], np.ndarray]
    decodificar: Callable[[np.ndarray], np.ndarray]
    pares_invalidos: Callable[[np.ndarray], int] | None = None


CODIGOS = {
    "nrz": CodigoLinha("NRZ", 1, nrz_codificar, nrz_decodificar),
    "manchester": CodigoLinha("Manchester", 2, manchester_codificar, manchester_decodificar,
                              manchester_pares_invalidos),
}
