"""Canal AWGN e controle da relação sinal-ruído.

  SNR   = Es/N0: energia por SÍMBOLO sobre a densidade de ruído
  Eb/N0 = energia por BIT de informação sobre a densidade de ruído

Eb/N0 é a medida justa para comparar esquemas que carregam quantidades diferentes de bits por símbolo:
  SNR (dB) = Eb/N0 (dB) + 10·log10(bits de informação por símbolo)
"""

import numpy as np


def ebn0_para_snr_db(ebn0_db: float, bits_por_simbolo: float) -> float:
    return ebn0_db + 10 * np.log10(bits_por_simbolo)


def snr_para_ebn0_db(snr_db: float, bits_por_simbolo: float) -> float:
    return snr_db - 10 * np.log10(bits_por_simbolo)


def awgn(simbolos: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Soma ruído gaussiano complexo com potência total N0, supondo Es = 1 (N0/2 em cada eixo)."""
    n0 = 10 ** (-snr_db / 10)
    ruido = rng.normal(size=len(simbolos)) + 1j * rng.normal(size=len(simbolos))
    return simbolos + np.sqrt(n0 / 2) * ruido
