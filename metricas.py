"""BER medida, BER teórica de cada esquema e eficiência espectral."""

import math

import numpy as np


def ber(enviados: np.ndarray, recebidos: np.ndarray) -> float:
    return float(np.count_nonzero(enviados != recebidos)) / len(enviados)


def q(x: float) -> float:
    """Probabilidade de uma gaussiana padrão ultrapassar x."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def ber_teorica(modulacao: str, codigo: str, ebn0_db: float) -> float:
    """BER em AWGN em função de g = Eb/N0 (linear), com receptor ótimo.

    BPSK (NRZ ou Manchester)  Q(√(2g))
    16-QAM + NRZ              [3Q(x) + 2Q(3x) - Q(5x)] / 4, com x = √(0,8g)
    16-QAM + Manchester       Q(√(1,6g))

    No último caso, os 2 chips de cada bit caem no mesmo eixo: só os pares 01 e 10 aparecem, ou seja,
    os níveis -1 e +3. A constelação encolhe para 4 pontos, com a distância 4/√10 entre os níveis.
    """
    g = 10 ** (ebn0_db / 10)
    if modulacao == "bpsk":
        return q(math.sqrt(2 * g))
    if codigo == "nrz":
        x = math.sqrt(0.8 * g)
        return (3 * q(x) + 2 * q(3 * x) - q(5 * x)) / 4
    return q(math.sqrt(1.6 * g))


def ebn0_necessario(modulacao: str, codigo: str, ber_alvo: float = 1e-5) -> float:
    """Eb/N0 (dB) que atinge a BER alvo pela teoria, por bisseção."""
    baixo, alto = -10.0, 40.0
    for _ in range(60):
        meio = (baixo + alto) / 2
        if ber_teorica(modulacao, codigo, meio) > ber_alvo:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def eficiencia_espectral(chips_por_simbolo: int, chips_por_bit: int) -> float:
    """Bits de informação por símbolo = bit/s/Hz, com 1 símbolo/s por Hz de banda."""
    return chips_por_simbolo / chips_por_bit


def limite_shannon_ebn0_db(eficiencia: float) -> float:
    """Menor Eb/N0 teoricamente possível para essa eficiência: (2^η - 1) / η."""
    return 10 * math.log10((2 ** eficiencia - 1) / eficiencia)
