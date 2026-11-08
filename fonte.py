"""Fonte de dados: converte mensagens ASCII em bits e bits em texto."""

import numpy as np

BITS_POR_CARACTERE = 8


def texto_para_bits(texto: str) -> np.ndarray:
    """Cada caractere vira 8 bits, do mais significativo para o menos significativo."""
    invalidos = sorted({c for c in texto if ord(c) > 127})
    if invalidos:
        raise ValueError(f"a mensagem deve ser ASCII (sem acentos); encontrado: {' '.join(invalidos)}")
    return np.unpackbits(np.frombuffer(texto.encode("ascii"), dtype=np.uint8))


def bits_para_texto(bits: np.ndarray) -> str:
    """Remonta o texto; bytes que não são ASCII imprimível (erros do canal) aparecem como '·'."""
    inteiros = len(bits) - len(bits) % BITS_POR_CARACTERE
    valores = np.packbits(bits[:inteiros].astype(np.uint8))
    return "".join(chr(v) if 32 <= v <= 126 or v == 10 else "·" for v in valores)


def bits_aleatorios(quantidade: int, rng: np.random.Generator) -> np.ndarray:
    """Bits equiprováveis, usados nas curvas de BER (que exigem milhões de bits)."""
    return rng.integers(0, 2, quantidade, dtype=np.uint8)
