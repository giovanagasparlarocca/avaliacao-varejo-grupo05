"""Etapa A — Brute Force: enumera TODAS as combinações de preços permitidos de um subconjunto.

Tempo  O(Π n_i · n)  = O(V^n · n)   (V = preços permitidos por SKU, n = SKUs do subconjunto)
Memória O(B · n)                    (B = tamanho do bloco; a enumeração é feita em blocos)
Domina: avaliação do fitness de cada combinação (soma de n termos).
"""
import numpy as np

from questao_01.modelo import fitness_populacao

LIMITE_COMBINACOES = 30_000_000


def forca_bruta(T, tamanho_bloco=400_000):
    n_g = [int(x) for x in T.n_gene]
    total = 1
    for v in n_g:
        total *= v
    if total > LIMITE_COMBINACOES:
        raise ValueError(f"{total} combinações excede o limite ({LIMITE_COMBINACOES}); reduza o subconjunto.")
    strides = []
    s = 1
    for v in n_g:
        strides.append(s)
        s *= v
    melhor_f, melhor_g, viaveis = -np.inf, None, 0
    for ini in range(0, total, tamanho_bloco):
        idx = np.arange(ini, min(ini + tamanho_bloco, total), dtype=np.int64)
        G = np.empty((len(idx), T.n), dtype=np.int16)
        for j in range(T.n):                       # decodificação em base mista
            G[:, j] = (idx // strides[j]) % n_g[j]
        f = fitness_populacao(T, G)
        un = T.un[np.arange(T.n)[None, :], G].sum(1)
        viaveis += int((un <= T.cap_pol + 1e-9).sum())
        k = int(f.argmax())
        if f[k] > melhor_f:
            melhor_f, melhor_g = float(f[k]), G[k].astype(int)
    return {"genes": melhor_g, "fitness": melhor_f, "avaliadas": total, "viaveis": viaveis}


def forca_bruta_itertools(T):
    """Versão didática/lenta (itertools.product) — usada nos testes para conferir a vetorizada."""
    import itertools
    from questao_01.modelo import fitness_individuo_loop
    melhor_f, melhor_g = -np.inf, None
    for g in itertools.product(*[range(int(v)) for v in T.n_gene]):
        f = fitness_individuo_loop(T, g)
        if f > melhor_f:
            melhor_f, melhor_g = f, np.array(g)
    return {"genes": melhor_g, "fitness": float(melhor_f)}
