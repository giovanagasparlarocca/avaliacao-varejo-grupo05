"""Etapa B — Algoritmo Genético para a política de preços do catálogo.

Cromossomo : vetor de n inteiros; gene i = índice do preço permitido do SKU i.
Seleção    : torneio de tamanho k (pressão seletiva ajustável, O(1) por pai).
Cruzamento : uniforme (cada gene herdado de um dos pais) com probabilidade p_cruzamento.
Mutação    : por gene, com probabilidade p_mutacao_gene, sorteia novo preço permitido.
Elitismo   : os E melhores passam intactos (fitness nunca piora entre gerações).
Parada     : max_geracoes ou 'paciencia' gerações sem melhoria > tolerancia.

Tempo  O(G · P · n)   (G gerações, P população, n SKUs)   — dominado pela avaliação do fitness.
Memória O(P · n).
"""
import numpy as np

from questao_01.modelo import fitness_individuo_loop, fitness_populacao


def _avaliar(T, pop, vetorizado):
    if vetorizado:
        return fitness_populacao(T, pop)
    return np.array([fitness_individuo_loop(T, g) for g in pop])


def _torneio(rng, fit, k, m):
    cand = rng.integers(0, len(fit), size=(m, k))
    return cand[np.arange(m), fit[cand].argmax(1)]


def executar_ga(T, cfg, seed):
    if cfg["populacao"] < 2 or cfg["elitismo"] >= cfg["populacao"]:
        raise ValueError("populacao deve ser >=2 e maior que o elitismo.")
    if not 0 <= cfg["p_mutacao_gene"] <= 1 or not 0 <= cfg["p_cruzamento"] <= 1:
        raise ValueError("probabilidades devem estar em [0,1].")
    rng = np.random.default_rng(seed)
    P, n, E = cfg["populacao"], T.n, cfg["elitismo"]
    vet = cfg.get("vetorizado", True)
    pop = rng.integers(0, T.n_gene, size=(P, n))
    hist_best, hist_mean = [], []
    melhor_f, melhor_g, sem_melhora, avaliacoes = -np.inf, None, 0, 0
    for ger in range(cfg["max_geracoes"]):
        fit = _avaliar(T, pop, vet)
        avaliacoes += P
        ordem = np.argsort(-fit)
        if fit[ordem[0]] > melhor_f + cfg["tolerancia"]:
            sem_melhora = 0
        else:
            sem_melhora += 1
        if fit[ordem[0]] > melhor_f:
            melhor_f, melhor_g = float(fit[ordem[0]]), pop[ordem[0]].copy()
        hist_best.append(melhor_f)
        hist_mean.append(float(fit.mean()))
        if sem_melhora >= cfg["paciencia"]:
            break
        m = P - E
        p1 = pop[_torneio(rng, fit, cfg["torneio"], m)]
        p2 = pop[_torneio(rng, fit, cfg["torneio"], m)]
        cruza = rng.random(m) < cfg["p_cruzamento"]
        mascara = rng.random((m, n)) < 0.5
        filhos = np.where(cruza[:, None] & mascara, p2, p1)
        mut = rng.random((m, n)) < cfg["p_mutacao_gene"]
        novos = rng.integers(0, T.n_gene, size=(m, n))
        filhos = np.where(mut, novos, filhos)
        pop = np.vstack([pop[ordem[:E]], filhos]) if E > 0 else filhos
    return {"genes": melhor_g, "fitness": melhor_f, "geracoes": len(hist_best),
            "hist_best": np.array(hist_best), "hist_mean": np.array(hist_mean),
            "avaliacoes": avaliacoes}
