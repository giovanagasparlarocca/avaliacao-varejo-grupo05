"""Etapas A e D — Monte Carlo: (A) estima demanda/caixa por nível de preço; (D) valida o plano completo.

Estimativa   Tempo O(S·T·N·K) + ordenação p/ quantis O(T·S·N·K·log S)   Memória O(S·T·N) (choques) + O(S·N·K) por semana
Validação    Tempo O(S·T·N)                                             Memória O(S·T·N) (choques) + O(S·N)
(S cenários, T semanas, N SKUs, K níveis de preço.)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Choques:
    fd: np.ndarray    # (S,T,N) choque de demanda
    comp: np.ndarray  # (S,T,N) preço do concorrente relativo à base
    fl: np.ndarray    # (S,T)   choque de custo logístico


def gerar_choques(inst, S, seed):
    if S < 1:
        raise ValueError("S deve ser >= 1")
    rng = np.random.default_rng(seed)
    zd = rng.standard_normal((S, inst.T, inst.N))
    zc = rng.standard_normal((S, inst.T, inst.N))
    zl = rng.standard_normal((S, inst.T))
    fd = np.exp(inst.sig_d * zd - 0.5 * inst.sig_d ** 2)
    comp = inst.params["fator_concorrente_inicial"] * np.exp(
        np.cumsum(inst.sig_c * zc - 0.5 * inst.sig_c ** 2, axis=1))
    fl = np.exp(inst.sig_l * zl - 0.5 * inst.sig_l ** 2)
    return Choques(fd, comp, fl)


@dataclass
class Estatisticas:
    q: np.ndarray        # (N,T,K,J) quantis inteiros da demanda semanal
    mu_dem: np.ndarray   # (N,T,K)   demanda média
    mm: np.ndarray       # (N,K)     margem unitária média
    por_nivel: pd.DataFrame


def estimar(inst, choques, seed, J=None):
    J = J or inst.params["J_quantis"]
    S = choques.fd.shape[0]
    rng = np.random.default_rng(seed)
    g = inst.params["gamma_cruzada"]
    preco = inst.p0[:, None] * inst.niv[None, :]
    mm = preco * (1 - inst.tax[:, None]) - inst.custo[:, None] - inst.lg[:, None]
    base_k = inst.base[:, None] * inst.niv[None, :] ** (-inst.e[:, None])           # (N,K)
    probs = (np.arange(J) + 0.5) / J
    q = np.empty((inst.N, inst.T, inst.K, J), dtype=np.int32)
    mu = np.empty((inst.N, inst.T, inst.K))
    caixa_k = np.zeros((S, inst.K))
    for t in range(inst.T):                                          # uma semana por vez: memória O(S·N·K)
        lam = base_k[None] * (choques.comp[:, t, :] ** g * choques.fd[:, t, :])[:, :, None]
        D = rng.poisson(lam)
        q[:, t] = np.rint(np.quantile(D, probs, axis=0)).astype(np.int32).transpose(1, 2, 0)
        mu[:, t] = D.mean(0)
        marg = (preco * (1 - inst.tax[:, None]) - inst.custo[:, None])[None] - \
               inst.lg[None, :, None] * choques.fl[:, t][:, None, None]          # (S,N,K)
        caixa_k += (D * marg).sum(1) / inst.T
    tab = pd.DataFrame({
        "nivel_preco": inst.niv,
        "demanda_semanal_media_total": mu.mean(1).sum(0),
        "caixa_semanal_media_sem_restricao": caixa_k.mean(0),
        "caixa_semanal_p10": np.percentile(caixa_k, 10, axis=0),
        "caixa_semanal_p90": np.percentile(caixa_k, 90, axis=0),
        "desvio_padrao_caixa": caixa_k.std(0),
    })
    return Estatisticas(q, mu, mm, tab)


def simular_plano(inst, POL, promo, up, custo_promo, choques, seed):
    """Valida um plano completo (política de preços por estoque + destaques) em cenários dados."""
    S = choques.fd.shape[0]
    rng = np.random.default_rng(seed)
    g = inst.params["gamma_cruzada"]
    estq = np.tile(inst.est0.astype(np.int64), (S, 1))
    bal = np.full(S, inst.caixa_ini, float)
    ar = np.arange(inst.N)[None, :]
    op = np.zeros(S); rec = np.zeros(S); promo_tot = np.zeros(S)
    pagas = {}
    for t in range(inst.T):
        k = POL[ar, t, estq]
        pausa = k >= inst.K                      # decisão K = pausar vendas do SKU na semana
        k = np.minimum(k, inst.K - 1)
        niv = inst.niv[k]
        preco = inst.p0[None, :] * niv
        lam = inst.base[None, :] * niv ** (-inst.e[None, :]) * choques.comp[:, t, :] ** g * choques.fd[:, t, :]
        lam = lam * (1 + up[None, :] * promo[t][None, :]) * (~pausa)
        vendas = np.minimum(estq, rng.poisson(lam))
        marg = preco * (1 - inst.tax[None, :]) - inst.custo[None, :] - inst.lg[None, :] * choques.fl[:, t][:, None]
        op_t = (vendas * marg).sum(1)
        cp = float(custo_promo[t].sum())
        op += op_t; promo_tot += cp; rec += (vendas * preco).sum(1)
        bal += op_t - cp
        estq -= vendas
        semana = t + 1
        if semana in inst.prazos:
            d = inst.dividas[inst.prazos.index(semana)]
            ok = bal >= d
            bal = bal - d * ok
            pagas[semana] = ok
    todas = np.logical_and.reduce(list(pagas.values()))
    perc = inst.params["percentil_risco"]
    return {
        "prob_pagamento": {w: float(v.mean()) for w, v in pagas.items()},
        "prob_todas": float(todas.mean()),
        "saldo_final_medio": float(bal.mean()),
        "saldo_final_percentil_risco": float(np.percentile(bal, perc)),
        "percentil_risco": perc,
        "caixa_operacional_medio": float(op.mean()),
        "custo_promocional_medio": float(promo_tot.mean()),
        "margem_pct": float(100 * op.sum() / rec.sum()) if rec.sum() > 0 else 0.0,
        "estoque_final_medio_un": float(estq.sum(1).mean()),
        "estoque_final_medio_custo": float((estq * inst.custo[None, :]).sum(1).mean()),
        "cenarios_falha": int((~todas).sum()),
        "cenarios_total": int(S),
        "falhas_por_prazo": {w: int((~v).sum()) for w, v in pagas.items()},
        "_saldo": bal, "_estoque_final": estq,
    }
