"""Etapa B — Programação Dinâmica (bottom-up) para a trajetória semanal de preços.

Um problema por SKU (a validação de caixa/dívidas é agregada e feita pelo Monte Carlo).
  Estado     (t, s)  t = semana (0..T-1), s = estoque remanescente (0..estoque_inicial)
  Decisão    k ∈ {0..K-1} nível de preço, ou k = K  'pausar vendas' (demanda 0)
  Transição  s' = s - min(s, D),  D ~ distribuição empírica (J quantis do Monte Carlo) para (t, k)
  Objetivo   V_t(s) = max_k  (1/J) Σ_j [ w_t · margem_k · min(s, D_j) + V_{t+1}(s - min(s, D_j)) ]
             V_T(s) = valor_residual · custo · s
  w_t        peso de urgência (>1 nas semanas que antecedem dívidas): empurra caixa para cedo.

Reutilização de subproblemas: V_{t+1}(·) é calculado UMA vez por (t+1, s') e consultado por todos os
(s, k, j) da semana t. Sem memoização o número de caminhos seria (K+1)^T; com ela, T·(S+1) estados.

Tempo  O(N · T · S · K · J)     Memória O(N · T · S)    (N SKUs, T semanas, S estoque máx., K níveis, J quantis)
"""
import numpy as np


def pesos_urgencia(T, prazos, lam, desconto=0.0):
    """w_t = (1 + lam * prazos_restantes/total_de_prazos) * (1-desconto)^t."""
    w = np.ones(T)
    for t in range(T):
        restantes = sum(1 for p in prazos if p >= t + 1)
        w[t] = (1 + lam * restantes / len(prazos)) * (1 - desconto) ** t   # caixa mais cedo vale um pouco mais
    return w


def resolver_sku(q, mm, S0, w, valor_residual_un):
    """q: (T,K,J) demandas; mm: (K,) margem unitária; devolve V (T+1,S0+1) e POL (T,S0+1)."""
    T, K, J = q.shape
    V = np.zeros((T + 1, S0 + 1))
    V[T] = valor_residual_un * np.arange(S0 + 1)
    POL = np.zeros((T, S0 + 1), dtype=np.int8)
    s = np.arange(S0 + 1)[:, None]
    for t in range(T - 1, -1, -1):
        melhor = np.full(S0 + 1, -np.inf)
        pol = np.zeros(S0 + 1, dtype=np.int8)
        for k in range(K):
            vendido = np.minimum(s, q[t, k][None, :])                       # (S0+1, J)
            val = (w[t] * mm[k] * vendido + V[t + 1][s - vendido]).mean(1)  # reutiliza V[t+1]
            melhor_k = val > melhor
            melhor = np.where(melhor_k, val, melhor)
            pol = np.where(melhor_k, k, pol).astype(np.int8)
        pausa = V[t + 1] > melhor + 1e-9 + 1e-3 * np.abs(melhor)             # opção K: só pausa se claramente melhor
        melhor = np.where(pausa, V[t + 1], melhor)
        pol = np.where(pausa, K, pol).astype(np.int8)
        pol[0] = K                                                          # sem estoque: nada a precificar
        V[t], POL[t] = melhor, pol
    return V, POL


def resolver_dp(inst, est, lam_urgencia):
    w = pesos_urgencia(inst.T, inst.prazos, lam_urgencia, inst.params.get("desconto_semanal", 0.0))
    smax = int(inst.est0.max())
    POL = np.full((inst.N, inst.T, smax + 1), inst.K, dtype=np.int8)
    V0 = np.zeros(inst.N)
    vres = inst.params["valor_residual_pct_custo"] * inst.custo
    for i in range(inst.N):
        V, pol = resolver_sku(est.q[i].astype(np.int64), est.mm[i], int(inst.est0[i]), w, vres[i])
        POL[i, :, :int(inst.est0[i]) + 1] = pol
        V0[i] = V[0, inst.est0[i]]
    return {"POL": POL, "V0": V0, "pesos": w}


def trajetoria_nominal(inst, est, POL):
    """Caminho 'esperado': segue a política a partir do estoque inicial, descontando a venda esperada."""
    N, T = inst.N, inst.T
    nivel = np.zeros((N, T), int); estoque = np.zeros((N, T), int); vend = np.zeros((N, T))
    for i in range(N):
        s = int(inst.est0[i])
        for t in range(T):
            k = int(POL[i, t, s])
            nivel[i, t], estoque[i, t] = k, s
            if k < inst.K:
                v = float(np.minimum(s, est.q[i, t, k]).mean())
                vend[i, t] = v
                s = max(0, s - int(round(v)))
    return {"nivel": nivel, "estoque_inicio": estoque, "vendas_esperadas": vend}
