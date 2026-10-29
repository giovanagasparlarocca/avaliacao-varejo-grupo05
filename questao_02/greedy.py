"""Etapa C — Greedy semanal: reparte o orçamento de 'destaques' (mídia/frete promocional) entre os SKUs.

O plano de preços da DP orienta a decisão: para cada (semana t, SKU i) usamos o nível de preço e o estoque
NOMINAL da trajetória da DP. Um destaque (no máx. 1 por SKU/semana):
    aumenta a demanda em uplift_i = min(uplift_max, uplift_base · elasticidade_i)
    custa  custo_destaque_pct · preço · demanda_média
    benefício = (vendas esperadas com destaque - sem destaque) · margem unitária   (limitado pelo estoque!)
Ordena por benefício/custo e compra enquanto couber no orçamento da semana e a razão >= razao_minima.

Tempo  O(T · N log N)    Memória O(T · N)
"""
import numpy as np


def uplift_por_sku(inst):
    p = inst.params
    return np.minimum(p["uplift_max"], p["uplift_base"] * inst.e)


def alocar_destaques(inst, est, POL, traj, orcamento_semanal=None):
    orc = inst.orc_sem if orcamento_semanal is None else orcamento_semanal
    if orc < 0:
        raise ValueError("orcamento negativo.")
    p = inst.params
    up = uplift_por_sku(inst)
    promo = np.zeros((inst.T, inst.N), bool)
    custo_promo = np.zeros((inst.T, inst.N))
    gasto_sem = np.zeros(inst.T)
    benef_sem = np.zeros(inst.T)
    for t in range(inst.T):
        cand = []
        for i in range(inst.N):
            k = int(traj["nivel"][i, t])
            s = int(traj["estoque_inicio"][i, t])
            if k >= inst.K or s <= 0:
                continue
            qq = est.q[i, t, k]
            e0 = np.minimum(s, qq).mean()
            e1 = np.minimum(s, np.rint(qq * (1 + up[i]))).mean()
            benef = (e1 - e0) * est.mm[i, k]
            custo = p["custo_destaque_pct"] * inst.p0[i] * inst.niv[k] * max(est.mu_dem[i, t, k], 1e-9)
            if custo > 0 and benef / custo >= p["razao_minima"]:
                cand.append((benef / custo, i, benef, custo))
        cand.sort(reverse=True)
        for r, i, benef, custo in cand:
            if gasto_sem[t] + custo <= orc + 1e-9:
                promo[t, i] = True
                custo_promo[t, i] = custo
                gasto_sem[t] += custo
                benef_sem[t] += benef
    return {"promo": promo, "custo_promo": custo_promo, "uplift": up,
            "gasto_semanal": gasto_sem, "beneficio_esperado_semanal": benef_sem}
