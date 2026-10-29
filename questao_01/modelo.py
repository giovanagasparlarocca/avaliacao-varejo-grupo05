"""Modelo econômico da Questão 1 (Operação Caixa 90).

Para cada SKU i e nível de preço k:
    preço        p = preco_atual * m_k
    demanda      d = demanda_base_90d * m_k ** (-elasticidade)
    unidades     u = min(estoque, round(d))
    margem unit. mu = p*(1-taxa_marketplace) - custo_unitario - custo_logistico
    caixa        = u * mu
    carregamento = (estoque-u) * custo_unitario * taxa_carregamento * (1 + idade/180)
    fitness      = soma_i (caixa - carregamento) - penalidade * excesso de unidades acima da capacidade

Restrições: o piso de margem (mu/p >= margem_minima) define os preços PERMITIDOS de cada SKU;
a capacidade logística da política acopla os SKUs (sem ela o problema seria separável por SKU).
"""
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

COLUNAS = ["sku", "categoria", "custo_unitario", "preco_atual", "estoque", "demanda_base_90d",
           "elasticidade", "taxa_marketplace", "custo_logistico", "idade_estoque_dias"]
PARAMS_ARQ = Path(__file__).resolve().parent / "parametros_q1.json"


def carregar_parametros(caminho=None, sobrescritas=None):
    with open(caminho or PARAMS_ARQ, "r", encoding="utf-8") as f:
        p = json.load(f)
    for k, v in (sobrescritas or {}).items():
        if v is not None:
            p[k] = v
    return p


def validar_entrada(df, config):
    faltam = [c for c in COLUNAS if c not in df.columns]
    if faltam:
        raise ValueError(f"Colunas ausentes no catálogo: {faltam}")
    if len(df) == 0:
        raise ValueError("Catálogo vazio.")
    for c in ["custo_unitario", "preco_atual", "estoque", "demanda_base_90d", "elasticidade"]:
        if (df[c] <= 0).any():
            raise ValueError(f"Valores não positivos em {c}.")
    niveis = config.get("niveis_preco_relativos", [])
    if len(niveis) == 0 or any(m <= 0 for m in niveis):
        raise ValueError("niveis_preco_relativos inválido.")
    if not 0 <= config["margem_minima_percentual"] < 1:
        raise ValueError("margem_minima_percentual deve estar em [0,1).")
    if config["orcamento_promocional"] < 0:
        raise ValueError("orcamento_promocional negativo.")
    return True


def aplicar_cenario(df, config, meta_fator=1.0, orcamento_fator=1.0, sku_indisponivel=()):
    """Cenário-surpresa por parâmetros: meta/orçamento escalados e SKUs removidos."""
    cfg = copy.deepcopy(config)
    cfg["meta_caixa_90d"] = cfg["meta_caixa_90d"] * meta_fator
    cfg["orcamento_promocional"] = cfg["orcamento_promocional"] * orcamento_fator
    df = df[~df["sku"].isin(list(sku_indisponivel))].reset_index(drop=True)
    return df, cfg


def replicar_catalogo(df, fator):
    """Replica o catálogo (apenas para benchmarks de escala)."""
    partes = []
    for r in range(fator):
        d = df.copy()
        d["sku"] = d["sku"] + f"#{r + 1}"
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


@dataclass
class Tabelas:
    skus: np.ndarray
    n: int
    K: int
    n_gene: np.ndarray      # nº de preços permitidos (>=1; 1 = SKU sem preço viável)
    fit: np.ndarray         # (n,K) caixa - carregamento
    un: np.ndarray
    cash: np.ndarray
    rec: np.ndarray
    carr: np.ndarray
    preco: np.ndarray
    mu: np.ndarray
    nivel: np.ndarray       # índice global do nível (-1 = SKU não vendável)
    niveis: np.ndarray
    est: np.ndarray
    e: np.ndarray
    custo: np.ndarray
    carry_unit: np.ndarray
    base_units: np.ndarray
    min_units: np.ndarray
    cap_pol: float
    cap_total: float
    penal: float
    params: dict


_POR_SKU = ["skus", "n_gene", "fit", "un", "cash", "rec", "carr", "preco", "mu", "nivel",
            "est", "e", "custo", "carry_unit", "base_units", "min_units"]


def _capacidades(base_units, min_units, params):
    cap_pol = max(params["cap_frac_politica"] * base_units.sum(), min_units.sum())
    cap_tot = max(params["cap_frac_total"] * base_units.sum(), cap_pol)
    return float(cap_pol), float(cap_tot)


def construir_tabelas(df, config, params):
    validar_entrada(df, config)
    n = len(df)
    niv = np.asarray(config["niveis_preco_relativos"], float)
    p0 = df["preco_atual"].to_numpy(float)
    custo = df["custo_unitario"].to_numpy(float)
    tax = df["taxa_marketplace"].to_numpy(float)
    log = df["custo_logistico"].to_numpy(float) * params["fator_logistico"]
    est = df["estoque"].to_numpy(float)
    dem0 = df["demanda_base_90d"].to_numpy(float)
    e = df["elasticidade"].to_numpy(float)
    idade = df["idade_estoque_dias"].to_numpy(float)

    preco = p0[:, None] * niv[None, :]
    mu = preco * (1 - tax[:, None]) - custo[:, None] - log[:, None]
    dem = dem0[:, None] * niv[None, :] ** (-e[:, None])
    un = np.minimum(est[:, None], np.rint(dem))
    carry_unit = custo * params["taxa_carregamento"] * (1 + idade / params["fator_idade_dias"])
    carr = (est[:, None] - un) * carry_unit[:, None]
    ok = (mu / preco) >= config["margem_minima_percentual"]

    n_gene = np.maximum(ok.sum(1), 1)
    K = int(n_gene.max())
    shp = (n, K)
    fit = np.zeros(shp); UN = np.zeros(shp); CASH = np.zeros(shp); REC = np.zeros(shp)
    CARR = np.zeros(shp); PRE = np.zeros(shp); MU = np.zeros(shp); NIV = -np.ones(shp, int)
    for i in range(n):
        idx = np.flatnonzero(ok[i])
        if len(idx) == 0:     # SKU sem preço que respeite o piso de margem: não é vendido
            CARR[i, 0] = est[i] * carry_unit[i]
            fit[i, 0] = -CARR[i, 0]
            PRE[i, 0] = p0[i]
            continue
        m = len(idx)
        UN[i, :m] = un[i, idx]; CASH[i, :m] = un[i, idx] * mu[i, idx]
        REC[i, :m] = un[i, idx] * preco[i, idx]; CARR[i, :m] = carr[i, idx]
        fit[i, :m] = CASH[i, :m] - CARR[i, :m]
        PRE[i, :m] = preco[i, idx]; MU[i, :m] = mu[i, idx]; NIV[i, :m] = idx
    base_units = np.minimum(est, dem0)
    min_units = np.array([UN[i, :n_gene[i]].min() for i in range(n)])
    cap_pol, cap_tot = _capacidades(base_units, min_units, params)
    penal = max(1.0, params["penalidade_mult"] * float(MU.max()))
    return Tabelas(df["sku"].to_numpy(), n, K, n_gene, fit, UN, CASH, REC, CARR, PRE, MU, NIV, niv,
                   est, e, custo, carry_unit, base_units, min_units, cap_pol, cap_tot, penal, params)


def subconjunto(T, idx):
    idx = np.asarray(idx)
    campos = {c: getattr(T, c)[idx] for c in _POR_SKU}
    campos["n"] = len(idx)
    campos["cap_pol"], campos["cap_total"] = _capacidades(campos["base_units"], campos["min_units"], T.params)
    return replace(T, **campos)


def fitness_populacao(T, G):
    """Fitness vetorizado de uma população G (P x n)."""
    ar = np.arange(T.n)[None, :]
    f = T.fit[ar, G].sum(1)
    u = T.un[ar, G].sum(1)
    return f - T.penal * np.maximum(0.0, u - T.cap_pol)


def fitness_individuo_loop(T, g):
    """Versão ingênua (laço Python) usada na comparação antes/depois da otimização."""
    f = 0.0
    u = 0.0
    for i in range(T.n):
        f += T.fit[i, g[i]]
        u += T.un[i, g[i]]
    return f - T.penal * max(0.0, u - T.cap_pol)


def limite_superior_relaxado(T):
    """Ótimo ignorando a capacidade (relaxação): cota superior válida para o fitness."""
    return float(sum(T.fit[i, :T.n_gene[i]].max() for i in range(T.n)))


def avaliar(T, genes):
    ar = np.arange(T.n)
    g = np.asarray(genes)
    un = T.un[ar, g]
    caixa = float(T.cash[ar, g].sum())
    carr = float(T.carr[ar, g].sum())
    return {"caixa": caixa, "receita": float(T.rec[ar, g].sum()), "unidades": float(un.sum()),
            "carregamento": carr, "fitness": caixa - carr,
            "estoque_remanescente": float((T.est - un).sum()),
            "viavel": bool(un.sum() <= T.cap_pol + 1e-9)}


def politica_para_df(T, genes):
    ar = np.arange(T.n)
    g = np.asarray(genes)
    nivel = T.nivel[ar, g]
    return pd.DataFrame({
        "sku": T.skus,
        "nivel_relativo": np.where(nivel >= 0, T.niveis[np.maximum(nivel, 0)], np.nan),
        "preco": np.where(nivel >= 0, T.preco[ar, g], np.nan),
        "unidades": T.un[ar, g],
        "margem_unitaria": T.mu[ar, g],
        "caixa": T.cash[ar, g],
        "estoque_remanescente": T.est - T.un[ar, g],
        "vendavel": nivel >= 0,
    })
