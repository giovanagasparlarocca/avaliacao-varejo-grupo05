"""Modelo da Questão 2 (Plano de Sobrevivência em 12 semanas).

Demanda semanal do SKU i, preço p = m_k * preco_atual, semana t, cenário s:
    lambda = demanda_base * m_k^(-elasticidade) * comp[s,t,i]^gamma * fd[s,t,i] * (1 + uplift_i * destaque)
    demanda ~ Poisson(lambda)
    comp[s,t,i] = preço do concorrente relativo ao preço-base (passeio aleatório log-normal, média 1)
    fd[s,t,i]   = choque de demanda log-normal (média 1)
    fl[s,t]     = choque de custo logístico log-normal (média 1), comum a todos os SKUs na semana
Caixa semanal = sum_i vendas * (p*(1-taxa) - custo_unitario - custo_logistico_base*fl) - custo dos destaques.
As dívidas das semanas 4, 8 e 12 são pagas se o saldo acumulado >= valor da dívida.
"""
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

COLUNAS = ["sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial", "demanda_base_semanal",
           "elasticidade", "taxa_marketplace", "custo_logistico_base", "preco_concorrente_base"]
PARAMS_ARQ = Path(__file__).resolve().parent / "parametros_q2.json"


def carregar_parametros(caminho=None, sobrescritas=None):
    with open(caminho or PARAMS_ARQ, "r", encoding="utf-8") as f:
        p = json.load(f)
    for k, v in (sobrescritas or {}).items():
        if v is not None:
            p[k] = v
    return p


def validar_entrada(df, cfg):
    faltam = [c for c in COLUNAS if c not in df.columns]
    if faltam:
        raise ValueError(f"Colunas ausentes no catálogo: {faltam}")
    if len(df) == 0:
        raise ValueError("Catálogo vazio.")
    for c in ["custo_unitario", "preco_atual", "estoque_inicial", "demanda_base_semanal", "elasticidade"]:
        if (df[c] <= 0).any():
            raise ValueError(f"Valores não positivos em {c}.")
    if cfg["numero_semanas"] < 1 or any(v < 0 for v in cfg["dividas"].values()):
        raise ValueError("Configuração inválida (semanas/dívidas).")
    if cfg["orcamento_promocional_semanal"] < 0:
        raise ValueError("orcamento_promocional_semanal negativo.")
    if any(v < 0 for v in cfg["variabilidade"].values()):
        raise ValueError("variabilidade negativa.")
    return True


def aplicar_cenario(df, cfg, orcamento_fator=1.0, divida_fator=1.0, sku_indisponivel=()):
    cfg = copy.deepcopy(cfg)
    cfg["orcamento_promocional_semanal"] *= orcamento_fator
    cfg["dividas"] = {k: v * divida_fator for k, v in cfg["dividas"].items()}
    df = df[~df["sku"].isin(list(sku_indisponivel))].reset_index(drop=True)
    return df, cfg


def replicar_catalogo(df, fator):
    partes = []
    for r in range(fator):
        d = df.copy()
        d["sku"] = d["sku"] + f"#{r + 1}"
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


@dataclass
class Instancia:
    skus: np.ndarray
    cat: np.ndarray
    custo: np.ndarray
    p0: np.ndarray
    est0: np.ndarray
    base: np.ndarray
    e: np.ndarray
    tax: np.ndarray
    lg: np.ndarray
    pc0: np.ndarray
    niv: np.ndarray
    T: int
    N: int
    K: int
    sig_d: float
    sig_c: float
    sig_l: float
    caixa_ini: float
    prazos: list
    dividas: np.ndarray
    orc_sem: float
    params: dict


_POR_SKU = ["skus", "cat", "custo", "p0", "est0", "base", "e", "tax", "lg", "pc0"]


def construir_instancia(df, cfg, params):
    validar_entrada(df, cfg)
    pz = sorted(int(k.split("_")[1]) for k in cfg["dividas"])
    div = np.array([cfg["dividas"][f"semana_{w}"] for w in pz], float)
    v = cfg["variabilidade"]
    niv = np.asarray(cfg["niveis_preco_relativos"], float)
    return Instancia(
        df["sku"].to_numpy(), df["categoria"].to_numpy(), df["custo_unitario"].to_numpy(float),
        df["preco_atual"].to_numpy(float), df["estoque_inicial"].to_numpy(int),
        df["demanda_base_semanal"].to_numpy(float), df["elasticidade"].to_numpy(float),
        df["taxa_marketplace"].to_numpy(float),
        df["custo_logistico_base"].to_numpy(float) * params["fator_logistico"],
        df["preco_concorrente_base"].to_numpy(float), niv, int(cfg["numero_semanas"]), len(df), len(niv),
        v["sigma_demanda"], v["sigma_preco_concorrente"], v["sigma_custo_logistico"],
        float(cfg["caixa_inicial"]), pz, div, float(cfg["orcamento_promocional_semanal"]), params)


def subconjunto(inst, idx):
    idx = np.asarray(idx)
    campos = {c: getattr(inst, c)[idx] for c in _POR_SKU}
    campos["N"] = len(idx)
    return replace(inst, **campos)


def indice_preco_atual(inst):
    return int(np.argmin(np.abs(inst.niv - 1.0)))
