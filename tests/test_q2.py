import itertools

import numpy as np
import pytest

from comum.io_dados import carregar_questao
from questao_02 import modelo as M
from questao_02 import monte_carlo as MC
from questao_02.dynamic_programming import resolver_sku, trajetoria_nominal, resolver_dp
from questao_02.greedy import alocar_destaques
from questao_02.main import baseline_fixo, construir_plano


@pytest.fixture(scope="module")
def inst4():
    df, cfg, _ = carregar_questao(2)
    p = M.carregar_parametros(); p["limite_risco_falha"] = 0.1
    return M.construir_instancia(df.iloc[:4].reset_index(drop=True), cfg, p)


def test_dp_igual_enumeracao_exaustiva():
    """Instância determinística (J=1): o V_0 da DP deve ser o máximo sobre todas as sequências de decisões."""
    T, K, S0 = 3, 3, 6
    q = np.array([[[1], [2], [4]], [[2], [3], [5]], [[1], [3], [4]]])       # (T,K,J) demandas
    mm = np.array([10.0, 6.0, 3.0])
    w = np.array([1.2, 1.1, 1.0])
    V, POL = resolver_sku(q, mm, S0, w, 0.0)
    melhor = 0.0
    for seq in itertools.product(range(K + 1), repeat=T):                    # K = pausar
        s, val = S0, 0.0
        for t, k in enumerate(seq):
            vend = 0 if k == K else min(s, q[t, k, 0])
            val += w[t] * (0 if k == K else mm[k]) * vend
            s -= vend
        melhor = max(melhor, val)
    assert V[0, S0] == pytest.approx(melhor)
    # a trajetória reconstruída pela política atinge o mesmo valor
    s, val = S0, 0.0
    for t in range(T):
        k = int(POL[t, s]); vend = 0 if k == K else min(s, q[t, k, 0])
        val += w[t] * (0 if k == K else mm[k]) * vend; s -= vend
    assert val == pytest.approx(melhor)


def test_dp_estoque_zero_vale_zero():
    q = np.ones((2, 2, 1), int) * 3
    V, _ = resolver_sku(q, np.array([5.0, 4.0]), 0, np.ones(2), 0.0)
    assert V[0, 0] == 0.0


def test_choques_reprodutiveis_e_cenarios_independentes(inst4):
    a, b = MC.gerar_choques(inst4, 100, 11), MC.gerar_choques(inst4, 100, 11)
    c = MC.gerar_choques(inst4, 100, 12)
    assert np.array_equal(a.fd, b.fd) and not np.array_equal(a.fd, c.fd)
    p = inst4.params
    assert p["offset_estimacao"] != p["offset_validacao"] != p["offset_calibracao"]


def test_choques_tem_media_um(inst4):
    ch = MC.gerar_choques(inst4, 20000, 5)
    assert abs(ch.fd.mean() - 1) < 0.01 and abs(ch.fl.mean() - 1) < 0.01


def test_choques_tamanho_invalido(inst4):
    with pytest.raises(ValueError):
        MC.gerar_choques(inst4, 0, 1)


def test_probabilidades_no_intervalo_e_extremos(inst4):
    ch = MC.gerar_choques(inst4, 300, 1)
    pol, promo, up, cp = baseline_fixo(inst4)
    r = MC.simular_plano(inst4, pol, promo, up, cp, ch, 2)
    assert all(0 <= v <= 1 for v in r["prob_pagamento"].values())
    import dataclasses
    sem_div = dataclasses.replace(inst4, dividas=inst4.dividas * 0)
    assert MC.simular_plano(sem_div, pol, promo, up, cp, ch, 2)["prob_todas"] == 1.0
    enorme = dataclasses.replace(inst4, dividas=inst4.dividas * 1e9)
    assert MC.simular_plano(enorme, pol, promo, up, cp, ch, 2)["prob_todas"] == 0.0


def test_simulacao_reprodutivel(inst4):
    ch = MC.gerar_choques(inst4, 200, 1)
    a = MC.simular_plano(inst4, *baseline_fixo(inst4), ch, 3)
    b = MC.simular_plano(inst4, *baseline_fixo(inst4), ch, 3)
    assert a["saldo_final_medio"] == b["saldo_final_medio"]


def test_estoque_final_nunca_negativo(inst4):
    ch = MC.gerar_choques(inst4, 200, 1)
    r = MC.simular_plano(inst4, *baseline_fixo(inst4), ch, 3)
    assert (r["_estoque_final"] >= 0).all()


def test_greedy_respeita_orcamento_semanal(inst4):
    est = MC.estimar(inst4, MC.gerar_choques(inst4, 300, 1), 2, J=10)
    d = resolver_dp(inst4, est, 1.0)
    tr = trajetoria_nominal(inst4, est, d["POL"])
    g = alocar_destaques(inst4, est, d["POL"], tr, orcamento_semanal=500.0)
    assert (g["gasto_semanal"] <= 500.0 + 1e-9).all()
    g0 = alocar_destaques(inst4, est, d["POL"], tr, orcamento_semanal=0.0)
    assert g0["promo"].sum() == 0
    with pytest.raises(ValueError):
        alocar_destaques(inst4, est, d["POL"], tr, orcamento_semanal=-1)


def test_pipeline_pequeno_integrado(inst4):
    p = dict(inst4.params, n_cenarios_estimacao=200, n_cenarios_calibracao=200, J_quantis=10,
             urgencia_grid=[0.0, 1.0])
    import dataclasses
    inst = dataclasses.replace(inst4, params=p)
    plano = construir_plano(inst, 5010)
    assert plano["lambda"] in (0.0, 1.0)
    assert plano["dp"]["POL"].shape[:2] == (inst.N, inst.T)
    assert len(plano["calibracao"]) == 2


def test_entrada_invalida_q2():
    df, cfg, _ = carregar_questao(2)
    with pytest.raises(ValueError):
        M.validar_entrada(df.drop(columns=["elasticidade"]), cfg)
    with pytest.raises(ValueError):
        M.validar_entrada(df, dict(cfg, orcamento_promocional_semanal=-5))
    d = df.copy(); d.loc[0, "estoque_inicial"] = 0
    with pytest.raises(ValueError):
        M.validar_entrada(d, cfg)
