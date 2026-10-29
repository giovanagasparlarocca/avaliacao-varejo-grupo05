import numpy as np
import pandas as pd
import pytest

from comum.io_dados import carregar_questao
from questao_01 import brute_force as BF
from questao_01 import modelo as M
from questao_01.genetic_algorithm import executar_ga
from questao_01.greedy import alocar_incentivos

NIVEIS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]


def df_pequeno(n=4):
    return pd.DataFrame({
        "sku": [f"T-{i}" for i in range(n)], "categoria": "X",
        "custo_unitario": [50.0, 80.0, 30.0, 120.0][:n], "preco_atual": [100.0, 140.0, 70.0, 190.0][:n],
        "estoque": [60, 40, 120, 30][:n], "demanda_base_90d": [50, 45, 100, 28][:n],
        "elasticidade": [1.5, 1.2, 2.0, 0.9][:n], "taxa_marketplace": 0.12, "custo_logistico": 6.0,
        "idade_estoque_dias": [100, 300, 20, 200][:n]})


def cfg_pequeno():
    return {"niveis_preco_relativos": NIVEIS, "margem_minima_percentual": 0.05,
            "orcamento_promocional": 800.0, "meta_caixa_90d": 1000.0, "divida_total_90d": 1500.0}


@pytest.fixture
def T4():
    return M.construir_tabelas(df_pequeno(), cfg_pequeno(), M.carregar_parametros())


def test_bruteforce_vetorizado_igual_itertools(T4):
    a = BF.forca_bruta(T4, tamanho_bloco=50)
    b = BF.forca_bruta_itertools(T4)
    assert a["fitness"] == pytest.approx(b["fitness"])
    assert a["avaliadas"] == int(np.prod(T4.n_gene))


def test_ga_encontra_otimo_em_instancia_pequena(T4):
    otimo = BF.forca_bruta(T4)["fitness"]
    cfg = M.carregar_parametros()["ga"]["B"]
    r = executar_ga(T4, cfg, seed=1)
    assert (otimo - r["fitness"]) / abs(otimo) < 0.01


def test_ga_reprodutivel_por_seed(T4):
    cfg = M.carregar_parametros()["ga"]["A"]
    a, b = executar_ga(T4, cfg, 7), executar_ga(T4, cfg, 7)
    assert np.array_equal(a["genes"], b["genes"]) and a["fitness"] == b["fitness"]
    assert np.array_equal(a["hist_best"], b["hist_best"])


def test_ga_historico_monotonico_com_elitismo(T4):
    r = executar_ga(T4, M.carregar_parametros()["ga"]["A"], 3)
    assert np.all(np.diff(r["hist_best"]) >= -1e-9)


def test_genes_dentro_dos_limites():
    df, cfg, _ = carregar_questao(1)
    T = M.construir_tabelas(df, cfg, M.carregar_parametros())
    r = executar_ga(T, M.carregar_parametros()["ga"]["A"], 5010)
    assert np.all(r["genes"] >= 0) and np.all(r["genes"] < T.n_gene)
    assert M.avaliar(T, r["genes"])["viavel"]


def test_fitness_vetorizado_igual_ao_laco(T4):
    G = np.random.default_rng(0).integers(0, T4.n_gene, size=(20, T4.n))
    v = M.fitness_populacao(T4, G)
    l = [M.fitness_individuo_loop(T4, g) for g in G]
    assert np.allclose(v, l)


def test_greedy_respeita_orcamento_e_estoque(T4):
    genes = np.zeros(T4.n, int)
    r = alocar_incentivos(T4, genes, 800.0)
    assert r["gasto"] <= 800.0 + 1e-9
    resto = T4.est - T4.un[np.arange(T4.n), genes]
    assert np.all(r["unidades_extra"] <= resto + 1e-9)


def test_greedy_orcamento_zero_nao_compra(T4):
    r = alocar_incentivos(T4, np.zeros(T4.n, int), 0.0)
    assert r["gasto"] == 0 and r["lotes"].sum() == 0


def test_greedy_orcamento_negativo_levanta(T4):
    with pytest.raises(ValueError):
        alocar_incentivos(T4, np.zeros(T4.n, int), -1.0)


def test_entrada_invalida():
    df, cfg = df_pequeno(), cfg_pequeno()
    with pytest.raises(ValueError):
        M.validar_entrada(df.drop(columns=["estoque"]), cfg)
    d2 = df.copy(); d2.loc[0, "estoque"] = 0
    with pytest.raises(ValueError):
        M.validar_entrada(d2, cfg)
    with pytest.raises(ValueError):
        M.validar_entrada(df, dict(cfg, margem_minima_percentual=1.5))
    with pytest.raises(ValueError):
        M.validar_entrada(df.iloc[0:0], cfg)


def test_sku_sem_preco_viavel_nao_vende():
    df = df_pequeno(); df.loc[0, "custo_unitario"] = 500.0   # custo > preço
    T = M.construir_tabelas(df, cfg_pequeno(), M.carregar_parametros())
    assert T.n_gene[0] == 1 and T.nivel[0, 0] == -1 and T.un[0, 0] == 0


def test_bruteforce_recusa_espaco_enorme(T4, monkeypatch):
    monkeypatch.setattr(BF, "LIMITE_COMBINACOES", 10)
    with pytest.raises(ValueError):
        BF.forca_bruta(T4)


def test_cenario_remove_sku_e_escala_parametros():
    df, cfg = M.aplicar_cenario(df_pequeno(), cfg_pequeno(), meta_fator=1.1, orcamento_fator=0.75,
                                sku_indisponivel=["T-1"])
    assert "T-1" not in set(df["sku"])
    assert cfg["meta_caixa_90d"] == pytest.approx(1100.0) and cfg["orcamento_promocional"] == pytest.approx(600.0)


def test_ga_parametros_invalidos(T4):
    cfg = dict(M.carregar_parametros()["ga"]["A"], elitismo=100)
    with pytest.raises(ValueError):
        executar_ga(T4, cfg, 1)
