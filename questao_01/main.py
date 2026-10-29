"""Questão 1 — Operação Caixa 90.   Fluxo:  Brute Force (valida) -> GA (política) -> Greedy (incentivos).

Uso:
    python questao_01/main.py --modo completo            # tudo, inclusive benchmark de desempenho
    python questao_01/main.py --modo rapido              # sem benchmark
    python questao_01/main.py --meta-fator 1.10 --orcamento-fator 0.75 --populacao 80 --mutacao 0.05
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum.io_dados import PASTA_DADOS, carregar_questao, verificar_integridade
from comum.medicao import medir
from questao_01 import modelo as M
from questao_01.brute_force import forca_bruta
from questao_01.genetic_algorithm import executar_ga
from questao_01.greedy import alocar_incentivos


def parse(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--seed", type=int, default=None, help="default = seed_final do manifesto")
    a.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    a.add_argument("--dados", default=str(PASTA_DADOS))
    a.add_argument("--saida", default=str(RAIZ / "questao_01" / "resultados"))
    a.add_argument("--meta-fator", type=float, default=1.0, help="ex.: 1.10 = meta de caixa +10%%")
    a.add_argument("--orcamento-fator", type=float, default=1.0, help="ex.: 0.75 = orçamento -25%%")
    a.add_argument("--fator-logistico", type=float, default=None, help="ex.: 1.15 = custo logístico +15%%")
    a.add_argument("--sku-indisponivel", default="", help="lista separada por vírgula, ex.: Q1-003,Q1-010")
    a.add_argument("--populacao", type=int, default=None)
    a.add_argument("--mutacao", type=float, default=None)
    a.add_argument("--n-sub", type=int, default=None, help="tamanho do subconjunto do Brute Force (>=6)")
    return a.parse_args(argv)


def gap_pct(otimo, valor):
    return 100.0 * (otimo - valor) / abs(otimo) if otimo else 0.0


def escolher_subconjunto(T, n_sub, seed):
    elegiveis = np.flatnonzero(T.n_gene >= 2)
    if len(elegiveis) < n_sub:
        raise ValueError("Poucos SKUs com >=2 preços permitidos para formar o subconjunto.")
    rng = np.random.default_rng(seed + 103)
    return np.sort(rng.choice(elegiveis, n_sub, replace=False))


def grafico_fitness(res, caminho, titulo):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nome, r in res.items():
        l, = ax.plot(r["hist_best"], label=f"Config {nome} — melhor")
        ax.plot(r["hist_mean"], "--", color=l.get_color(), alpha=0.6, label=f"Config {nome} — média")
    ax.set_xlabel("Geração"); ax.set_ylabel("Fitness (R$)"); ax.set_title(titulo)
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(caminho, dpi=140); plt.close(fig)


def tabela_antes_depois(T, genes, aloc, cfg):
    av = M.avaliar(T, genes)
    caixa_a = av["caixa"]
    caixa_d = caixa_a + aloc["ganho_bruto"] - aloc["gasto"]
    rec_d = av["receita"] + aloc["receita_extra"]
    un_d = av["unidades"] + aloc["unidades_extra"].sum()
    rem_d = av["estoque_remanescente"] - aloc["unidades_extra"].sum()
    div, meta = cfg["divida_total_90d"], cfg["meta_caixa_90d"]
    linhas = [
        ("Caixa operacional (R$)", caixa_a, caixa_d),
        ("Receita bruta (R$)", av["receita"], rec_d),
        ("Margem de caixa sobre receita (%)", 100 * caixa_a / av["receita"], 100 * caixa_d / rec_d),
        ("Unidades vendidas", av["unidades"], un_d),
        ("Estoque remanescente (un.)", av["estoque_remanescente"], rem_d),
        ("Gasto do orçamento promocional (R$)", 0.0, aloc["gasto"]),
        ("Cobertura da dívida (%)", 100 * caixa_a / div, 100 * caixa_d / div),
        ("Folga (+) / insuficiência (-) vs meta (R$)", caixa_a - meta, caixa_d - meta),
    ]
    return pd.DataFrame(linhas, columns=["metrica", "antes_greedy (GA)", "depois_greedy"])


def benchmark(df, cfg, params, seed, saida):
    """Medição real em entradas pequena/média/grande para os três algoritmos."""
    linhas = []
    T30 = M.construir_tabelas(df, cfg, params)
    for n in (4, 6, 8):
        S = M.subconjunto(T30, escolher_subconjunto(T30, n, seed))
        r, t, mem = medir(forca_bruta, S)
        linhas.append(("Brute Force", f"n={n}", int(r["avaliadas"]), t, mem))
    cfg_fixo = dict(params["ga"]["A"], paciencia=10**9, max_geracoes=100)
    for fator in (1, 4):
        d = M.replicar_catalogo(df, fator)
        T = M.construir_tabelas(d, cfg, params)
        _, t, mem = medir(executar_ga, T, cfg_fixo, seed)
        linhas.append(("Algoritmo Genético", f"n={T.n}", cfg_fixo["populacao"] * 100, t, mem))
        g = executar_ga(T, cfg_fixo, seed)["genes"]
        _, tg, memg = medir(alocar_incentivos, T, g, cfg["orcamento_promocional"] * fator, repeticoes=5)
        linhas.append(("Greedy", f"n={T.n}", T.n, tg, memg))
    S8 = M.subconjunto(T30, escolher_subconjunto(T30, 8, seed))
    _, t, mem = medir(executar_ga, S8, cfg_fixo, seed)
    linhas.append(("Algoritmo Genético", "n=8", cfg_fixo["populacao"] * 100, t, mem))
    _, tg, memg = medir(alocar_incentivos, S8, executar_ga(S8, cfg_fixo, seed)["genes"],
                        cfg["orcamento_promocional"], repeticoes=5)
    linhas.append(("Greedy", "n=8", S8.n, tg, memg))
    out = pd.DataFrame(linhas, columns=["algoritmo", "entrada", "unidade_de_trabalho", "tempo_s", "pico_memoria_KiB"])
    out["_n"] = out["entrada"].str.extract(r"n=(\d+)").astype(int)
    out = out.sort_values(["algoritmo", "_n"]).drop(columns="_n").reset_index(drop=True)
    out.to_csv(Path(saida) / "desempenho_q1.csv", index=False, encoding="utf-8")
    return out


def main(argv=None):
    args = parse(argv)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    verificar_integridade(args.dados)
    df, cfg, man = carregar_questao(1, args.dados)
    seed = args.seed if args.seed is not None else man["seed_final"]
    params = M.carregar_parametros(sobrescritas={"fator_logistico": args.fator_logistico,
                                                 "n_subconjunto": args.n_sub})
    for k in params["ga"]:
        if args.populacao is not None:
            params["ga"][k]["populacao"] = args.populacao
            params["ga"][k]["elitismo"] = min(params["ga"][k]["elitismo"], args.populacao - 1)
        if args.mutacao is not None:
            params["ga"][k]["p_mutacao_gene"] = args.mutacao
    if params["n_subconjunto"] < 6:
        raise ValueError("O enunciado exige subconjunto com pelo menos 6 produtos.")
    indisp = [s.strip() for s in args.sku_indisponivel.split(",") if s.strip()]
    df, cfg = M.aplicar_cenario(df, cfg, args.meta_fator, args.orcamento_fator, indisp)
    T = M.construir_tabelas(df, cfg, params)
    print(f"[Q1] instância {man['id_instancia']} | seed={seed} | SKUs={T.n} | "
          f"sem preço viável pelo piso de margem: {int((T.nivel[:, 0] < 0).sum())}")

    # ---- Etapa A: Brute Force no subconjunto + validação do GA ----
    idx = escolher_subconjunto(T, params["n_subconjunto"], seed)
    S = M.subconjunto(T, idx)
    bf, t_bf, mem_bf = medir(forca_bruta, S)
    linhas = [("Brute Force (ótimo)", "-", bf["fitness"], 0.0, int(bf["avaliadas"]), t_bf, mem_bf)]
    for nome, c in params["ga"].items():
        r, t, mem = medir(executar_ga, S, c, seed)
        linhas.append((f"GA config {nome}", nome, r["fitness"], gap_pct(bf["fitness"], r["fitness"]),
                       r["avaliacoes"], t, mem))
    tab_bf = pd.DataFrame(linhas, columns=["metodo", "config", "fitness", "gap_pct_vs_BF", "avaliacoes", "tempo_s", "pico_KiB"])
    tab_bf.to_csv(saida / "tabela_bruteforce_vs_ga.csv", index=False)
    print("\n[Q1] Brute Force vs GA no subconjunto:", list(S.skus))
    print(tab_bf.to_string(index=False))
    multi = []
    for nome, c in params["ga"].items():
        gaps = [gap_pct(bf["fitness"], executar_ga(S, c, seed + k)["fitness"]) for k in range(10)]
        multi.append((nome, np.mean(gaps), np.max(gaps), sum(g < 1e-9 for g in gaps)))
    pd.DataFrame(multi, columns=["config", "gap_medio_pct", "gap_max_pct", "ots_de_10"]).to_csv(
        saida / "gap_ga_10_seeds.csv", index=False)

    # ---- Etapa B: GA no catálogo completo (2 configurações) ----
    res, linhas = {}, []
    ub = M.limite_superior_relaxado(T)
    for nome, c in params["ga"].items():
        r, t, mem = medir(executar_ga, T, c, seed)
        res[nome] = r
        av = M.avaliar(T, r["genes"])
        linhas.append((nome, c["populacao"], c["p_mutacao_gene"], c["elitismo"], r["fitness"],
                       gap_pct(ub, r["fitness"]), r["geracoes"], r["avaliacoes"], t, mem, av["viavel"]))
    tab_ga = pd.DataFrame(linhas, columns=["config", "populacao", "p_mutacao", "elitismo", "fitness",
                                           "gap_pct_vs_limite_relaxado", "geracoes", "avaliacoes",
                                           "tempo_s", "pico_KiB", "viavel"])
    tab_ga.to_csv(saida / "tabela_ga_configuracoes.csv", index=False)
    print("\n[Q1] GA no catálogo completo (limite superior relaxado = %.2f):" % ub)
    print(tab_ga.to_string(index=False))
    grafico_fitness(res, saida / "fitness_por_geracao.png", "GA — fitness por geração (catálogo completo)")
    pd.DataFrame({f"{n}_{k}": pd.Series(r[f'hist_{k}']) for n, r in res.items() for k in ("best", "mean")}
                 ).to_csv(saida / "fitness_por_geracao.csv", index_label="geracao")

    # ---- Etapa C: Greedy alimentado pela política do GA ----
    nome_p = params["ga_principal"]
    genes = res[nome_p]["genes"]
    aloc = alocar_incentivos(T, genes, cfg["orcamento_promocional"])
    antes_depois = tabela_antes_depois(T, genes, aloc, cfg)
    antes_depois.to_csv(saida / "tabela_antes_depois_greedy.csv", index=False)
    pol = M.politica_para_df(T, genes)
    pol["lotes_incentivo"] = aloc["lotes"]
    pol["unidades_extra"] = aloc["unidades_extra"].round(2)
    pol.to_csv(saida / "politica_final_por_sku.csv", index=False)
    print("\n[Q1] Antes/depois do Greedy (política GA config %s):" % nome_p)
    print(antes_depois.to_string(index=False))

    # ---- Otimização de código: fitness em laço Python (antes) vs vetorizado (depois) ----
    c_loop = dict(params["ga"][nome_p], vetorizado=False)
    c_vet = dict(params["ga"][nome_p], vetorizado=True)
    r0, t0, m0 = medir(executar_ga, T, c_loop, seed, repeticoes=3)
    r1, t1, m1 = medir(executar_ga, T, c_vet, seed, repeticoes=3)
    otim = pd.DataFrame([("antes: fitness em laço Python", t0, m0, r0["fitness"]),
                         ("depois: fitness vetorizado (NumPy)", t1, m1, r1["fitness"])],
                        columns=["versao", "tempo_s", "pico_KiB", "fitness_final"])
    otim["speedup"] = t0 / otim["tempo_s"]
    otim.to_csv(saida / "otimizacao_antes_depois.csv", index=False)
    print("\n[Q1] Otimização:\n", otim.to_string(index=False))

    # ---- Resultado final ----
    caixa_f = float(antes_depois.iloc[0, 2])
    final = {
        "grupo": man["grupo"], "id_instancia": man["id_instancia"], "seed_usada": seed,
        "config_ga_principal": nome_p,
        "caixa_apos_ga": float(antes_depois.iloc[0, 1]), "caixa_final": caixa_f,
        "meta_caixa_90d": cfg["meta_caixa_90d"], "divida_total_90d": cfg["divida_total_90d"],
        "meta_atingida": bool(caixa_f >= cfg["meta_caixa_90d"]),
        "folga_vs_meta": caixa_f - cfg["meta_caixa_90d"],
        "cobertura_divida_pct": 100 * caixa_f / cfg["divida_total_90d"],
        "margem_caixa_sobre_receita_pct": float(antes_depois.iloc[2, 2]),
        "estoque_remanescente": float(antes_depois.iloc[4, 2]),
        "orcamento_promocional": cfg["orcamento_promocional"], "gasto_promocional": aloc["gasto"],
        "uso_orcamento_pct": 100 * aloc["gasto"] / cfg["orcamento_promocional"] if cfg["orcamento_promocional"] else 0.0,
        "capacidade_politica_unid": T.cap_pol, "politica_viavel": bool(M.avaliar(T, genes)["viavel"]),
    }
    with open(saida / "resultado_final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print("\n[Q1] Resultado final:", json.dumps(final, ensure_ascii=False, indent=2))

    if args.modo == "completo":
        b = benchmark(df, cfg, params, seed, saida)
        print("\n[Q1] Benchmark (pequena/média/grande):\n", b.to_string(index=False))
    return final


if __name__ == "__main__":
    main()
