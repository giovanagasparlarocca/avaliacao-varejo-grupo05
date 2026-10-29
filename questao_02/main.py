"""Questão 2 — Plano de Sobrevivência em 12 Semanas.
Fluxo:  Monte Carlo (estima) -> Programação Dinâmica (preços) -> Greedy (destaques) -> Monte Carlo (valida)

Uso:
    python questao_02/main.py --cenarios 5000                 # execução principal
    python questao_02/main.py --modo completo                 # + escalabilidade e testes de estresse de dívida
    python questao_02/main.py --concorrente-fator 0.92 --fator-logistico 1.15 --orcamento-fator 0.75
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
from questao_02 import modelo as M
from questao_02 import monte_carlo as MC
from questao_02.dynamic_programming import resolver_dp, trajetoria_nominal
from questao_02.greedy import alocar_destaques


def parse(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--seed", type=int, default=None, help="default = seed_final do manifesto")
    a.add_argument("--modo", choices=["rapido", "completo"], default="rapido")
    a.add_argument("--cenarios", type=int, default=None, help="cenários de validação final (default 5000)")
    a.add_argument("--cenarios-estimacao", type=int, default=None, help="default 2000")
    a.add_argument("--dados", default=str(PASTA_DADOS))
    a.add_argument("--saida", default=str(RAIZ / "questao_02" / "resultados"))
    a.add_argument("--orcamento-fator", type=float, default=1.0, help="0.75 = orçamento promocional -25%%")
    a.add_argument("--divida-fator", type=float, default=1.0, help="1.10 = dívidas +10%%")
    a.add_argument("--fator-logistico", type=float, default=None, help="1.15 = custo logístico +15%%")
    a.add_argument("--concorrente-fator", type=float, default=None, help="0.92 = concorrente reduz preço em 8%%")
    a.add_argument("--sku-indisponivel", default="")
    a.add_argument("--lambda-urgencia", type=float, default=None, help="força λ (pula a calibração)")
    return a.parse_args(argv)


def baseline_fixo(inst):
    """Política ingênua: preço atual o tempo todo, sem destaques (referência para medir o valor do plano)."""
    k1 = M.indice_preco_atual(inst)
    pol = np.full((inst.N, inst.T, int(inst.est0.max()) + 1), k1, dtype=np.int8)
    z = np.zeros((inst.T, inst.N))
    return pol, z.astype(bool), np.zeros(inst.N), z


def construir_plano(inst, seed, lam_forcado=None):
    """Etapas A (estimar) + B (DP) + C (Greedy). Calibra o peso de urgência λ em cenários próprios."""
    p = inst.params
    S_est = p["n_cenarios_estimacao"]
    est = MC.estimar(inst, MC.gerar_choques(inst, S_est, seed + p["offset_estimacao"]), seed + 1)
    cal = MC.gerar_choques(inst, p["n_cenarios_calibracao"], seed + p["offset_calibracao"])
    grade = [lam_forcado] if lam_forcado is not None else p["urgencia_grid"]
    linhas, melhor = [], None
    for lam in grade:
        d = resolver_dp(inst, est, lam)
        tr = trajetoria_nominal(inst, est, d["POL"])
        g = alocar_destaques(inst, est, d["POL"], tr)
        r = MC.simular_plano(inst, d["POL"], g["promo"], g["uplift"], g["custo_promo"], cal, seed + 7)
        linhas.append({"lambda": lam, "prob_todas": r["prob_todas"], "saldo_medio": r["saldo_final_medio"],
                       "saldo_p_risco": r["saldo_final_percentil_risco"],
                       **{f"prob_sem{w}": v for w, v in r["prob_pagamento"].items()}})
        meta_ok = r["prob_todas"] >= 1 - p.get("limite_risco_falha", 0.10)
        chave = (meta_ok, r["saldo_final_medio"] if meta_ok else r["prob_todas"])
        if melhor is None or chave > melhor[0]:
            melhor = (chave, lam, d, tr, g)
    _, lam, d, tr, g = melhor
    return {"est": est, "lambda": lam, "dp": d, "traj": tr, "greedy": g, "calibracao": pd.DataFrame(linhas)}


def validar(inst, plano, S, seed):
    val = MC.gerar_choques(inst, S, seed + inst.params["offset_validacao"])
    g = plano["greedy"]
    r_plano = MC.simular_plano(inst, plano["dp"]["POL"], g["promo"], g["uplift"], g["custo_promo"], val, seed + 9)
    r_base = MC.simular_plano(inst, *baseline_fixo(inst), val, seed + 9)
    return r_plano, r_base


def graf_distribuicao(r_plano, r_base, inst, caminho):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r_base["_saldo"], bins=60, alpha=0.5, label="Preço fixo (referência)")
    ax.hist(r_plano["_saldo"], bins=60, alpha=0.6, label="Plano MC→DP→Greedy")
    ax.axvline(r_plano["saldo_final_percentil_risco"], color="red", ls="--",
               label=f"P{r_plano['percentil_risco']} do plano")
    ax.set_xlabel("Saldo final de caixa após dívidas pagas (R$)"); ax.set_ylabel("Cenários")
    ax.set_title("Distribuição do caixa final — validação Monte Carlo"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(caminho, dpi=140); plt.close(fig)


def graf_trajetorias(inst, plano, caminho):
    V0 = plano["dp"]["V0"]
    ordem = np.argsort(V0)
    escolhidos = [int(ordem[-1]), int(ordem[len(ordem) // 2]), int(ordem[0])]
    tr = plano["traj"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    sem = np.arange(1, inst.T + 1)
    for ax, i in zip(axs, escolhidos):
        k = tr["nivel"][i]
        preco = np.where(k < inst.K, inst.p0[i] * inst.niv[np.minimum(k, inst.K - 1)], np.nan)
        ax.step(sem, preco, where="mid", color="tab:blue", label="Preço (R$)")
        ax.set_xlabel("Semana"); ax.set_ylabel("Preço (R$)", color="tab:blue")
        ax2 = ax.twinx(); ax2.bar(sem, tr["estoque_inicio"][i], alpha=0.25, color="tab:orange")
        ax2.set_ylabel("Estoque no início da semana", color="tab:orange")
        ax.set_title(f"{inst.skus[i]} ({inst.cat[i]})")
    fig.suptitle("Trajetória nominal da DP: preço e estoque"); fig.tight_layout()
    fig.savefig(caminho, dpi=140); plt.close(fig)
    return [inst.skus[i] for i in escolhidos]


def graf_prob(r_plano, r_base, caminho):
    ws = sorted(r_plano["prob_pagamento"])
    x = np.arange(len(ws))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.2, [r_base["prob_pagamento"][w] for w in ws], 0.4, label="Preço fixo")
    ax.bar(x + 0.2, [r_plano["prob_pagamento"][w] for w in ws], 0.4, label="Plano")
    ax.set_xticks(x); ax.set_xticklabels([f"Semana {w}" for w in ws]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilidade de pagar no prazo"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(caminho, dpi=140); plt.close(fig)


def etapa_dp(inst, est):
    d = resolver_dp(inst, est, 1.0)
    trajetoria_nominal(inst, est, d["POL"])
    return d


def escalabilidade(df, cfg, params, seed, saida):
    """Tamanhos de entrada: N ∈ {6,12,24,48} SKUs (S=2000) e S ∈ {500,2000,5000} cenários (N=24)."""
    linhas = []

    def rodar(df_n, S, rotulo):
        inst = M.construir_instancia(df_n, cfg, params)
        ch, t, m = medir(MC.gerar_choques, inst, S, seed + 1)
        linhas.append((rotulo, "Monte Carlo — gerar choques", inst.N, S, t, m))
        est, t, m = medir(MC.estimar, inst, ch, seed + 2)
        linhas.append((rotulo, "Monte Carlo — estimar", inst.N, S, t, m))
        d, t, m = medir(etapa_dp, inst, est)
        linhas.append((rotulo, "Programação Dinâmica", inst.N, S, t, m))
        tr = trajetoria_nominal(inst, est, d["POL"])
        g, t, m = medir(alocar_destaques, inst, est, d["POL"], tr, repeticoes=3)
        linhas.append((rotulo, "Greedy", inst.N, S, t, m))
        _, t, m = medir(MC.simular_plano, inst, d["POL"], g["promo"], g["uplift"], g["custo_promo"], ch, seed + 3)
        linhas.append((rotulo, "Monte Carlo — validar", inst.N, S, t, m))

    for n in (6, 12, 24):
        rodar(df.iloc[:n].reset_index(drop=True), 2000, f"N={n}")
    rodar(M.replicar_catalogo(df, 2).reset_index(drop=True), 2000, "N=48")
    for S in (500, 5000):
        rodar(df, S, f"S={S}")
    out = pd.DataFrame(linhas, columns=["tamanho", "etapa", "N_skus", "S_cenarios", "tempo_s", "pico_memoria_KiB"])
    out.to_csv(Path(saida) / "escalabilidade_q2.csv", index=False)
    return out


def main(argv=None):
    args = parse(argv)
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    verificar_integridade(args.dados)
    df, cfg, man = carregar_questao(2, args.dados)
    seed = args.seed if args.seed is not None else man["seed_final"]
    params = M.carregar_parametros(sobrescritas={
        "fator_logistico": args.fator_logistico, "fator_concorrente_inicial": args.concorrente_fator,
        "n_cenarios_estimacao": args.cenarios_estimacao, "n_cenarios_validacao": args.cenarios})
    params["limite_risco_falha"] = cfg["limite_risco_falha"]
    indisp = [s.strip() for s in args.sku_indisponivel.split(",") if s.strip()]
    df0, cfg0 = df, cfg
    df, cfg = M.aplicar_cenario(df, cfg, args.orcamento_fator, args.divida_fator, indisp)
    inst = M.construir_instancia(df, cfg, params)
    S_val = params["n_cenarios_validacao"]
    if S_val < 5000 or params["n_cenarios_estimacao"] < 2000:
        print("[aviso] o enunciado exige >=2000 cenários (estimação) e >=5000 (validação).")
    print(f"[Q2] instância {man['id_instancia']} | seed={seed} | SKUs={inst.N} | semanas={inst.T} | "
          f"dívidas={ {int(w): round(float(v), 2) for w, v in zip(inst.prazos, inst.dividas)} }")

    plano = construir_plano(inst, seed, args.lambda_urgencia)
    plano["est"].por_nivel.to_csv(saida / "mc_estimativa_por_preco.csv", index=False)
    plano["calibracao"].to_csv(saida / "calibracao_lambda.csv", index=False)
    print("\n[Q2] Calibração do peso de urgência λ (cenários de calibração):")
    print(plano["calibracao"].round(3).to_string(index=False))
    print("λ escolhido:", plano["lambda"])

    r_plano, r_base = validar(inst, plano, S_val, seed)
    tr, g = plano["traj"], plano["greedy"]
    linhas = []
    for i in range(inst.N):
        for t in range(inst.T):
            k = int(tr["nivel"][i, t])
            linhas.append((inst.skus[i], inst.cat[i], t + 1, "sem_venda" if k >= inst.K else inst.niv[k],
                           np.nan if k >= inst.K else round(inst.p0[i] * inst.niv[k], 2),
                           int(tr["estoque_inicio"][i, t]), round(float(tr["vendas_esperadas"][i, t]), 2),
                           bool(g["promo"][t, i])))
    pd.DataFrame(linhas, columns=["sku", "categoria", "semana", "nivel_preco", "preco", "estoque_inicio",
                                  "vendas_esperadas", "destaque"]).to_csv(saida / "trajetoria_dp.csv", index=False)
    repres = graf_trajetorias(inst, plano, saida / "trajetoria_representativos.png")
    graf_distribuicao(r_plano, r_base, inst, saida / "distribuicao_caixa_final.png")
    graf_prob(r_plano, r_base, saida / "probabilidade_pagamento.png")
    pd.DataFrame({"percentil": [1, 5, 10, 25, 50, 75, 90, 95, 99],
                  "saldo_final_plano": np.percentile(r_plano["_saldo"], [1, 5, 10, 25, 50, 75, 90, 95, 99]),
                  "saldo_final_preco_fixo": np.percentile(r_base["_saldo"], [1, 5, 10, 25, 50, 75, 90, 95, 99])}
                 ).to_csv(saida / "distribuicao_caixa_final.csv", index=False)

    def resumo(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    final = {"grupo": man["grupo"], "id_instancia": man["id_instancia"], "seed_usada": seed,
             "lambda_urgencia": plano["lambda"], "cenarios_estimacao": params["n_cenarios_estimacao"],
             "cenarios_validacao": S_val, "skus_representativos": repres,
             "plano": resumo(r_plano), "preco_fixo_referencia": resumo(r_base),
             "gasto_promocional_semanal_medio": float(g["gasto_semanal"].mean()),
             "orcamento_promocional_semanal": inst.orc_sem,
             "limite_risco_falha": cfg["limite_risco_falha"],
             "risco_aceito": bool(r_plano["prob_todas"] >= 1 - cfg["limite_risco_falha"])}
    with open(saida / "resultado_final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print("\n[Q2] Validação (cenários independentes) — plano vs preço fixo:")
    print(json.dumps({"plano": resumo(r_plano), "preco_fixo": resumo(r_base)}, ensure_ascii=False, indent=2))

    if args.modo == "completo":
        # estresse: dívidas maiores mostram onde o plano passa a valer a pena
        linhas = []
        for f_div in (1.0, 1.25, 1.5, 1.75):
            d2, c2 = M.aplicar_cenario(df0, cfg0, args.orcamento_fator, f_div, indisp)
            i2 = M.construir_instancia(d2, c2, params)
            pl = construir_plano(i2, seed, args.lambda_urgencia)
            a, b = validar(i2, pl, S_val, seed)
            linhas.append((f_div, pl["lambda"], b["prob_todas"], a["prob_todas"],
                           b["saldo_final_medio"], a["saldo_final_medio"],
                           *[a["prob_pagamento"][w] for w in sorted(a["prob_pagamento"])]))
        pd.DataFrame(linhas, columns=["fator_divida", "lambda", "P_todas_preco_fixo", "P_todas_plano",
                                      "saldo_medio_preco_fixo", "saldo_medio_plano",
                                      "P_sem4_plano", "P_sem8_plano", "P_sem12_plano"]
                     ).to_csv(saida / "estresse_divida.csv", index=False)
        print("\n[Q2] Estresse de dívida:\n", pd.read_csv(saida / "estresse_divida.csv").round(3).to_string(index=False))
        esc = escalabilidade(df0, cfg0, params, seed, saida)
        print("\n[Q2] Escalabilidade:\n", esc.round(4).to_string(index=False))
    return final


if __name__ == "__main__":
    main()
