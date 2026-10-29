"""Gera docs/arquitetura.png, docs/desempenho.md e README.md a partir dos resultados em */resultados/.
Uso: python tools/gerar_docs.py   (rode depois de questao_01/main.py e questao_02/main.py --modo completo)
"""
import json
import platform
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R1, R2 = RAIZ / "questao_01" / "resultados", RAIZ / "questao_02" / "resultados"


def md(df, casas=4):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: f"{x:,.{casas}f}")
    cab = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return cab + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values) + "\n"


def caixa(ax, x, y, w, h, texto, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=cor, ec="black"))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=9)


def seta(ax, x0, y0, x1, y1, rot="", estilo="-|>", **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo, mutation_scale=14, **kw))
    if rot:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.05, rot, ha="center", fontsize=8)


def diagrama():
    fig, ax = plt.subplots(figsize=(13, 6.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(0.1, 5.9, "Questão 1 — Operação Caixa 90", fontsize=11, weight="bold")
    caixa(ax, 0.2, 4.3, 2.6, 1.0, "Brute Force\n(ótimo no subconjunto)", "#f7d9a8")
    caixa(ax, 3.6, 4.3, 2.6, 1.0, "Algoritmo Genético\n(catálogo completo)", "#bfe0f5")
    caixa(ax, 7.0, 4.3, 2.6, 1.0, "Greedy\n(orçamento de incentivos)", "#cfeacb")
    caixa(ax, 10.4, 4.3, 2.4, 1.0, "Resultado final\ncaixa, margem, dívida", "#eeeeee")
    seta(ax, 2.8, 4.8, 3.6, 4.8, "valida (gap %)")
    seta(ax, 6.2, 4.8, 7.0, 4.8, "política de preços")
    seta(ax, 9.6, 4.8, 10.4, 4.8)
    ax.text(0.1, 3.4, "Questão 2 — Plano de Sobrevivência em 12 Semanas", fontsize=11, weight="bold")
    caixa(ax, 0.2, 1.6, 2.6, 1.2, "Monte Carlo (A)\nestima demanda/caixa\n≥2.000 cenários", "#f7d9a8")
    caixa(ax, 3.6, 1.6, 2.6, 1.2, "Prog. Dinâmica (B)\nestado (semana, estoque)\ntrajetória de preços", "#bfe0f5")
    caixa(ax, 7.0, 1.6, 2.6, 1.2, "Greedy (C)\ndestaques semanais\npor benefício/custo", "#cfeacb")
    caixa(ax, 10.4, 1.6, 2.4, 1.2, "Monte Carlo (D)\nvalidação ≥5.000\ncenários novos", "#f7d9a8")
    seta(ax, 2.8, 2.2, 3.6, 2.2, "quantis, margens")
    seta(ax, 6.2, 2.2, 7.0, 2.2, "plano de preços")
    seta(ax, 9.6, 2.2, 10.4, 2.2, "plano completo")
    ax.add_patch(FancyArrowPatch((11.6, 1.6), (1.5, 1.6), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", mutation_scale=14, ls="--", color="gray"))
    ax.text(6.5, 0.45, "P(pagar semanas 4, 8, 12) e risco  →  recalibra o peso de urgência λ da DP", ha="center", fontsize=8)
    fig.savefig(RAIZ / "docs" / "arquitetura.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def desempenho_md():
    q1 = pd.read_csv(R1 / "desempenho_q1.csv")
    q2 = pd.read_csv(R2 / "escalabilidade_q2.csv")
    ot = pd.read_csv(R1 / "otimizacao_antes_depois.csv")
    txt = ["# Desempenho medido\n",
           f"Python {platform.python_version()} · NumPy {np.__version__} · pandas {pd.__version__}. "
           "Tempo com `time.perf_counter()` e memória de pico com `tracemalloc` (execuções separadas). "
           "Os números mudam um pouco entre máquinas; a **tendência** é o que se compara com o Big O.\n",
           "## Questão 1 — pequena / média / grande\n", md(q1, 5),
           "Leitura: Brute Force cresce ~V^n (324 → 26.460 → 846.720 combinações; tempo ×~40 a cada +2 SKUs). "
           "GA e Greedy crescem aproximadamente linear em n.\n",
           "## Questão 1 — otimização antes/depois (fitness do GA)\n", md(ot, 5),
           "Mesmo `fitness_final` nas duas versões (mesma seed): a otimização não altera o resultado, só o tempo.\n",
           "## Questão 2 — escalabilidade (N SKUs e S cenários)\n", md(q2, 5),
           "Leitura: choques/estimação/validação crescem ~linear em N·S; a DP cresce ~linear em N (e depende do estoque máximo)."
           " O Greedy é desprezível frente às demais etapas.\n"]
    (RAIZ / "docs" / "desempenho.md").write_text("\n".join(txt), encoding="utf-8")
    return ot


def readme(ot):
    man = json.load(open(RAIZ / "data" / "manifesto_dados.json", encoding="utf-8"))
    f1 = json.load(open(R1 / "resultado_final.json", encoding="utf-8"))
    f2 = json.load(open(R2 / "resultado_final.json", encoding="utf-8"))
    bf = pd.read_csv(R1 / "tabela_bruteforce_vs_ga.csv")
    gaps = pd.read_csv(R1 / "gap_ga_10_seeds.csv")
    cfgs = pd.read_csv(R1 / "tabela_ga_configuracoes.csv")
    ad = pd.read_csv(R1 / "tabela_antes_depois_greedy.csv")
    est = pd.read_csv(R2 / "estresse_divida.csv")
    cal = pd.read_csv(R2 / "calibracao_lambda.csv")
    p, b = f2["plano"], f2["preco_fixo_referencia"]
    prob = " · ".join(f"semana {w}: **{100 * v:.1f}%**" for w, v in p["prob_pagamento"].items())
    probb = " · ".join(f"semana {w}: {100 * v:.1f}%" for w, v in b["prob_pagamento"].items())
    txt = f"""# Checkpoint 6 — Recuperação financeira de varejistas por precificação online

**Grupo: G05**
- Giovana Gaspar Larocca — RM564965
- Giovanna Lins Sayama — RM565901
- Rayssa Luzia Portela Aquino — RM562024

**Seed base:** {man['seed_base']} · **Seed final:** {man['seed_final']} · **ID da instância:** `{man['id_instancia']}`

Python {platform.python_version()} · dependências em `requirements.txt`. Dados oficiais em `data/` (não editar: `manifesto_dados.json` guarda os SHA-256; cada `main.py` confere a integridade antes de rodar).

## Como reproduzir
```bash
pip install -r requirements.txt
python questao_01/main.py --modo completo        # ~10 s  (seed padrão = seed_final do manifesto = {man['seed_final']})
python questao_02/main.py --modo completo --cenarios 5000   # ~1 min com estresse e escalabilidade
python -m pytest -q tests                        # testes automatizados
python tools/gerar_docs.py                       # regenera docs/desempenho.md, arquitetura.png e este README
```
`--seed N` troca a seed dos componentes aleatórios (os arquivos de dados continuam os oficiais).
Todo o resultado numérico abaixo vem de `questao_0X/resultados/`.

## Cenário-surpresa: tudo por parâmetro (sem mexer no código)
| Alteração pedida | Questão 1 | Questão 2 |
|---|---|---|
| meta de caixa +10% | `--meta-fator 1.10` | — (dívidas: `--divida-fator 1.10`) |
| concorrente reduz preço em 8% | — | `--concorrente-fator 0.92` |
| custo logístico +15% | `--fator-logistico 1.15` | `--fator-logistico 1.15` |
| orçamento promocional −25% | `--orcamento-fator 0.75` | `--orcamento-fator 0.75` |
| um SKU indisponível | `--sku-indisponivel Q1-003` | `--sku-indisponivel Q2-003` |
| mutação / população do GA | `--mutacao 0.05` · `--populacao 80` | — |
Demais hiperparâmetros (taxa de carregamento, capacidade, cupom, uplift, λ, nº de cenários, quantis…) ficam em `questao_01/parametros_q1.json` e `questao_02/parametros_q2.json`.

## Estrutura
```
data/            instância oficial do G05 + manifesto + notebook gerador (só Bloco 02 alterado)
comum/           leitura/integridade dos dados (io_dados.py) e medição tempo/memória (medicao.py)
questao_01/      modelo.py · brute_force.py · genetic_algorithm.py · greedy.py · main.py · resultados/
questao_02/      modelo.py · monte_carlo.py · dynamic_programming.py · greedy.py · main.py · resultados/
tests/           testes automatizados (pytest)
docs/            arquitetura.png · complexidade.md · desempenho.md · decisoes.md
tools/           gerar_docs.py
```
Diagrama do fluxo: `docs/arquitetura.png`. Big O: `docs/complexidade.md`. Desempenho medido: `docs/desempenho.md`. Hipóteses e decisões: `docs/decisoes.md`. Uso de IA: `IA_USO.md`.

---
## Questão 1 — Operação Caixa 90
**Nesta questão utilizamos 3 algoritmos.** O *Brute Force* resolve com exatidão um subconjunto de {len(pd.read_csv(R1 / 'politica_final_por_sku.csv'))} SKUs... o subconjunto reduzido tem 8 SKUs e serve de **benchmark ótimo para validar o GA**; o *Algoritmo Genético* busca a política de preços do catálogo completo (30 SKUs); a política do GA é a **entrada** do *Greedy*, que reparte o orçamento de incentivos (cupom/cashback) pelos produtos de maior benefício marginal por real.
Fluxo: `brute_force.forca_bruta` ⇢ valida `genetic_algorithm.executar_ga` → `politica (genes)` → `greedy.alocar_incentivos` → `resultado_final.json`.

**Resultado final** (`questao_01/resultados/resultado_final.json`, produzido por `main.py`):
- Meta de caixa 90d: R$ {f1['meta_caixa_90d']:,.2f} → caixa final **R$ {f1['caixa_final']:,.2f}** — meta **{'atingida' if f1['meta_atingida'] else 'NÃO atingida'}**, folga R$ {f1['folga_vs_meta']:,.2f}.
- Cobertura da dívida (R$ {f1['divida_total_90d']:,.2f}): **{f1['cobertura_divida_pct']:.1f}%**. Margem de caixa sobre receita: {f1['margem_caixa_sobre_receita_pct']:.2f}%. Estoque remanescente: {f1['estoque_remanescente']:.1f} un.
- Orçamento promocional: usado R$ {f1['gasto_promocional']:,.2f} de R$ {f1['orcamento_promocional']:,.2f} ({f1['uso_orcamento_pct']:.1f}%) — não há mais lotes com retorno ≥ 1 dentro do estoque remanescente e da capacidade logística.

| Evidência | Arquivo | Função |
|---|---|---|
| Brute Force × GA (subconjunto) | `tabela_bruteforce_vs_ga.csv`, `gap_ga_10_seeds.csv` | `brute_force.forca_bruta`, `genetic_algorithm.executar_ga` |
| Fitness por geração | `fitness_por_geracao.png/.csv` | `genetic_algorithm.executar_ga` (`hist_best`, `hist_mean`) |
| Comparação de 2 configs do GA | `tabela_ga_configuracoes.csv` | `main.py` (bloco Etapa B) |
| Antes/depois do Greedy | `tabela_antes_depois_greedy.csv`, `politica_final_por_sku.csv` | `greedy.alocar_incentivos`, `main.tabela_antes_depois` |
| Otimização antes/depois | `otimizacao_antes_depois.csv` | `modelo.fitness_individuo_loop` × `modelo.fitness_populacao` |
| Big O e desempenho | `docs/complexidade.md`, `desempenho_q1.csv` | `main.benchmark` |

**Brute Force × GA** (subconjunto de 8 SKUs, {int(bf.iloc[0]['avaliacoes']):,} combinações):
{md(bf, 4)}
Em 10 seeds diferentes: config A — gap médio {gaps.iloc[0]['gap_medio_pct']:.3f}% (máx. {gaps.iloc[0]['gap_max_pct']:.2f}%, achou o ótimo em {int(gaps.iloc[0]['ots_de_10'])}/10); config B — gap médio {gaps.iloc[1]['gap_medio_pct']:.3f}% (ótimo em {int(gaps.iloc[1]['ots_de_10'])}/10).

**GA no catálogo completo** (gap medido contra o limite superior relaxado, que ignora a capacidade logística):
{md(cfgs, 4)}
Justificativas (população, torneio k=3, cruzamento uniforme, mutação por gene, elitismo, parada por paciência) em `docs/decisoes.md`.

**Antes/depois do Greedy:**
{md(ad, 2)}
**Otimização demonstrada:** avaliar o fitness com laço Python → vetorizado com NumPy: {ot.iloc[0]['tempo_s']:.3f}s → {ot.iloc[1]['tempo_s']:.3f}s (**{ot.iloc[1]['speedup']:.1f}×**), mesmo fitness final.

---
## Questão 2 — Plano de Sobrevivência em 12 Semanas
**Nesta questão utilizamos 3 algoritmos.** *Monte Carlo* (A) estima por simulação a distribuição de demanda e caixa para cada (SKU, semana, nível de preço) com {f2['cenarios_estimacao']:,} cenários e, no final, valida o plano em {f2['cenarios_validacao']:,} cenários novos; a *Programação Dinâmica* (B) usa essas estatísticas (quantis de demanda e margens) para escolher, por SKU, o preço de cada semana dado o estoque restante; o *Greedy* (C) usa a trajetória da DP (preço e estoque esperados) para repartir o orçamento semanal de destaques. O plano completo volta ao Monte Carlo para validação de risco.
Fluxo: `monte_carlo.estimar` → `dynamic_programming.resolver_dp` → `greedy.alocar_destaques` → `monte_carlo.simular_plano` (cenários independentes). O peso de urgência λ da DP é calibrado em um terceiro conjunto de cenários (λ = {f2['lambda_urgencia']}).

**DP:** estado (semana, estoque restante); decisão = nível de preço (ou pausar); transição `s' = s − min(s, D)` com D amostrada dos quantis do Monte Carlo; objetivo = soma da margem esperada ponderada por urgência. Reutilização de subproblemas: `V[t+1]` é calculada uma vez por estoque e consultada por todos os (preço, quantil) — ver docstring de `dynamic_programming.py`. Trajetória reconstruída em `trajetoria_dp.csv` e `trajetoria_representativos.png` (SKUs {', '.join(f2['skus_representativos'])}).

**Validação** ({f2['cenarios_validacao']:,} cenários independentes; `questao_02/resultados/resultado_final.json`):
- Probabilidade de pagar no prazo (plano): {prob}. Preço fixo de referência: {probb}.
- Saldo final esperado após as dívidas: **R$ {p['saldo_final_medio']:,.2f}** (preço fixo: R$ {b['saldo_final_medio']:,.2f}). Percentil de risco escolhido: **P{p['percentil_risco']}** = R$ {p['saldo_final_percentil_risco']:,.2f}.
- Margem {p['margem_pct']:.2f}% (preço fixo {b['margem_pct']:.2f}%). Estoque final médio {p['estoque_final_medio_un']:.0f} un. (≈ R$ {p['estoque_final_medio_custo']:,.0f} a custo) contra {b['estoque_final_medio_un']:.0f} un. no preço fixo. Cenários de falha: {p['cenarios_falha']} de {p['cenarios_total']}.
- Gasto médio com destaques: R$ {f2['gasto_promocional_semanal_medio']:,.2f}/semana de R$ {f2['orcamento_promocional_semanal']:,.2f} de orçamento.

**Atenção à leitura:** nesta instância as dívidas são *folgadas* (preço fixo também paga tudo), então P=100% não distingue as políticas. Por isso `--modo completo` roda um **estresse** multiplicando as dívidas (`estresse_divida.csv`), onde o plano passa a fazer diferença:
{md(est, 3)}
Observação: no preço fixo, dívidas não pagas **não** saem do saldo, por isso o saldo médio dele parece alto quando a probabilidade de pagar é 0 — compare as probabilidades, não o saldo. Como na instância original todos os λ dão P=100% (saldos quase iguais, diferença < 0,1%), a escolha do λ ali é praticamente um empate; ele só pesa nos cenários de estresse.

Calibração de λ (cenários de calibração): `calibracao_lambda.csv`. Estimativas por nível de preço: `mc_estimativa_por_preco.csv`. Distribuição do caixa final: `distribuicao_caixa_final.png/.csv`. Escalabilidade (N = 6, 12, 24, 48 SKUs; S = 500, 2.000, 5.000): `escalabilidade_q2.csv` e `docs/desempenho.md`.

| Evidência | Arquivo | Função |
|---|---|---|
| Diagrama MC→DP→Greedy→MC | `docs/arquitetura.png` | — |
| Distribuição do caixa final | `distribuicao_caixa_final.png` | `main.graf_distribuicao` |
| Trajetória semanal de preço/estoque | `trajetoria_representativos.png`, `trajetoria_dp.csv` | `dynamic_programming.trajetoria_nominal` |
| Prob. de pagamento por prazo | `resultado_final.json`, `probabilidade_pagamento.png` | `monte_carlo.simular_plano` |
| Big O (MC, DP, Greedy) | `docs/complexidade.md` | — |

## Limitações conhecidas
Veja `docs/decisoes.md` (modelo de demanda, caixa = margem de contribuição, DP por SKU com peso de urgência em vez de otimização conjunta do caixa agregado, semântica de dívida não paga). Pontos a discutir antes da defesa estão listados lá.
"""
    txt = txt.replace(f"um subconjunto de {len(pd.read_csv(R1 / 'politica_final_por_sku.csv'))} SKUs... o subconjunto reduzido tem 8 SKUs e serve",
                      "um subconjunto reduzido de 8 SKUs e serve")
    (RAIZ / "README.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    diagrama()
    ot = desempenho_md()
    readme(ot)
    print("docs gerados.")
