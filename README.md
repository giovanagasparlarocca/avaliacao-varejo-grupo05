# Checkpoint 6 — Recuperação financeira de varejistas por precificação online

**Grupo: G05**
- Giovana Gaspar Larocca — RM564965
- Giovanna Lins Sayama — RM565901
- Rayssa Luzia Portela Aquino — RM562024

**Seed base:** 10 · **Seed final:** 5010 · **ID da instância:** `0f71f458126e`

Python 3.12.3 · dependências em `requirements.txt`. Dados oficiais em `data/` (não editar: `manifesto_dados.json` guarda os SHA-256; cada `main.py` confere a integridade antes de rodar).

## Como reproduzir
```bash
pip install -r requirements.txt
python questao_01/main.py --modo completo        # ~10 s  (seed padrão = seed_final do manifesto = 5010)
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
**Nesta questão utilizamos 3 algoritmos.** O *Brute Force* resolve com exatidão um subconjunto reduzido de 8 SKUs e serve de **benchmark ótimo para validar o GA**; o *Algoritmo Genético* busca a política de preços do catálogo completo (30 SKUs); a política do GA é a **entrada** do *Greedy*, que reparte o orçamento de incentivos (cupom/cashback) pelos produtos de maior benefício marginal por real.
Fluxo: `brute_force.forca_bruta` ⇢ valida `genetic_algorithm.executar_ga` → `politica (genes)` → `greedy.alocar_incentivos` → `resultado_final.json`.

**Resultado final** (`questao_01/resultados/resultado_final.json`, produzido por `main.py`):
- Meta de caixa 90d: R$ 285,569.78 → caixa final **R$ 453,844.77** — meta **atingida**, folga R$ 168,274.99.
- Cobertura da dívida (R$ 304,077.37): **149.3%**. Margem de caixa sobre receita: 17.84%. Estoque remanescente: 385.6 un.
- Orçamento promocional: usado R$ 27,031.84 de R$ 93,085.96 (29.0%) — não há mais lotes com retorno ≥ 1 dentro do estoque remanescente e da capacidade logística.

| Evidência | Arquivo | Função |
|---|---|---|
| Brute Force × GA (subconjunto) | `tabela_bruteforce_vs_ga.csv`, `gap_ga_10_seeds.csv` | `brute_force.forca_bruta`, `genetic_algorithm.executar_ga` |
| Fitness por geração | `fitness_por_geracao.png/.csv` | `genetic_algorithm.executar_ga` (`hist_best`, `hist_mean`) |
| Comparação de 2 configs do GA | `tabela_ga_configuracoes.csv` | `main.py` (bloco Etapa B) |
| Antes/depois do Greedy | `tabela_antes_depois_greedy.csv`, `politica_final_por_sku.csv` | `greedy.alocar_incentivos`, `main.tabela_antes_depois` |
| Otimização antes/depois | `otimizacao_antes_depois.csv` | `modelo.fitness_individuo_loop` × `modelo.fitness_populacao` |
| Big O e desempenho | `docs/complexidade.md`, `desempenho_q1.csv` | `main.benchmark` |

**Brute Force × GA** (subconjunto de 8 SKUs, 846,720 combinações):
| metodo | config | fitness | gap_pct_vs_BF | avaliacoes | tempo_s | pico_KiB |
|---|---|---|---|---|---|---|
| Brute Force (ótimo) | - | 83,145.0982 | 0.0000 | 846720 | 0.3114 | 46,877.3672 |
| GA config A | A | 82,601.3294 | 0.6540 | 1840 | 0.0076 | 28.0146 |
| GA config B | B | 83,145.0982 | 0.0000 | 7950 | 0.0119 | 77.2988 |

Em 10 seeds diferentes: config A — gap médio 0.705% (máx. 5.09%, achou o ótimo em 6/10); config B — gap médio 0.000% (ótimo em 10/10).

**GA no catálogo completo** (gap medido contra o limite superior relaxado, que ignora a capacidade logística):
| config | populacao | p_mutacao | elitismo | fitness | gap_pct_vs_limite_relaxado | geracoes | avaliacoes | tempo_s | pico_KiB | viavel |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 40 | 0.0100 | 1 | 215,337.4921 | 1.8552 | 74 | 2960 | 0.0165 | 75.4014 | True |
| B | 150 | 0.0300 | 4 | 216,931.5511 | 1.1286 | 66 | 9900 | 0.0272 | 261.9395 | True |

Justificativas (população, torneio k=3, cruzamento uniforme, mutação por gene, elitismo, parada por paciência) em `docs/decisoes.md`.

**Antes/depois do Greedy:**
| metrica | antes_greedy (GA) | depois_greedy |
|---|---|---|
| Caixa operacional (R$) | 430,020.78 | 453,844.77 |
| Receita bruta (R$) | 2,347,020.62 | 2,543,387.52 |
| Margem de caixa sobre receita (%) | 18.32 | 17.84 |
| Unidades vendidas | 3,633.00 | 4,208.43 |
| Estoque remanescente (un.) | 961.00 | 385.57 |
| Gasto do orçamento promocional (R$) | 0.00 | 27,031.84 |
| Cobertura da dívida (%) | 141.42 | 149.25 |
| Folga (+) / insuficiência (-) vs meta (R$) | 144,451.00 | 168,274.99 |

**Otimização demonstrada:** avaliar o fitness com laço Python → vetorizado com NumPy: 0.251s → 0.032s (**7.8×**), mesmo fitness final.

---
## Questão 2 — Plano de Sobrevivência em 12 Semanas
**Nesta questão utilizamos 3 algoritmos.** *Monte Carlo* (A) estima por simulação a distribuição de demanda e caixa para cada (SKU, semana, nível de preço) com 2,000 cenários e, no final, valida o plano em 5,000 cenários novos; a *Programação Dinâmica* (B) usa essas estatísticas (quantis de demanda e margens) para escolher, por SKU, o preço de cada semana dado o estoque restante; o *Greedy* (C) usa a trajetória da DP (preço e estoque esperados) para repartir o orçamento semanal de destaques. O plano completo volta ao Monte Carlo para validação de risco.
Fluxo: `monte_carlo.estimar` → `dynamic_programming.resolver_dp` → `greedy.alocar_destaques` → `monte_carlo.simular_plano` (cenários independentes). O peso de urgência λ da DP é calibrado em um terceiro conjunto de cenários (λ = 4.0).

**DP:** estado (semana, estoque restante); decisão = nível de preço (ou pausar); transição `s' = s − min(s, D)` com D amostrada dos quantis do Monte Carlo; objetivo = soma da margem esperada ponderada por urgência. Reutilização de subproblemas: `V[t+1]` é calculada uma vez por estoque e consultada por todos os (preço, quantil) — ver docstring de `dynamic_programming.py`. Trajetória reconstruída em `trajetoria_dp.csv` e `trajetoria_representativos.png` (SKUs Q2-004, Q2-022, Q2-002).

**Validação** (5,000 cenários independentes; `questao_02/resultados/resultado_final.json`):
- Probabilidade de pagar no prazo (plano): semana 4: **100.0%** · semana 8: **100.0%** · semana 12: **100.0%**. Preço fixo de referência: semana 4: 100.0% · semana 8: 100.0% · semana 12: 100.0%.
- Saldo final esperado após as dívidas: **R$ 196,350.71** (preço fixo: R$ 67,410.58). Percentil de risco escolhido: **P10** = R$ 186,721.08.
- Margem 25.37% (preço fixo 16.76%). Estoque final médio 235 un. (≈ R$ 159,008 a custo) contra 140 un. no preço fixo. Cenários de falha: 0 de 5000.
- Gasto médio com destaques: R$ 4,042.49/semana de R$ 6,954.07 de orçamento.

**Atenção à leitura:** nesta instância as dívidas são *folgadas* (preço fixo também paga tudo), então P=100% não distingue as políticas. Por isso `--modo completo` roda um **estresse** multiplicando as dívidas (`estresse_divida.csv`), onde o plano passa a fazer diferença:
| fator_divida | lambda | P_todas_preco_fixo | P_todas_plano | saldo_medio_preco_fixo | saldo_medio_plano | P_sem4_plano | P_sem8_plano | P_sem12_plano |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 4.000 | 1.000 | 1.000 | 67,410.581 | 196,350.707 | 1.000 | 1.000 | 1.000 |
| 1.250 | 4.000 | 0.882 | 1.000 | 22,546.684 | 135,307.237 | 1.000 | 1.000 | 1.000 |
| 1.500 | 4.000 | 0.000 | 1.000 | 110,141.016 | 74,263.767 | 1.000 | 1.000 | 1.000 |
| 1.750 | 0.500 | 0.000 | 0.958 | 76,567.109 | 21,342.420 | 1.000 | 1.000 | 0.958 |

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
