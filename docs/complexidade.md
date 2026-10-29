# Complexidade (Big O) — tempo e memória

Variáveis: **n** = nº de SKUs; **v_i** = nº de preços permitidos do SKU i (≤ 7; `n_gene` em `questao_01/modelo.py`); **V** = max v_i;
**G** = gerações do GA; **P** = tamanho da população; **L** = lotes de incentivo comprados no Greedy da Q1;
**S** = nº de cenários Monte Carlo; **T** = semanas (12); **N** = SKUs da Q2; **K** = níveis de preço (6);
**J** = quantis por (semana, preço) usados pela DP (`J_quantis`=40); **E_max** = maior estoque inicial entre os SKUs.
Medições reais (pequena/média/grande) em `docs/desempenho.md`.

## Questão 1
| Algoritmo | Tempo | Memória | Trecho que domina | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| **Brute Force** (`brute_force.forca_bruta`) | O(n · Π v_i) ≈ **O(n · Vⁿ)** | O(B · n) (B = 400.000, tamanho do bloco) + O(n·V) tabelas | `fitness_populacao` aplicado a cada bloco de combinações (soma de n termos por combinação) | n ≳ 12 (7¹² ≈ 1,4·10¹⁰ combinações). Medido: n=4, 6, 8 → 324, 26.460, 846.720 combinações. **Alternativa: GA** (usado), ou DP sobre a capacidade, já que o único acoplamento é a restrição de unidades. |
| **Algoritmo Genético** (`genetic_algorithm.executar_ga`) | O(G · P · n) (+ O(G · P log P) da ordenação) | O(P · n + n · V) | avaliação do fitness da população inteira a cada geração (`fitness_populacao`) | Não dá garantia de ótimo (gap medido contra o Brute Force); com n muito grande o custo por geração cresce linear em n. **Alternativas:** avaliação incremental dos filhos, paralelismo, população menor com mais gerações. |
| **Greedy** (`greedy.alocar_incentivos`) | O(n + L · log n) | O(n) | operações de heap (`heappush/heappop`) a cada lote comprado | Deixa de ser ótimo se o benefício não for monotônico/decrescente (lotes interdependentes). **Alternativa:** mochila (DP) sobre o orçamento, O(n · orçamento/passo). |

## Questão 2
| Algoritmo | Tempo | Memória | Trecho que domina | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| **Monte Carlo — estimação** (`monte_carlo.estimar`) | O(S · T · N · K) para simular + O(T · N · K · S log S) para os quantis | O(S · T · N) (choques) + O(S · N · K) por semana + O(N · T · K · J) de saída | `rng.poisson(lam)` e `np.quantile` sobre o tensor (S, N, K) de cada semana | S·N·K grande demais para a RAM → processar por blocos de cenários (o código já faz uma semana por vez) ou reduzir K/J. |
| **Programação Dinâmica** (`dynamic_programming.resolver_sku`) | O(N · T · E_max · K · J) | O(N · T · E_max) (política) + O(T · E_max) (V por SKU) | laço sobre k com a matriz (E+1)×J: `V[t+1][s − vendido]` | Estoques de dezenas de milhares de unidades por SKU → **discretizar o estoque em faixas** (reduz E_max) e/ou reduzir J. Sem memoização o nº de caminhos seria (K+1)ᵀ ≈ 7¹² ≈ 1,4·10¹⁰; com ela são T·(E+1) estados por SKU. |
| **Greedy semanal** (`greedy.alocar_destaques`) | O(T · N · (J + log N)) | O(T · N) | cálculo do benefício esperado (média sobre J quantis) e ordenação por razão | N muito grande → pré-agregar por categoria. Deixa de ser ótimo se destaques tiverem custos fracionáveis ou interações entre SKUs. |
| **Monte Carlo — validação** (`monte_carlo.simular_plano`) | O(S · T · N) | O(S · T · N) (choques) + O(S · N) estado | `rng.poisson` + indexação da política `POL[ar, t, estoque]` por semana | S·T·N grande → gerar os choques semana a semana (menos memória, mesma complexidade de tempo). |

## Como a expressão aparece no código
- **O(G·P·n)** do GA: o `for ger in range(max_geracoes)` chama `_avaliar`, que faz `T.fit[ar, G].sum(1)` — uma matriz P×n por geração.
- **O(T·E·K·J)** da DP: os dois laços (`for t`, `for k`) e a matriz `vendido` de forma (E+1, J). O `V[t+1]` é reaproveitado por todos os (s, k, j): é onde está a reutilização de subproblemas.
- **O(S·T·N)** da validação: `for t in range(inst.T)` com operações vetorizadas sobre arrays (S, N).
