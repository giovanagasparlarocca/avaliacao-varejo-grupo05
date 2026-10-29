# Decisões de modelagem, hipóteses e pontos para a defesa

> O enunciado deixa várias escolhas em aberto (“a modelagem deve ser explicada pelos alunos”). Abaixo está o que foi **assumido** neste rascunho. Cada item é uma decisão do modelo, não um fato dos dados — o grupo deve conferir se concorda e ser capaz de alterá-la.

## Questão 1
1. **Demanda:** elasticidade constante, `d = demanda_base · m^(−elasticidade)`; vendas `u = min(estoque, round(d))`. `m` é o nível relativo de preço de `niveis_preco_relativos`.
2. **Caixa operacional = margem de contribuição:** `u · (p·(1−taxa_marketplace) − custo_unitario − custo_logistico)`. É a mesma base com a qual o gerador calibrou meta e dívida (`caixa_base_estimado`). O custo do estoque é tratado como já pago.
3. **Preços permitidos:** níveis cujo `margem unitária / preço ≥ margem_minima_percentual`. 3 dos 30 SKUs não têm nenhum nível viável nesta instância e ficam **sem venda** (gene único, 0 unidades).
4. **Custo de carregamento (hipótese nossa):** sem algo que penalize estoque parado, o problema recomenda quase sempre o maior preço. Foi adicionado `custo_carregamento = estoque_remanescente · custo · 0,30 · (1 + idade/180)`, usando o campo `idade_estoque_dias` do dataset. O fitness é `caixa − carregamento`; a **meta de caixa é medida só com caixa** (sem o carregamento). Taxa e fator estão em `parametros_q1.json`.
5. **Capacidade logística (hipótese nossa):** com o fitness separável por SKU, o Brute Force/GA não teriam o que “resolver”. Foi adicionada uma capacidade de unidades da política (95% das unidades-base) que acopla os SKUs, tratada por penalidade no GA/BF. O Greedy pode usar até 110% (capacidade total).
6. **GA:** cromossomo = índice do preço permitido por SKU. Torneio k=3 (pressão seletiva moderada), cruzamento uniforme (genes independentes entre SKUs), mutação por gene (reinicia o preço), elitismo (não perde o melhor), parada por paciência (para quando estagna). Config A (40 indivíduos, mutação 1%) × Config B (150, 3%) — B explora mais e achou o ótimo do subconjunto em 10/10 seeds.
7. **Greedy (incentivos):** lote = 10 cupons de 5% do preço; conversão por cupom `min(0,6; 0,25·elasticidade)`; razão = (unidades extra · margem) / custo do lote; compra só se razão ≥ 1. O gasto do orçamento reduz o caixa final.
8. **Gap do GA no catálogo completo** é medido contra o *limite superior relaxado* (ótimo ignorando a capacidade), pois o Brute Force não escala para 30 SKUs.

## Questão 2
1. **Demanda:** `Poisson(λ)`, `λ = base · m^(−e) · comp^γ · choque_demanda · (1+uplift·destaque)` com γ = 0,5 (sensibilidade ao preço relativo do concorrente).
2. **Três fontes de incerteza:** demanda (log-normal, σ_demanda + Poisson), preço do concorrente (passeio aleatório log-normal, σ_preco_concorrente) e custo logístico (log-normal comum à semana, σ_custo_logistico). Todos com média 1.
3. **Cenários:** 2.000 para estimar, **outros** 2.000 para calibrar λ e 5.000 **independentes** para validar (seeds distintas: `seed+401/402/403`).
4. **DP por SKU, não conjunta:** o estado é (semana, estoque do SKU). A restrição de caixa/dívidas é agregada e seria um estado multidimensional; em vez disso a DP maximiza margem esperada com **peso de urgência λ** (mais peso às semanas antes das dívidas) e um desconto de 1%/semana (caixa cedo vale mais). O λ é **escolhido por Monte Carlo** (maior saldo esperado entre os que mantêm P(pagar tudo) ≥ 90%). É uma aproximação: não garante o ótimo conjunto.
5. **“Pausar vendas” (decisão K):** SKUs com margem unitária negativa em todos os preços (ex.: Q2-002) seriam vendidos com prejuízo; a DP pode pausar.
6. **Valor residual do estoque = 0** (`valor_residual_pct_custo`): o plano ignora o valor do estoque que sobra, o que o leva a preços altos e deixa estoque final maior que o preço fixo (ver README). **Ponto para discutir:** subir esse parâmetro muda a política.
7. **Greedy:** destaque custa 5% da receita esperada do SKU na semana e aumenta a demanda em `min(0,35; 0,20·e)`; usa o preço e o estoque **da trajetória da DP**; compra por benefício/custo até o orçamento semanal.
8. **Dívidas:** cada prazo é checado com o saldo acumulado; paga se saldo ≥ dívida (e então o saldo cai). Dívida não paga **não** é carregada para o prazo seguinte (cada prazo conta como falha/sucesso próprio).
9. **Instância folgada:** com os parâmetros do gerador do G05, até o preço fixo paga as 3 dívidas em 100% dos cenários. Para mostrar o valor do plano há o teste de estresse (`--modo completo`, `--divida-fator`).

## Para o grupo revisar/alterar antes da defesa
- Concordam com as hipóteses 4 e 5 (Q1) e 4 e 6 (Q2)? Elas moldam os números.
- Consigam explicar por que o problema da Q1 sem capacidade/carregamento seria trivial (recomenda sempre o preço máximo).
- Treinem os cenários-surpresa da tabela do README e discutam o efeito esperado antes de rodar.
