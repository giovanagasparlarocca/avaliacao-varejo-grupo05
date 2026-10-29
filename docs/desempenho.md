# Desempenho medido

Python 3.12.3 · NumPy 2.4.4 · pandas 3.0.2. Tempo com `time.perf_counter()` e memória de pico com `tracemalloc` (execuções separadas). Os números mudam um pouco entre máquinas; a **tendência** é o que se compara com o Big O.

## Questão 1 — pequena / média / grande

| algoritmo | entrada | unidade_de_trabalho | tempo_s | pico_memoria_KiB |
|---|---|---|---|---|
| Algoritmo Genético | n=8 | 4000 | 0.01054 | 29.17090 |
| Algoritmo Genético | n=30 | 4000 | 0.02586 | 76.37012 |
| Algoritmo Genético | n=120 | 4000 | 0.04924 | 279.81934 |
| Brute Force | n=4 | 324 | 0.00019 | 41.82812 |
| Brute Force | n=6 | 26460 | 0.00723 | 2,172.35156 |
| Brute Force | n=8 | 846720 | 0.28350 | 46,877.32031 |
| Greedy | n=8 | 8 | 0.00023 | 4.13281 |
| Greedy | n=30 | 30 | 0.00170 | 5.06055 |
| Greedy | n=120 | 120 | 0.00290 | 12.87500 |

Leitura: Brute Force cresce ~V^n (324 → 26.460 → 846.720 combinações; tempo ×~40 a cada +2 SKUs). GA e Greedy crescem aproximadamente linear em n.

## Questão 1 — otimização antes/depois (fitness do GA)

| versao | tempo_s | pico_KiB | fitness_final | speedup |
|---|---|---|---|---|
| antes: fitness em laço Python | 0.25109 | 229.83789 | 216,931.55114 | 1.00000 |
| depois: fitness vetorizado (NumPy) | 0.03205 | 261.98633 | 216,931.55114 | 7.83422 |

Mesmo `fitness_final` nas duas versões (mesma seed): a otimização não altera o resultado, só o tempo.

## Questão 2 — escalabilidade (N SKUs e S cenários)

| tamanho | etapa | N_skus | S_cenarios | tempo_s | pico_memoria_KiB |
|---|---|---|---|---|---|
| N=6 | Monte Carlo — gerar choques | 6 | 2000 | 0.00987 | 5,814.63574 |
| N=6 | Monte Carlo — estimar | 6 | 2000 | 0.12456 | 2,642.81250 |
| N=6 | Programação Dinâmica | 6 | 2000 | 0.02204 | 598.44727 |
| N=6 | Greedy | 6 | 2000 | 0.00145 | 3.25781 |
| N=6 | Monte Carlo — validar | 6 | 2000 | 0.02362 | 988.82812 |
| N=12 | Monte Carlo — gerar choques | 12 | 2000 | 0.01866 | 11,439.63574 |
| N=12 | Monte Carlo — estimar | 12 | 2000 | 0.24802 | 5,057.93750 |
| N=12 | Programação Dinâmica | 12 | 2000 | 0.04239 | 625.96289 |
| N=12 | Greedy | 12 | 2000 | 0.00285 | 4.07812 |
| N=12 | Monte Carlo — validar | 12 | 2000 | 0.03924 | 1,820.67188 |
| N=24 | Monte Carlo — gerar choques | 24 | 2000 | 0.03074 | 22,689.63574 |
| N=24 | Monte Carlo — estimar | 24 | 2000 | 0.45805 | 9,888.31250 |
| N=24 | Programação Dinâmica | 24 | 2000 | 0.08506 | 981.13574 |
| N=24 | Greedy | 24 | 2000 | 0.00365 | 6.04688 |
| N=24 | Monte Carlo — validar | 24 | 2000 | 0.06012 | 3,309.18750 |
| N=48 | Monte Carlo — gerar choques | 48 | 2000 | 0.06039 | 45,189.63574 |
| N=48 | Monte Carlo — estimar | 48 | 2000 | 0.90982 | 19,551.31250 |
| N=48 | Programação Dinâmica | 48 | 2000 | 0.25540 | 1,138.16699 |
| N=48 | Greedy | 48 | 2000 | 0.01051 | 9.80469 |
| N=48 | Monte Carlo — validar | 48 | 2000 | 0.13878 | 6,402.75000 |
| S=500 | Monte Carlo — gerar choques | 24 | 500 | 0.00784 | 5,674.01074 |
| S=500 | Monte Carlo — estimar | 24 | 500 | 0.14598 | 2,883.67969 |
| S=500 | Programação Dinâmica | 24 | 500 | 0.08332 | 981.13574 |
| S=500 | Greedy | 24 | 500 | 0.00349 | 6.04688 |
| S=500 | Monte Carlo — validar | 24 | 500 | 0.02213 | 927.35156 |
| S=5000 | Monte Carlo — gerar choques | 24 | 5000 | 0.08617 | 56,720.88574 |
| S=5000 | Monte Carlo — estimar | 24 | 5000 | 1.56134 | 24,091.43750 |
| S=5000 | Programação Dinâmica | 24 | 5000 | 0.14296 | 981.13574 |
| S=5000 | Greedy | 24 | 5000 | 0.00449 | 6.04688 |
| S=5000 | Monte Carlo — validar | 24 | 5000 | 0.18042 | 8,072.85938 |

Leitura: choques/estimação/validação crescem ~linear em N·S; a DP cresce ~linear em N (e depende do estoque máximo). O Greedy é desprezível frente às demais etapas.
