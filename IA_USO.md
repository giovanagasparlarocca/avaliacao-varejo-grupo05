# Uso de IA generativa (registro exigido pelo enunciado, seção 3)

**Ferramenta:** Claude (Anthropic), em conversa com a integrante Giovana Gaspar Larocca, em 28/10/2026.

**O que a IA produziu neste repositório (rascunho inicial):**
- preenchimento do Bloco 02 do notebook gerador e execução dos blocos para gerar os arquivos de `data/`;
- código das duas questões (`questao_01/`, `questao_02/`, `comum/`), testes em `tests/`, scripts em `tools/`;
- documentação em `docs/`, `README.md` e este arquivo.


## O que o grupo alterou / validou (preencher)
| Data | Integrante | Alteração ou estudo | Arquivo/função |
| Data | Integrante | Alteração ou estudo | Arquivo/função |
|---|---|---|---|
| 28/10/2026 | Equipe | Gerou os arquivos oficiais de entrada do G05 (seed final 5010) preenchendo apenas o Bloco 02 do notebook gerador e executando os demais blocos sem alterá-los | `data/Gerador_Inputs_Checkpoint_6_G05.ipynb`, `data/manifesto_dados.json` |
| 28/10/2026 | Equipe  | Criou a leitura dos dados com verificação de SHA-256 contra o manifesto, para recusar execução com arquivos alterados | `comum/io_dados.py` · `verificar_integridade()` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Implementou o modelo econômico da Q1 com piso de margem, capacidade logística e custo de carregamento de estoque (hipóteses não previstas no enunciado) | `questao_01/modelo.py` · `construir_tabelas()`, `fitness_populacao()` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Implementou o Brute Force em blocos sobre o subconjunto de 8 SKUs, usado como benchmark ótimo | `questao_01/brute_force.py` · `forca_bruta()` |
| 28/10/2026 | Equipe  | Implementou o GA com torneio, cruzamento uniforme, mutação, elitismo e parada por paciência; comparou as configurações A e B | `questao_01/genetic_algorithm.py` · `executar_ga()` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Implementou o Greedy de incentivos por razão benefício/custo com heap | `questao_01/greedy.py` · `alocar_incentivos()` |
| 28/10/2026 | Equipe  | Trocou o fitness do GA de laço Python para NumPy vetorizado (otimização medida antes/depois) | `questao_01/modelo.py` · `fitness_populacao()` × `fitness_individuo_loop()` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Implementou o modelo da Q2 (choques de demanda, concorrente e logística) e o Monte Carlo de estimação e validação | `questao_02/modelo.py`, `questao_02/monte_carlo.py` |
| 28/10/2026 | Equipe  | Implementou a Programação Dinâmica com estado (semana, estoque), decisão de nível de preço ou pausa, e reutilização de subproblemas; ajustou o desempate da pausa e a política com estoque zero | `questao_02/dynamic_programming.py` · `resolver_sku()`, `resolver_dp()` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Implementou o Greedy semanal de destaques com benefício/custo e orçamento semanal | `questao_02/greedy.py` · `alocar_destaques()` |
| 28/10/2026 | Equipe  | Calibrou o peso de urgência λ da DP em cenários separados e fez o teste de estresse de dívida | `questao_02/main.py` · `construir_plano()`, bloco `--modo completo` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Criou a suíte de testes (29 casos), incluindo DP comparada à enumeração exaustiva e Brute Force comparado ao itertools | `tests/` |
| 28/10/2026 | Equipe (rascunho gerado com IA) | Corrigiu erro de ambiente: `pytest` não instalado; instalado com `pip install pytest --break-system-packages`; versões mínimas fixadas em `requirements.txt` | `requirements.txt` |
| | | | |
