"""Etapa C — Greedy: distribui o orçamento de incentivo (cupom/cashback) após a política do GA.

Modelo: um 'lote' no SKU i custa  c_i = cupom_pct * preço_i * lote_cupons  e converte
        conv_i = min(conv_max, conv_fator * elasticidade_i) por cupom em venda ADICIONAL
        (SKUs mais elásticos reagem mais), limitada pelo estoque que sobrou e pela capacidade total.
Benefício do lote = Δu * margem_unitária (caixa); razão = benefício / custo do lote.
Regra gulosa: sempre o lote de maior razão (benefício marginal por real); só entram lotes com razão >= razao_minima.

Tempo  O(L log n)  (L = lotes comprados, n SKUs; heap)        Memória O(n)
"""
import heapq

import numpy as np


def alocar_incentivos(T, genes, orcamento, params=None):
    p = params or T.params
    if orcamento < 0:
        raise ValueError("orcamento negativo.")
    ar = np.arange(T.n)
    g = np.asarray(genes)
    un = T.un[ar, g].astype(float).copy()
    preco = T.preco[ar, g]
    mu = T.mu[ar, g]
    vend = T.nivel[ar, g] >= 0
    rem = T.est - un
    lote = p["lote_cupons"]
    custo_lote = p["cupom_pct"] * preco * lote
    conv = np.minimum(p["conv_max"], p["conv_fator"] * T.e)
    ganho_un = mu                      # decisão pelo ganho de CAIXA (carregamento evitado é só reportado)
    lotes = np.zeros(T.n, int)
    extra = np.zeros(T.n)
    gasto, total_un = 0.0, float(un.sum())

    def du_atual(i):
        return max(0.0, min(conv[i] * lote, rem[i], T.cap_total - total_un))

    def razao(i):
        d = du_atual(i)
        return d * ganho_un[i] / custo_lote[i] if d > 0 and custo_lote[i] > 0 else 0.0

    heap = [(-razao(i), i) for i in range(T.n) if vend[i]]
    heapq.heapify(heap)
    while heap:
        r_neg, i = heapq.heappop(heap)
        r = razao(i)
        if r < p["razao_minima"]:
            continue
        if abs(-r_neg - r) > 1e-12:                 # prioridade desatualizada: reinsere
            heapq.heappush(heap, (-r, i))
            continue
        if gasto + custo_lote[i] > orcamento + 1e-9:
            continue                                # não cabe; talvez caiba um mais barato
        d = du_atual(i)
        gasto += custo_lote[i]
        extra[i] += d
        rem[i] -= d
        total_un += d
        lotes[i] += 1
        heapq.heappush(heap, (-razao(i), i))
    ganho_bruto = float((extra * mu).sum())
    carr_evitado = float((extra * T.carry_unit).sum())
    return {"lotes": lotes, "unidades_extra": extra, "gasto": gasto, "orcamento": orcamento,
            "ganho_bruto": ganho_bruto, "carregamento_evitado": carr_evitado,
            "ganho_liquido": ganho_bruto + carr_evitado - gasto,
            "receita_extra": float((extra * preco).sum())}
