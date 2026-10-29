"""Medição de tempo (perf_counter) e memória de pico (tracemalloc)."""
import time
import tracemalloc


def medir(func, *args, repeticoes=1, medir_memoria=True, **kwargs):
    """Executa func e devolve (resultado, tempo_medio_s, pico_memoria_KiB).

    Tempo e memória são medidos em execuções separadas, porque o tracemalloc
    deixa o código mais lento e distorceria o tempo.
    """
    tempos, resultado = [], None
    for _ in range(max(1, repeticoes)):
        t0 = time.perf_counter()
        resultado = func(*args, **kwargs)
        tempos.append(time.perf_counter() - t0)
    pico = float("nan")
    if medir_memoria:
        tracemalloc.start()
        try:
            func(*args, **kwargs)
            _, pico_b = tracemalloc.get_traced_memory()
            pico = pico_b / 1024
        finally:
            tracemalloc.stop()
    return resultado, sum(tempos) / len(tempos), pico
