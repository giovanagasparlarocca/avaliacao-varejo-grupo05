"""Leitura e verificação de integridade dos arquivos oficiais de entrada (Grupo 05)."""
import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data"


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_json(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_manifesto(pasta=PASTA_DADOS):
    return carregar_json(Path(pasta) / "manifesto_dados.json")


def verificar_integridade(pasta=PASTA_DADOS):
    """Recalcula os SHA-256 e compara com o manifesto. Levanta ValueError se divergir."""
    pasta = Path(pasta)
    man = carregar_manifesto(pasta)
    for nome, hash_esperado in man["arquivos"].items():
        arq = pasta / nome
        if not arq.exists():
            raise ValueError(f"Arquivo oficial ausente: {nome}")
        if sha256_arquivo(arq) != hash_esperado:
            raise ValueError(f"Hash diferente do manifesto (arquivo alterado?): {nome}")
    return True


def carregar_questao(questao, pasta=PASTA_DADOS):
    """Retorna (DataFrame do catálogo, dict de configuração, manifesto) da questão 1 ou 2."""
    pasta = Path(pasta)
    man = carregar_manifesto(pasta)
    g = man["grupo"]
    df = pd.read_csv(pasta / f"q{questao}_catalogo_{g}.csv", encoding="utf-8-sig")
    cfg = carregar_json(pasta / f"q{questao}_config_{g}.json")
    return df, cfg, man
