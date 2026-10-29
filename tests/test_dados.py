import shutil

import pytest

from comum.io_dados import PASTA_DADOS, carregar_manifesto, carregar_questao, verificar_integridade


def test_manifesto_do_grupo_05():
    m = carregar_manifesto()
    assert m["grupo"] == "G05" and m["seed_base"] == 10 and m["seed_final"] == 5010


def test_integridade_ok():
    assert verificar_integridade() is True


def test_integridade_detecta_alteracao(tmp_path):
    for arq in PASTA_DADOS.iterdir():
        if arq.suffix in (".csv", ".json"):
            shutil.copy(arq, tmp_path / arq.name)
    with open(tmp_path / "q1_catalogo_G05.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError):
        verificar_integridade(tmp_path)


def test_tamanhos_das_instancias():
    df1, _, _ = carregar_questao(1)
    df2, _, _ = carregar_questao(2)
    assert len(df1) == 30 and len(df2) == 24
