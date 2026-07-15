import math
from scripts.davis_prep import base_gene, kd_to_pkd


def test_base_gene_strips_mutation_and_domain():
    assert base_gene("ABL1(F317I)") == "ABL1"
    assert base_gene("CSNK1A1") == "CSNK1A1"
    assert base_gene("MAP3K1-domain") == "MAP3K1"
    assert base_gene("JAK3(JH1domain-catalytic)") == "JAK3"


def test_kd_to_pkd():
    assert abs(kd_to_pkd(1.0) - 9.0) < 1e-9        # 1 nM -> pKd 9
    assert abs(kd_to_pkd(10000.0) - 5.0) < 1e-9    # 10 uM (non-binder) -> pKd 5
