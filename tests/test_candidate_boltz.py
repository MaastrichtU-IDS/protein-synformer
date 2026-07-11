import pandas as pd
from scripts.candidate_boltz import stratified_sample


def test_stratified_sample_is_deterministic_and_spans_range():
    # one target, 30 candidates smina -15..+14 (strong..weak)
    df = pd.DataFrame({"target": ["T"] * 30, "molecule": [f"m{i}" for i in range(30)],
                       "smina": [float(-15 + i) for i in range(30)]})
    a = stratified_sample(df, n_per_target=6, strata=3)
    b = stratified_sample(df, n_per_target=6, strata=3)
    assert list(a.molecule) == list(b.molecule)              # deterministic
    assert len(a) == 6                                        # 2 per stratum x 3
    smi = sorted(a.smina.tolist())
    # spans strong / mid / weak: min in bottom third, max in top third
    assert smi[0] <= -10 and smi[-1] >= 9


def test_stratified_sample_per_target():
    df = pd.DataFrame({"target": ["A"] * 12 + ["B"] * 12,
                       "molecule": [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)],
                       "smina": [float(i) for i in range(12)] * 2})
    s = stratified_sample(df, n_per_target=6, strata=3)
    assert set(s.target) == {"A", "B"}
    assert (s.target == "A").sum() == 6 and (s.target == "B").sum() == 6
