import pandas as pd
from scripts.admet_score import load_pool, admet_pass, profile


def test_load_pool_dedups_files_and_dirs(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("CCO\n\nCCO\nc1ccccc1\n")
    assert load_pool([str(f)]) == ["CCO", "c1ccccc1"]


def test_admet_pass_clean_passes_toxic_fails():
    df = pd.DataFrame({
        "hERG": [0.1, 0.9], "DILI": [0.2, 0.2], "ClinTox": [0.1, 0.1],
        "Carcinogens_Lagunin": [0.2, 0.2], "HIA_Hou": [0.9, 0.9],
    })
    p = admet_pass(df)
    assert bool(p.iloc[0]) is True     # clean
    assert bool(p.iloc[1]) is False    # high hERG


def test_admet_pass_low_hia_fails():
    df = pd.DataFrame({"hERG":[0.1],"DILI":[0.1],"ClinTox":[0.1],
                       "Carcinogens_Lagunin":[0.1],"HIA_Hou":[0.2]})
    assert bool(admet_pass(df).iloc[0]) is False


def test_profile_pass_rate():
    df = pd.DataFrame({"hERG":[0.1,0.9],"DILI":[0.1,0.1],"ClinTox":[0.1,0.1],
                       "Carcinogens_Lagunin":[0.1,0.1],"HIA_Hou":[0.9,0.9]})
    ps = admet_pass(df)
    prof = profile(df, ps)
    assert prof["n"] == 2 and abs(prof["pass_rate"] - 0.5) < 1e-9
