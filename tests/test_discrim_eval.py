from scripts.discrim_eval import winrate


def test_winrate():
    # (ll_bind, ll_nonbind): 2 correct, 1 wrong -> 2/3
    assert abs(winrate([(1.0, 0.0), (2.0, 1.0), (0.0, 1.0)]) - 2 / 3) < 1e-9
