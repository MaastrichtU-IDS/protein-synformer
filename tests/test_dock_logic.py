import numpy as np
from synformer.dock.geometry import box_from_coords, select_topm, mismatch_summary

def test_box_from_coords_center_and_size():
    coords = np.array([[0.,0.,0.],[2.,4.,6.]])
    center, size = box_from_coords(coords, padding=1.0)
    assert center == (1.0, 2.0, 3.0)
    assert size == (4.0, 6.0, 8.0)   # extent (2,4,6) + 2*padding

def test_select_topm_lowest_scores():
    assert select_topm([-5.0, -9.0, -1.0, -7.0], 2) == [1, 3]   # -9 then -7

def test_mismatch_summary_target_specific():
    # own (diagonal) much better (lower) than off-diagonal -> target-specific
    M = np.array([[-9.0, -5.0, -5.0],
                  [-5.0, -9.0, -5.0],
                  [-5.0, -5.0, -9.0]])
    s = mismatch_summary(M)
    assert s["own_mean"] == -9.0 and s["offdiag_mean"] == -5.0
    assert s["delta"] == -4.0 and s["win_rate"] == 1.0

def test_mismatch_summary_not_specific():
    M = np.full((3,3), -6.0)
    s = mismatch_summary(M)
    assert s["delta"] == 0.0 and s["win_rate"] == 0.0   # own never strictly better
