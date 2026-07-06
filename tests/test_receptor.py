import os
from synformer.dock.receptor import prepare_target


def test_prepare_target_writes_receptor_and_ligand(tmp_path):
    spec = prepare_target("1STP", str(tmp_path), ligand_resname="BTN")  # streptavidin/biotin
    assert os.path.getsize(spec.receptor_path) > 0
    assert os.path.getsize(spec.ref_ligand_path) > 0
    # receptor has protein atoms, ref ligand has the small molecule
    assert open(spec.receptor_path).read().count("ATOM") > 100
    assert "HETATM" in open(spec.ref_ligand_path).read()
