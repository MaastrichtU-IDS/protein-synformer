"""Tier-2 data: fetch measured bioactivities (pChEMBL) from ChEMBL for the calibration targets, cached
per target so proxy timeouts don't lose progress. Captures SMILES + best pChEMBL per (compound, target)
so the same file feeds both the cross-target overlap check and the docking/analysis.

    .venv/bin/python -m scripts.tier2_fetch      # resumable; writes data/dock/tier2/raw/<target>.json

Targets = the 6 docking-working panel targets (KIT/JAK3/CDK5 kinases; 5HT1A/5HT2A/A1R GPCRs), so
within-family (paralog) and cross-family pairs are both available.
"""
import json
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGETS = {"P10721": "KIT", "P52333": "JAK3", "Q00535": "CDK5",
           "P08908": "5HT1A", "P28223": "5HT2A", "P30542": "A1R"}
TYPES = {"Ki", "Kd", "IC50", "EC50"}
OUT = Path("data/dock/tier2/raw")


def session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


def chembl_id(s, acc):
    r = s.get(f"{BASE}/target?target_components__accession={acc}&format=json", timeout=90).json()
    for t in r.get("targets", []):
        if t["target_type"] == "SINGLE PROTEIN":
            return t["target_chembl_id"]
    return None


def fetch_target(s, cid, cap=8000):
    """molecule_chembl_id -> {smiles, pchembl (best), type}."""
    out = {}
    off = 0
    while off < cap:
        url = (f"{BASE}/activity?target_chembl_id={cid}&pchembl_value__isnull=false"
               f"&limit=1000&offset={off}&format=json")
        for attempt in range(4):
            try:
                r = s.get(url, timeout=120).json()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        for a in r.get("activities", []):
            m = a.get("molecule_chembl_id")
            p = a.get("pchembl_value")
            smi = a.get("canonical_smiles")
            st = a.get("standard_type")
            if not (m and p and smi) or st not in TYPES:
                continue
            try:
                pv = float(p)
            except Exception:
                continue
            if m not in out or pv > out[m]["pchembl"]:
                out[m] = {"smiles": smi, "pchembl": pv, "type": st}
        if not r.get("page_meta", {}).get("next"):
            break
        off += 1000
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = session()
    for acc, name in TARGETS.items():
        f = OUT / f"{name}.json"
        if f.exists():
            print(f"{name}: cached ({len(json.loads(f.read_text()))} compounds) — skip", flush=True)
            continue
        cid = chembl_id(s, acc)
        data = fetch_target(s, cid) if cid else {}
        f.write_text(json.dumps({"acc": acc, "chembl_id": cid, "compounds": data}))
        print(f"{name} ({acc} -> {cid}): {len(data)} compounds with pChEMBL -> {f}", flush=True)
    print("TIER2 FETCH DONE", flush=True)


if __name__ == "__main__":
    main()
