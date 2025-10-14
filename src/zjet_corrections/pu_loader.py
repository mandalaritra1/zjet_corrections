import gzip
import json
from functools import lru_cache
from importlib.resources import files
import numpy as np
import awkward as ak
import correctionlib

# Map your IOV to the key used inside the JSON
_HNAME = {
    "2016APV": "Collisions16_UltraLegacy_goldenJSON",
    "2016"   : "Collisions16_UltraLegacy_goldenJSON",
    "2017"   : "Collisions17_UltraLegacy_goldenJSON",
    "2018"   : "Collisions18_UltraLegacy_goldenJSON",
}

@lru_cache(maxsize=None)
def _load_cset(iov: str) -> correctionlib.CorrectionSet:
    """Load and cache the CorrectionSet from packaged resources."""
    pkg_path = f"zjet_corrections/pu/{iov}_UL/puWeights.json.gz"
    with files("zjet_corrections").joinpath(f"pu/{iov}_UL/puWeights.json.gz").open("rb") as fh:
        raw = fh.read()
    data = gzip.decompress(raw).decode("utf-8")
    return correctionlib.CorrectionSet.from_string(data)

def get_pu_weights(events, iov: str):
    """Return (puNom, puUp, puDown) numpy arrays for given events and IOV."""
    cset = _load_cset(iov)
    key = _HNAME[iov]
    corr = cset[key]

    # robust extraction of nTrueInt as numpy
    ntrue = ak.to_numpy(events.Pileup.nTrueInt)
    # Some JSONs accept (ntrue, "variation")
    pu_nom  = corr.evaluate(ntrue, "nominal")
    pu_up   = corr.evaluate(ntrue, "up")
    pu_down = corr.evaluate(ntrue, "down")
    return pu_nom, pu_up, pu_down
