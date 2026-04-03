import gzip
import json
from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import files, as_file
from pathlib import Path
from typing import Optional

import numpy as np
import awkward as ak
import correctionlib
import tempfile
from coffea.lumi_tools import LumiMask
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools import extractor

from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .roccor import RoccoR

# Map your IOV to the key used inside the JSON
_HNAME = {
    "2016APV": "Collisions16_UltraLegacy_goldenJSON",
    "2016"   : "Collisions16_UltraLegacy_goldenJSON",
    "2017"   : "Collisions17_UltraLegacy_goldenJSON",
    "2018"   : "Collisions18_UltraLegacy_goldenJSON",
}

_CORRLIB_NAME_MAP = {
    "2016APV": "2016preVFP_UL",
    "2016"   : "2016postVFP_UL",
    "2017"   : "2017_UL",
    "2018"   : "2018_UL",
}

_ELE_SF_YEAR_LABEL = {
    "2016APV": "2016preVFP",
    "2016"   : "2016postVFP",
    "2017"   : "2017",
    "2018"   : "2018",
}

_JME_MODE_FILE = {
    "AK8": "fatJet_jerc.json.gz",
    "AK4": "jet_jerc.json.gz",
}

_JME_AK_LABEL = {
    "AK8": "AK8PFPuppi",
    "AK4": "AK4PFchs",
}

_JEC_DATA_TAGS = {
    "2016APV": {
        "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
        "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
        "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
        "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
        "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
    },
    "2016": {
        "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
        "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
        "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
    },
    "2017": {
        "Run2017B": "Summer19UL17_RunB_V5_DATA",
        "Run2017C": "Summer19UL17_RunC_V5_DATA",
        "Run2017D": "Summer19UL17_RunD_V5_DATA",
        "Run2017E": "Summer19UL17_RunE_V5_DATA",
        "Run2017F": "Summer19UL17_RunF_V5_DATA",
    },
    "2018": {
        "Run2018A": "Summer19UL18_RunA_V5_DATA",
        "Run2018B": "Summer19UL18_RunB_V5_DATA",
        "Run2018C": "Summer19UL18_RunC_V5_DATA",
        "Run2018D": "Summer19UL18_RunD_V5_DATA",
    },
}


def _resolve_jec_tags(iov: str):
    if iov == "2018":
        return (
            "Summer19UL18_V5_MC",
            {
                "Run2018A": "Summer19UL18_RunA_V6_DATA",
                "Run2018B": "Summer19UL18_RunB_V6_DATA",
                "Run2018C": "Summer19UL18_RunC_V6_DATA",
                "Run2018D": "Summer19UL18_RunD_V6_DATA",
            },
            "Summer19UL18_JRV2_MC",
        )
    if iov == "2017":
        return (
            "Summer19UL17_V5_MC",
            {
                "Run2017B": "Summer19UL17_RunB_V6_DATA",
                "Run2017C": "Summer19UL17_RunC_V6_DATA",
                "Run2017D": "Summer19UL17_RunD_V6_DATA",
                "Run2017E": "Summer19UL17_RunE_V6_DATA",
                "Run2017F": "Summer19UL17_RunF_V6_DATA",
            },
            "Summer19UL17_JRV3_MC",
        )
    if iov == "2016":
        return (
            "Summer19UL16_V7_MC",
            {
                "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
                "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
                "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
            },
            "Summer20UL16_JRV3_MC",
        )
    if iov == "2016APV":
        return (
            "Summer19UL16_V7_MC",
            {
                "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
                "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
                "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
                "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
                "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
            },
            "Summer20UL16APV_JRV3_MC",
        )
    raise ValueError(f"Unsupported IOV '{iov}' for jet energy corrections")

class PtRhoWeighter:
    def __init__(self, npz_path: str):
        dat = np.load(npz_path, allow_pickle=True)
        self.pt_edges = np.asarray(dat["pt_edges"], dtype=float)
        self.rho_grids = dat["rho_grids"]   # object array
        self.w_grids = dat["w_grids"]       # object array

    def _pt_bin(self, pt: float) -> int:
        k = np.searchsorted(self.pt_edges, pt, side="right") - 1
        return int(np.clip(k, 0, len(self.pt_edges) - 2))

    def weight(self, pt: float, rho: float) -> float:
        k = self._pt_bin(pt)
        x = np.asarray(self.rho_grids[k], dtype=float)
        w = np.asarray(self.w_grids[k], dtype=float)
        return float(np.interp(rho, x, w, left=w[0], right=w[-1]))

    def weight_array(self, pt_arr, rho_arr):
        pt_arr = np.asarray(pt_arr, dtype=float)
        rho_arr = np.asarray(rho_arr, dtype=float)
        out = np.empty_like(rho_arr, dtype=float)

        k_arr = np.searchsorted(self.pt_edges, pt_arr, side="right") - 1
        k_arr = np.clip(k_arr, 0, len(self.pt_edges) - 2)

        for k in range(len(self.pt_edges) - 1):
            mask = (k_arr == k)
            if not np.any(mask):
                continue
            x = np.asarray(self.rho_grids[k], dtype=float)
            w = np.asarray(self.w_grids[k], dtype=float)
            out[mask] = np.interp(rho_arr[mask], x, w, left=w[0], right=w[-1])

        return out
        
from importlib.resources import files, as_file

@lru_cache(maxsize=None)
def get_herwig_weight_g() -> PtRhoWeighter:
    resource = files("zjet_corrections") / "corrections" / "spline_groomed.npz"
    with as_file(resource) as p:          # p is a real pathlib.Path on disk
        return PtRhoWeighter(p)

@lru_cache(maxsize=None)
def get_herwig_weight_u() -> PtRhoWeighter:
    resource = files("zjet_corrections") / "corrections" / "spline_ungroomed.npz"
    with as_file(resource) as p:
        return PtRhoWeighter(p)

@lru_cache(maxsize=None)
def get_rocc_corrections(iov=None) -> RoccoR:
    # you can map iov -> filename later if you want
    resource = (
        files("zjet_corrections")
        / "corrections" / "muonSF" / "UL2016" / "RoccoR2016aUL.txt"
    )
    with as_file(resource) as p:
        return RoccoR(str(p))   # str is safe; RoccoR opens the path

@lru_cache(maxsize=None)
def _load_jme_corrections(iov: str, mode: str):
    """Return a {name: correctionlib_wrapper} dict for the requested IOV and jet mode."""
    mode_key = mode.upper()
    if mode_key not in _JME_MODE_FILE:
        raise ValueError(f"Unsupported jet mode '{mode}'. Expected one of {tuple(_JME_MODE_FILE)}.")
    if iov not in _CORRLIB_NAME_MAP:
        raise ValueError(f"Unsupported IOV '{iov}'. Expected one of {tuple(_CORRLIB_NAME_MAP)}.")

    data_dir = files("zjet_corrections") / "corrections" / "JME" / _CORRLIB_NAME_MAP[iov]
    data_path = data_dir / _JME_MODE_FILE[mode_key]
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing JME correction file at {data_path}")

    cset = correctionlib.CorrectionSet.from_file(str(data_path))
    return {name: correctionlib_wrapper(cset[name]) for name in cset.keys()}

@lru_cache(maxsize=None)
def _load_cset(iov: str) -> correctionlib.CorrectionSet:
    """Load and cache the CorrectionSet from packaged resources."""
    data_path = files("zjet_corrections") / "corrections" / "pu" / f"{iov}_UL" / "puWeights.json.gz"
    with data_path.open("rb") as fh:
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


@lru_cache(maxsize=None)
def _load_ele_trig_corrections() -> correctionlib.CorrectionSet:
    """Load electron trigger efficiency corrections from packaged resources."""
    data_path = files("zjet_corrections") / "corrections" / "eleSF" / "egammaEffi_EGM2D.json"
    with data_path.open("r", encoding="utf-8") as fh:
        return correctionlib.CorrectionSet.from_string(fh.read())


@lru_cache(maxsize=None)
def _load_ele_sf_corrections(iov: str) -> correctionlib.CorrectionSet:
    """Load electron identification scale factors from packaged resources."""
    period = _CORRLIB_NAME_MAP[iov]
    data_path = files("zjet_corrections") / "corrections" / "EGM" / period / "electron.json.gz"
    with data_path.open("rb") as fh:
        raw = fh.read()
    return correctionlib.CorrectionSet.from_string(gzip.decompress(raw).decode("utf-8"))


@lru_cache(maxsize=None)
def _load_muon_sf_corrections(iov: str) -> correctionlib.CorrectionSet:
    """Load muon scale factor corrections from packaged resources."""
    data_path = files("zjet_corrections") / "corrections" / "muonSF" / f"UL{iov}" / "muon_Z.json.gz"
    with data_path.open("rb") as fh:
        raw = fh.read()
    return correctionlib.CorrectionSet.from_string(gzip.decompress(raw).decode("utf-8"))


def GetPDFweights(df):
    """Return (pdf_nominal, pdf_up, pdf_down) arrays for PDF variations."""
    pdf = ak.ones_like(df.Pileup.nTrueInt)
    if "LHEPdfWeight" in ak.fields(df):
        pdfUnc = ak.std(df.LHEPdfWeight, axis=1) / ak.mean(df.LHEPdfWeight, axis=1)
    else:
        pdfUnc = ak.zeros_like(pdf)
    return pdf, pdf + pdfUnc, pdf - pdfUnc

def GetL1PreFiringWeight(IOV, df):
    """Return (nominal, up, down) L1 prefiring weights from the event record."""
    if "L1PreFiringWeight" not in ak.fields(df):
        ones = np.ones(len(df), dtype=np.float64)
        return ones, ones, ones

    weights = df["L1PreFiringWeight"]
    nom = ak.to_numpy(ak.fill_none(weights["Nom"], 1.0))
    up = ak.to_numpy(ak.fill_none(weights["Up"], 1.0))
    down = ak.to_numpy(ak.fill_none(weights["Dn"], 1.0))
    return nom, up, down


def GetPSweights(df, shower = "ISR"):
    """ Return nominal, up, down weights for ISR or FSR """
    if shower == "ISR":
        ones = ak.ones_like(df.event)
        try:
            return ones, df.PSWeight[:,0], df.PSWeight[:,2]
        except:
            return ones, ones, ones
    elif shower == "FSR":
        ones = ak.ones_like(df.event)
        try:
            return ones, df.PSWeight[:,1], df.PSWeight[:,3]
        except:
            return ones, ones, ones

def GetQ2weights(df, var="nominal"):
    q2 = ak.ones_like(df.event, dtype = float)
    q2_up = ak.ones_like(df.event, dtype = float)
    q2_down = ak.ones_like(df.event, dtype = float)
    if ("LHEScaleWeight" in ak.fields(df)):
        if ak.all(ak.num(df.LHEScaleWeight, axis=1) == 9):
            nom = df.LHEScaleWeight[:, 4]
            scales = df.LHEScaleWeight[:, [0, 1, 3, 5, 7, 8]]
            q2_up = ak.max(scales, axis=1) / nom
            q2_down = ak.min(scales, axis=1) / nom
        elif ak.all(ak.num(df.LHEScaleWeight, axis=1) == 8):
            scales = df.LHEScaleWeight[:, [0, 1, 3, 4, 6, 7]]
            q2_up = ak.max(scales, axis=1)
            q2_down = ak.min(scales, axis=1)

    return q2, q2_up, q2_down
    
def GetEleTrigEff(IOV, lep0pT, lep0eta):
    """Return (nominal, up, down) electron trigger efficiencies."""
    counts = ak.num(lep0pT)
    ceval = _load_ele_trig_corrections()

    flat_eta = ak.flatten(lep0eta)
    flat_pt = ak.flatten(lep0pT)

    sf_nom = ceval["pt_reweight"].evaluate(flat_eta, flat_pt)
    sf_up = ceval["pt_reweight_up"].evaluate(flat_eta, flat_pt)
    sf_down = ceval["pt_reweight_down"].evaluate(flat_eta, flat_pt)

    return (
        ak.unflatten(sf_nom, counts),
        ak.unflatten(sf_up, counts),
        ak.unflatten(sf_down, counts),
    )
    
# def getRapidity(p4):
#     return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))
def getRapidity(obj, eps=1e-12):
    pt, eta, m = obj.pt, obj.eta, obj.mass
    pz = pt * np.sinh(eta)
    E  = np.sqrt((pt * np.cosh(eta))**2 + m**2)
    # print("[IN FUNCTION] Computing rapidity")
    # print("[IN FUNCTION] Energy ", E)
    # print("[IN FUNCTION] pz", pz)

    num = E + pz
    den = E - pz
    # print("[IN FUNCTION] num ", num)
    # print("[IN FUNCTION] den ", den)
    den = ak.where(den > eps, den, np.nan)
    num = ak.where(num > eps, num, np.nan)
    rapidity = 0.5 * np.log(num / den)
    # print("[IN FUNCTION] Rapidity ", rapidity)
    # print("[IN FUNCTION] Eta ", eta)
    return rapidity
    




# def compute_rapidity(obj):
#     """
#     Compute rapidity for NanoAOD objects with pt, eta, phi, mass
#     Works with awkward arrays (coffea NanoEvents).
#     """

#     pt = obj.pt
#     eta = obj.eta
#     mass = obj.mass

#     pz = pt * np.sinh(eta)
#     energy = np.sqrt((pt * np.cosh(eta))**2 + mass**2)

#     y = 0.5 * np.log((energy + pz) / (energy - pz))
#     return y
def compute_rapidity(obj, eps=1e-12):
    pt = obj.pt
    eta = obj.eta
    m = obj.mass

    pz = pt * np.sinh(eta)
    E  = np.sqrt((pt * np.cosh(eta))**2 + m**2)

    num = E + pz
    den = E - pz

    # protect against den <= 0 (or tiny) from weird values / rounding
    den = ak.where(den > eps, den, np.nan)
    num = ak.where(num > eps, num, np.nan)

    return 0.5 * np.log(num / den)
def GetEleSF(IOV, wp, eta, pt):
    """Return (nominal, systup, systdown) electron identification scale factors."""
    counts = ak.num(pt)
    evaluator = _load_ele_sf_corrections(IOV)

    mask = pt > 20
    adj_pt = ak.where(mask, pt, 22)

    flat_eta = np.array(ak.flatten(eta))
    flat_pt = np.array(ak.flatten(adj_pt))

    year = _ELE_SF_YEAR_LABEL[IOV]

    sf_nom_flat = evaluator["UL-Electron-ID-SF"].evaluate(year, "sf", wp, flat_eta, flat_pt)
    sf_up_flat  = evaluator["UL-Electron-ID-SF"].evaluate(year, "sfup", wp, flat_eta, flat_pt)
    sf_down_flat= evaluator["UL-Electron-ID-SF"].evaluate(year, "sfdown", wp, flat_eta, flat_pt)

    sf_nom  = ak.unflatten(sf_nom_flat, counts)
    sf_up   = ak.unflatten(sf_up_flat, counts)
    sf_down = ak.unflatten(sf_down_flat, counts)

    return (
        ak.where(mask, sf_nom, ak.ones_like(sf_nom)),
        ak.where(mask, sf_up,  ak.ones_like(sf_up)),
        ak.where(mask, sf_down,ak.ones_like(sf_down)),
    )


def GetMuonSF(IOV, corrset, abseta, pt):
    """Return (nominal, systup, systdown) muon scale factors for the requested set."""
    counts = ak.num(pt)
    evaluator = _load_muon_sf_corrections(IOV)

    # clamp to valid phase space covered by the correction files
    adj_abseta = ak.where(abseta < 2.4, abseta, 2.39)
    adj_pt = pt

    if corrset == "RECO":
        hname = "NUM_GlobalMuons_DEN_genTracks"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    elif corrset == "HLT":
        if IOV == "2016" or IOV == "2016APV":
            hname = "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight"
            adj_pt = ak.where(adj_pt < 26, 26.1, adj_pt)
        elif IOV == "2017":
            hname = "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight"
            adj_pt = ak.where(adj_pt < 29, 29.1, adj_pt)
        elif IOV == "2018":
            hname = "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight"
            adj_pt = ak.where(adj_pt < 26, 26.1, adj_pt)
        else:
            raise ValueError(f"Unsupported IOV '{IOV}' for HLT muon scale factors")
    elif corrset == "ID":
        hname = "NUM_TightID_DEN_TrackerMuons"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    elif corrset == "ISO":
        hname = "NUM_TightRelIso_DEN_TightIDandIPCut"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    else:
        raise ValueError(f"Unsupported corrset '{corrset}' for muon scale factors")

    flat_eta = np.array(ak.flatten(adj_abseta))
    flat_pt = np.array(ak.flatten(adj_pt))
    wrapper = evaluator[hname]

    sf_nom = wrapper.evaluate(flat_eta, flat_pt, "nominal")
    sf_up = wrapper.evaluate(flat_eta, flat_pt, "systup")
    sf_down = wrapper.evaluate(flat_eta, flat_pt, "systdown")

    return (
        ak.unflatten(sf_nom, counts),
        ak.unflatten(sf_up, counts),
        ak.unflatten(sf_down, counts),
    )

from contextlib import ExitStack
from importlib.resources import files, as_file
from pathlib import Path

# One global stack per worker process keeps extracted files alive.
_resource_stack = ExitStack()

def resource_path(package: str, *parts: str) -> str:
    """
    Return a real filesystem path for a resource inside `package`.
    Works even when the package is a .zip (zipimport).
    The underlying temp file is kept alive by the process-global ExitStack.
    """
    ref = files(package).joinpath(*parts)
    path = _resource_stack.enter_context(as_file(ref))
    return str(path)

# def getLumiMaskRun2():


#     golden_json_dir = files("zjet_corrections") / "corrections" / "goldenJsons"
#     golden_json_path_2016 = golden_json_dir / "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
#     golden_json_path_2017 = golden_json_dir / "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
#     golden_json_path_2018 = golden_json_dir / "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"

#     masks = {
#         "2016APV": LumiMask(str(golden_json_path_2016)),
#         "2016": LumiMask(str(golden_json_path_2016)),
#         "2017": LumiMask(str(golden_json_path_2017)),
#         "2018": LumiMask(str(golden_json_path_2018)),
#     }

#     return masks

def getLumiMaskRun2():
    pkg = "zjet_corrections"  # package root

    golden_json_path_2016 = resource_path(pkg, "corrections", "goldenJsons",
        "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
    golden_json_path_2017 = resource_path(pkg, "corrections", "goldenJsons",
        "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
    golden_json_path_2018 = resource_path(pkg, "corrections", "goldenJsons",
        "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")

    masks = {
        "2016APV": LumiMask(golden_json_path_2016),
        "2016"   : LumiMask(golden_json_path_2016),
        "2017"   : LumiMask(golden_json_path_2017),
        "2018"   : LumiMask(golden_json_path_2018),
    }
    return masks


def debug_jec_weightset(iov: str = "2018", mode: str = "AK8", is_data: bool = False, run: Optional[str] = None):
    """Small helper to test reading a single JEC text file with coffea's extractor."""
    mode_key = mode.upper()
    if mode_key not in _JME_AK_LABEL:
        raise ValueError(f"Unsupported jet mode '{mode}'. Expected one of {tuple(_JME_AK_LABEL)}.")

    ak_label = _JME_AK_LABEL[mode_key]
    jec_tag, jec_tag_data, _ = _resolve_jec_tags(iov)

    corrections_root = files("zjet_corrections").joinpath("corrections")
    if is_data:
        if run is None:
            raise ValueError("Parameter 'run' must be provided when is_data=True.")
        if run not in jec_tag_data:
            raise KeyError(f"Run '{run}' not available for IOV '{iov}'.")
        tag = jec_tag_data[run]
        target_resource = corrections_root.joinpath("JEC", tag, f"{tag}_L1FastJet_{ak_label}.jec.txt")
    else:
        target_resource = corrections_root.joinpath("JEC", jec_tag, f"{jec_tag}_L1FastJet_{ak_label}.jec.txt")

    exists_in_package = target_resource.is_file()

    with ExitStack() as stack:
        tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        target_path = tmp_dir / target_resource.name
        with target_resource.open("rb") as src, target_path.open("wb") as dst:
            dst.write(src.read())
        ext = extractor()
        ext.add_weight_sets([f"* * {target_path.as_posix()}"])
        ext.finalize()
        evaluator = ext.make_evaluator()
        keys = list(evaluator.keys())

    return {
        "path": target_path.as_posix(),
        "resource_exists": exists_in_package,
        "evaluator_keys": keys,
    }


def jmssf(IOV, FatJet,  var = ''):
    jmsSF = {

        "2016APV":{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906}, 

        "2016"   :{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906}, 

        "2017"   :{"sf": 0.982, "sfup": 0.986, "sfdown": 0.978},

        "2018"   :{"sf": 0.999, "sfup": 1.001, "sfdown": 0.997}} 
    
    out = jmsSF[IOV]["sf"+var]
    

    FatJet = ak.with_field(FatJet, FatJet.mass * out, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * out, 'msoftdrop')
    return FatJet

def jmrsf(IOV, FatJet, var = ''):
    jmrSF = {

       #"2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
        "2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 
        "2016"   :{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8}, 

        "2017"   :{"sf": 1.09, "sfup": 1.14, "sfdown": 1.04},

        "2018"   :{"sf": 1.108, "sfup": 1.142, "sfdown": 1.074}}     
    
    jmrvalnom = jmrSF[IOV]["sf"+var]
    print("How many none in Fatjet.mass before processing inside jmrnom", ak.sum(ak.is_none(ak.firsts(FatJet).mass)))
    recomass = FatJet.mass
    genmass = FatJet.matched_gen.mass

    print("Genmass inside jmrsf", genmass)

    print("How many none in Fatjet.matched_gen inside jmrnom", ak.sum(ak.is_none(FatJet.matched_gen.mass)))
    # counts = ak.num(recomass)
    # recomass = ak.flatten(recomass)
    # genmass = ak.flatten(genmass)
    
    
    deltamass = (recomass-genmass)*(jmrvalnom-1.0)
    condition = ((recomass+deltamass)/recomass) > 0
    jmrnom = ak.where( recomass <= 0.0, 0 , ak.where( condition , (recomass+deltamass)/recomass, 0 ))
    print(jmrnom)
    print("How many none in Fatjet inside jmrnom", ak.sum(ak.is_none(FatJet.mass)))

    FatJet = ak.with_field(FatJet, FatJet.mass * jmrnom, 'mass')
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * jmrnom, 'msoftdrop')
    
    print("How many none in Fatjet.mass that is being returned inside jmrnom", ak.sum(ak.is_none(ak.firsts(FatJet).mass)))
    return FatJet


def GetJetCorrections(FatJets, events, era, IOV, isData=False, uncertainties = None, mode = 'AK8' ):
    AK_str = 'AK8PFPuppi'
    if mode == 'AK4':
        AK_str = 'AK4PFPuppi'

    print(f"Using TAG {AK_str}")
    if uncertainties == None:
        uncertainty_sources = ["AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample", "RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    else:
        uncertainty_sources = uncertainties
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    jec_tag, jec_tag_data, jer_tag = _resolve_jec_tags(IOV)


    #print("extracting corrections from files for " + jec_tag)
    ext = extractor()
    pkg_root = files("zjet_corrections")
    corrections_root = pkg_root.joinpath("corrections")
    jec_dir = corrections_root.joinpath("JEC", jec_tag)
    data_file = jec_dir.joinpath(f"{jec_tag}_L1FastJet_{AK_str}.jec.txt")
    #print("File exists check for JEC: ", str(data_file))
    #print(data_file.is_file())

    with ExitStack() as stack:
        tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        cache = {}

        def resource_file(*parts: str) -> Path:
            traversable = corrections_root.joinpath(*parts)
            if not traversable.is_file():
                raise FileNotFoundError(f"Missing correction resource at {traversable}")
            key = "/".join(parts)
            if key not in cache:
                target_path = tmp_dir / traversable.name
                with traversable.open("rb") as src, target_path.open("wb") as dst:
                    dst.write(src.read())
                cache[key] = target_path
            return cache[key]

        if not isData:
        #For MC
            ext.add_weight_sets([
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_L1FastJet_{AK_str}.jec.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_L2Relative_{AK_str}.jec.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_L3Absolute_{AK_str}.jec.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_UncertaintySources_{AK_str}.junc.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_Uncertainty_{AK_str}.junc.txt").as_posix(),
            ])
            #### Do AK8PUPPI jer files exist??
            if jer_tag:
                ext.add_weight_sets([
                '* * ' + resource_file("JER", jer_tag, f"{jer_tag}_PtResolution_{AK_str}.jr.txt").as_posix(),
                '* * ' + resource_file("JER", jer_tag, f"{jer_tag}_SF_{AK_str}.jersf.txt").as_posix()])
                # print("JER SF added")
        else:       
            
            #For data, make sure we don't duplicate
            tags_done = []
            print("In the DATA section")
            for run, tag in jec_tag_data.items():
                if not (tag in tags_done):

                    #print("Doing", tag, AK_str)
                    ext.add_weight_sets([
                    '* * ' + resource_file("JEC", tag, f"{tag}_L1FastJet_{AK_str}.jec.txt").as_posix(),
                    '* * ' + resource_file("JEC", tag, f"{tag}_L2Relative_{AK_str}.jec.txt").as_posix(),
                    '* * ' + resource_file("JEC", tag, f"{tag}_L3Absolute_{AK_str}.jec.txt").as_posix(),
                    '* * ' + resource_file("JEC", tag, f"{tag}_L2L3Residual_{AK_str}.jec.txt").as_posix(),
                    ])
                    tags_done += [tag]
                    #print("Done", tag, AK_str)
            print("Added JEC weight sets")

        
        ext.finalize()

        evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_{1}'.format(jec_tag, AK_str),
            '{0}_L2Relative_{1}'.format(jec_tag, AK_str),
            '{0}_L3Absolute_{1}'.format(jec_tag, AK_str)]
        #### if jes in arguments add total uncertainty values for comparison and easy plotting
        if 'jes' in uncertainty_sources:
            jec_names.extend(['{0}_Uncertainty_{1}'.format(jec_tag, AK_str)])
            uncertainty_sources.remove('jes')
        jec_names.extend(['{0}_UncertaintySources_{1}_{2}'.format(jec_tag, AK_str, unc_src) for unc_src in uncertainty_sources])

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}'.format(jer_tag, AK_str),
                              '{0}_SF_{1}'.format(jer_tag, AK_str)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}'.format(tag, AK_str),
                '{0}_L3Absolute_{1}'.format(tag, AK_str),
                '{0}_L2Relative_{1}'.format(tag, AK_str),
                '{0}_L2L3Residual_{1}'.format(tag, AK_str),]



    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}


    # print("jec_input", jec_inputs)
    jec_stack = JECStack(jec_inputs)
    
    
    if not isData:
        if mode == 'AK8':
            FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)
        if mode == 'AK4':        
            SubGenJetAK8 = events.SubGenJetAK8
            SubGenJetAK8['p4'] = ak.with_name(SubGenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.p4.nearest(SubGenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
    if mode == 'AK4':          
        FatJets['area'] = ak.full_like( FatJets.pt, 0.503)
    
    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['jec_rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
    #print("Rho value for jets ", events.fixedGridRhoFastjetAll)

    

        

        # if not isData:
        #     SubJets['pt_gen'] = ak.values_astype(ak.fill_none(SubJets.matched_gen.pt, 0), np.float32)
    name_map = jec_stack.blank_name_map
    #print("N events missing pt entry ", ak.sum(ak.num(FatJets.pt)<1))
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['Rho'] = 'jec_rho'

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)


    corrected_jets = jet_factory.build(FatJets)
    # print("Available uncertainties: ", jet_factory.uncertainties())
    # print("Corrected jets object: ", corrected_jets.fields)
    #print("pt and mass before correction ", FatJets['pt_raw'], ", ", FatJets['mass_raw'], " and after correction ", corrected_jets["pt"], ", ", corrected_jets["mass"])
    return corrected_jets    

def pad_none_lep(array):
    """Pad an awkward array of leptons to have exactly two entries per event, filling with None."""
    return ak.pad_none(array, 2, axis=1, clip=True)
    
def add_lepton_weights(events_j, twoReco_ee_sel, twoReco_mm_sel, weights, IOV):
    elereco_nom, elereco_up, elereco_down = GetEleSF(IOV, "RecoAbove20", events_j.Electron.eta, events_j.Electron.pt)
    elereco_nom, elereco_up, elereco_down = pad_none_lep(elereco_nom), pad_none_lep(elereco_up), pad_none_lep(elereco_down)       
    elereco_nom = ak.where(twoReco_ee_sel, elereco_nom[:,0] * elereco_nom[:,1], 1.0)
    elereco_up = ak.where(twoReco_ee_sel, elereco_up[:,0] * elereco_up[:,1], 1.0)
    elereco_down = ak.where(twoReco_ee_sel, elereco_down[:,0] * elereco_down[:,1], 1.0)
    weights.add(name = "elereco", weight = elereco_nom, weightUp = elereco_up, weightDown = elereco_down)

    eleid_nom, eleid_up, eleid_down = GetEleSF(IOV, "Tight", events_j.Electron.eta, events_j.Electron.pt)
    eleid_nom, eleid_up, eleid_down = pad_none_lep(eleid_nom), pad_none_lep(eleid_up), pad_none_lep(eleid_down)
    eleid_nom = ak.where(twoReco_ee_sel, eleid_nom[:,0] * eleid_nom[:,1], 1.0)
    eleid_up = ak.where(twoReco_ee_sel, eleid_up[:,0] * eleid_up[:,1], 1.0)
    eleid_down = ak.where(twoReco_ee_sel, eleid_down[:,0] * eleid_down[:,1], 1.0)
    weights.add(name = "eleid", weight = eleid_nom, weightUp = eleid_up, weightDown = eleid_down)

    eletrig_nom, eletrig_up, eletrig_down = GetEleTrigEff(IOV, events_j.Electron.pt, events_j.Electron.eta)
    eletrig_nom, eletrig_up, eletrig_down = pad_none_lep(eletrig_nom), pad_none_lep(eletrig_up), pad_none_lep(eletrig_down)
    eletrig_nom = ak.where(twoReco_ee_sel, eletrig_nom[:,0] * eletrig_nom[:,1], 1.0)
    eletrig_up = ak.where(twoReco_ee_sel, eletrig_up[:,0] * eletrig_up[:,1], 1.0)
    eletrig_down = ak.where(twoReco_ee_sel, eletrig_down[:,0] * eletrig_down[:,1], 1.0)
    weights.add(name = "eletrig", weight = eletrig_nom, weightUp = eletrig_up, weightDown = eletrig_down)


    # Muon SFs in the same style as electrons
    mureco_nom, mureco_up, mureco_down = GetMuonSF(IOV, "RECO", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    mureco_nom, mureco_up, mureco_down = pad_none_lep(mureco_nom), pad_none_lep(mureco_up), pad_none_lep(mureco_down)
    mureco_nom = ak.where(twoReco_mm_sel, mureco_nom[:,0] * mureco_nom[:,1], 1.0)
    mureco_up  = ak.where(twoReco_mm_sel, mureco_up[:,0]  * mureco_up[:,1],  1.0)
    mureco_down= ak.where(twoReco_mm_sel, mureco_down[:,0]* mureco_down[:,1],1.0)
    weights.add(name="mureco", weight=mureco_nom, weightUp=mureco_up, weightDown=mureco_down)

    muid_nom, muid_up, muid_down = GetMuonSF(IOV, "ID", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    muid_nom, muid_up, muid_down = pad_none_lep(muid_nom), pad_none_lep(muid_up), pad_none_lep(muid_down)
    muid_nom = ak.where(twoReco_mm_sel, muid_nom[:,0] * muid_nom[:,1], 1.0)
    muid_up  = ak.where(twoReco_mm_sel, muid_up[:,0]  * muid_up[:,1],  1.0)
    muid_down= ak.where(twoReco_mm_sel, muid_down[:,0]* muid_down[:,1],1.0)
    weights.add(name="muid", weight=muid_nom, weightUp=muid_up, weightDown=muid_down)

    muiso_nom, muiso_up, muiso_down = GetMuonSF(IOV, "ISO", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    muiso_nom, muiso_up, muiso_down = pad_none_lep(muiso_nom), pad_none_lep(muiso_up), pad_none_lep(muiso_down)
    muiso_nom = ak.where(twoReco_mm_sel, muiso_nom[:,0] * muiso_nom[:,1], 1.0)
    muiso_up  = ak.where(twoReco_mm_sel, muiso_up[:,0]  * muiso_up[:,1],  1.0)
    muiso_down= ak.where(twoReco_mm_sel, muiso_down[:,0]* muiso_down[:,1],1.0)
    weights.add(name="muiso", weight=muiso_nom, weightUp=muiso_up, weightDown=muiso_down)

    mutrig_nom, mutrig_up, mutrig_down = GetMuonSF(IOV, "HLT", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    mutrig_nom, mutrig_up, mutrig_down = pad_none_lep(mutrig_nom), pad_none_lep(mutrig_up), pad_none_lep(mutrig_down)
    mutrig_nom = ak.where(twoReco_mm_sel, mutrig_nom[:,0] * mutrig_nom[:,1], 1.0)
    mutrig_up  = ak.where(twoReco_mm_sel, mutrig_up[:,0]  * mutrig_up[:,1],  1.0)
    mutrig_down= ak.where(twoReco_mm_sel, mutrig_down[:,0]* mutrig_down[:,1],1.0)
    weights.add(name="mutrig", weight=mutrig_nom, weightUp=mutrig_up, weightDown=mutrig_down)

def HEMVeto(FatJets, runs, iov):
    ## from https://github.com/laurenhay/GluonJetMass/blob/main/python/corrections.py
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    
    runid = (runs >= 319077)
    print(runid)
    # print("Fat jet phi ", FatJets.phi)
    # print("Fat jet phi length ", len(FatJets.phi))
    # print("Fat jet eta ", FatJets.eta)
    # print("Fat jet eta length ", len(FatJets.eta))
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -1.3) & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -2.5) & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    vetoHEMFatJets = ak.any((detector_region1 & jet_selection & runid) ^ (detector_region2 & jet_selection & runid), axis=1)
    #print("Number of hem vetoed jets: ", ak.sum(vetoHEMFatJets))
    vetoHEM = ~(vetoHEMFatJets)
    
    return vetoHEM

def HEMVeto(FatJets, runs, isMC=False, year="2018"):
    """
    HEM veto for data and MC (flat weight approach for MC).
    For data: returns boolean mask (True = event passes veto).
    For MC:   returns per-event weight (1.0 or flat lumi-fraction weight).
    """

    # Luminosity fractions for 2018 (adjust to your golden JSON)
    L_total    = 59.74  # fb-1, total 2018
    L_preHEM   = 21.07  # fb-1, before run 319077 (2018A+B+early C)
    L_postHEM  = 38.67  # fb-1, run >= 319077 (late 2018C + 2018D)
    f_affected = L_postHEM / L_total  # ~0.647

    # Define dead detector regions
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                        (FatJets.eta < -1.3)  & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                        (FatJets.eta < -2.5)  & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    in_HEM_region = (detector_region1 | detector_region2) & jet_selection
    event_has_HEM_jet = ak.any(in_HEM_region, axis=1)  # True = bad jet present

    if not isMC:
        # ---- DATA: hard veto, only for affected runs ----
        runid = (runs >= 319077)
        vetoHEMFatJets = ak.any(in_HEM_region & runid, axis=1)
        return ~vetoHEMFatJets  # True = event passes

    else:
        # ---- MC (2018 only): flat luminosity-fraction weight ----
        # Events with a HEM jet: they would be vetoed in the affected lumi fraction
        #   → weight = (1 - f_affected) = L_preHEM / L_total
        # Events without a HEM jet: always pass
        #   → weight = 1.0
 
        hem_weight = ak.where(
            event_has_HEM_jet,
            (1.0 - f_affected),   # ~0.353 — only the "safe" lumi counts
            1.0                   # no HEM jet → unaffected
        )
        return hem_weight
