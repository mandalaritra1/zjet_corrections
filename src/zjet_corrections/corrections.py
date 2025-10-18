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


def GetPDFweights(df, var="nominal"):
    ## determines the pdf up and down variations
    pdf = ak.ones_like(df.Pileup.nTrueInt)
    if ("LHEPdfWeight" in ak.fields(df)):
        pdfUnc = ak.std(df.LHEPdfWeight,axis=1)/ak.mean(df.LHEPdfWeight,axis=1)
    else:
        pdfUnc = 0
    if var == "up":
        pdf = pdf + pdfUnc
    elif var == "down":
        pdf = pdf - pdfUnc
    return pdf

def GetL1PreFiringWeight(IOV, df):
    """Return (nominal, up, down) L1 prefiring weights from the event record."""
    if "L1PreFiringWeight" not in ak.fields(df):
        ones = ak.ones_like(df.event)
        return ones, ones, ones

    weights = df["L1PreFiringWeight"]
    return weights["Nom"], weights["Up"], weights["Dn"]
    
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


def GetEleSF(IOV, wp, eta, pt, var=""):
    """Return electron identification scale factors for the requested working point."""
    counts = ak.num(pt)
    evaluator = _load_ele_sf_corrections(IOV)

    mask = pt > 20
    adj_pt = ak.where(mask, pt, 22)

    sf_flat = evaluator["UL-Electron-ID-SF"].evaluate(
        _ELE_SF_YEAR_LABEL[IOV],
        "sf" + var,
        wp,
        np.array(ak.flatten(eta)),
        np.array(ak.flatten(adj_pt)),
    )

    sf = ak.unflatten(sf_flat, counts)
    return ak.where(mask, sf, ak.ones_like(sf))


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
        hname = "NUM_MediumID_DEN_TrackerMuons"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    elif corrset == "ISO":
        hname = "NUM_TightRelIso_DEN_MediumID"
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

def getLumiMaskRun2():
    golden_json_dir = files("zjet_corrections") / "corrections" / "goldenJsons"
    golden_json_path_2016 = golden_json_dir / "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
    golden_json_path_2017 = golden_json_dir / "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
    golden_json_path_2018 = golden_json_dir / "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"

    masks = {
        "2016APV": LumiMask(str(golden_json_path_2016)),
        "2016": LumiMask(str(golden_json_path_2016)),
        "2017": LumiMask(str(golden_json_path_2017)),
        "2018": LumiMask(str(golden_json_path_2018)),
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
    print("File exists check for JEC: ", str(data_file))
    print(data_file.is_file())

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
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
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
    name_map['Rho'] = 'rho'

    events_cache = events.caches[0]

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)


    corrected_jets = jet_factory.build(FatJets, lazy_cache=events_cache)
    # print("Available uncertainties: ", jet_factory.uncertainties())
    # print("Corrected jets object: ", corrected_jets.fields)
    #print("pt and mass before correction ", FatJets['pt_raw'], ", ", FatJets['mass_raw'], " and after correction ", corrected_jets["pt"], ", ", corrected_jets["mass"])
    return corrected_jets    
