import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
import time
import os
import pandas as pd
import time
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
import gc
import tokenize as tok
import re
import logging
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .weight_class import Weights
from .hist_utils import *



class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen = True, mode = "minimal", debug = False):
        '''
        Args:
            do_gen (bool): whether to run gen-level analysis and create response matrices
            mode (str): "minimal" or "full". "minimal" runs a smaller set of histograms for testing purposes. 
                        "full" runs the full set of histograms for publication.
            testing (bool): whether to run in testing mode (faster, less data)
        '''
        self._do_gen = do_gen
        self._mode = mode
        self._debug = debug
        
        if self._debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        binning = util_binning()

        # Define axes
        ptreco_axis = binning.ptreco_axis
        mreco_axis = binning.mreco_axis
        ptgen_axis = binning.ptgen_axis     
        mgen_axis = binning.mgen_axis

        dataset_axis = binning.dataset_axis
        lep_axis = binning.lep_axis
        n_axis = binning.n_axis
        mass_axis = binning.mass_axis
        zmass_axis = binning.zmass_axis
        pt_axis = binning.pt_axis
        frac_axis = binning.frac_axis
        dr_axis = binning.dr_axis
        dr_fine_axis = binning.dr_fine_axis
        dphi_axis = binning.dphi_axis    
        syst_axis = binning.syst_axis
        eta_axis = binning.eta_axis
        phi_axis = binning.phi_axis
        ptfine_axis = binning.ptfine_axis
        
        ptgen_axis_fine  = binning.ptgen_axis_fine 

        
        mptreco_axis = binning.mreco_over_pt_axis
        mptgen_axis = binning.mgen_over_pt_axis

        

        ht_axis = hist.axis.StrCategory([],growth = True, name = "ht_bin", label = "h_T bin")
        channel_axis = hist.axis.StrCategory([],growth = True, name = "channel", label = "Channel")
        
        weight_axis = hist.axis.Regular(100, 0, 10, name="corrWeight", label=r"Weight")
        met_pt_axis = hist.axis.Regular(20, 0, 200, name = "pt", label = r"$p_T$")

        mcut_reco_u_axis = binning.mcut_reco_u_axis
        mcut_reco_g_axis = binning.mcut_reco_g_axis

        mcut_gen_u_axis = binning.mcut_gen_u_axis
        mcut_gen_g_axis = binning.mcut_gen_g_axis
        #### weight to check what is causing this
        
        #self.gen_binning = binning.gen_binning
        #self.reco_binning = binning.reco_binning

        self.hists = processor.dict_accumulator()

        if self._mode == "minimal":
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_u", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "response_matrix_g", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])

        
        self.hists["sumw"] = processor.defaultdict_accumulator(float)
        self.hists["nev"] = processor.defaultdict_accumulator(int)
        self.hists["cutflow"] = processor.defaultdict_accumulator(int)

    @property
    def accumulator(self):
        return self.hists

    def process(self, events_all):
        t0 = time.time()
        dataset = events_all.metadata["dataset"]
        filename = events_all.metadata['filename']
        logging.info(f"Starting processing for dataset: {dataset} and file: {filename}")

        logging.debug(f"Total events in chunk: {len(events_all)}")

        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'UL2016APV', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')

        self.hists["cutflow"][f"{dataset}_all"] += len(events_all)
        self.hists["nev"][dataset] += len(events_all)
        self.hists["sumw"][dataset] += ak.sum(events_all.genWeight)

        sel = PackedSelection()

        # if (self._do_gen):
        #     era = None
        #     firstidx = filename.find( "store/mc/" )
        #     fname2 = filename[firstidx:]
        #     fname_toks = fname2.split("/")
        #     year = fname_toks[ fname_toks.index("mc") + 1]
        #     ht_bin = fname_toks[ fname_toks.index("mc") + 2]

        #     ## Flag used for number of events
        #     herwig = False
        #     if 'herwig' in filename: herwig = True
        # else:
        #     firstidx = filename.find( "store/data/" )
        #     fname2 = filename[firstidx:]
        #     fname_toks = fname2.split("/")
        #     era = fname_toks[ fname_toks.index("data") + 1]
        #     channel = fname_toks[ fname_toks.index('NANOAOD') - 1]
        #     ht_bin = 'all'
        year = '2018'
        ht_bin = 'all'
        herwig = False
        channel = 'SingleMuon'
            # lumi_mask = self.lumimasks[IOV](events_all.run, events_all.luminosityBlock)
            # # print("RUN", events_all.run )
            # # print("Lumi block", events_all.luminosityBlock )
            # # print("lumi_mask", lumi_mask)
            # print("Len of evnets after mask", len(events_all[lumi_mask]) )
            # events_all = events_all[lumi_mask]

        # Trigger selection
        events1 = events_all
        if not self._do_gen:
            logging.info("Channel {}".format(channel))
            if "UL2016" in dataset: 
                if channel == "SingleMuon":
                    trigsel = events1.HLT.IsoMu24  
                else:
                    logging.info("Doing {channel} in {dataset}")
                    trigsel = events1.HLT.Ele27_WPTight_Gsf | events1.HLT.Photon175
            elif "UL2017" in dataset:
                if channel == "SingleMuon":
                    trigsel = events1.HLT.IsoMu27  
                else:
                    logging.info(f"Doing {channel} in {dataset}")
                    trigsel = events1.HLT.Ele35_WPTight_Gsf | events1.HLT.Photon200
            elif "UL2018" in dataset:
                if channel == "SingleMuon":
                    trigsel = events1.HLT.IsoMu24  
                else:
                    logging.info("Doing {channel} in {dataset}")
                    trigsel = events1.HLT.Ele32_WPTight_Gsf | events1.HLT.Photon200
            else:
                raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
            sel.add("trigsel", trigsel)    

            logging.debug("Trigger Selection ", ak.sum(sel.require(trigsel = True)))
        ptreco = ak.flatten(events1.FatJet.pt)
        logging.debug(f"ptreco before cleaning: {ptreco}")
        ptreco = ptreco[~ak.is_none(ptreco)]
        logging.debug(f"ptreco after cleaning: {ptreco}")
        mreco = ak.flatten(events1.FatJet.mass)
        mreco = mreco[~ak.is_none(mreco)]
        mreco_g = ak.flatten(events1.FatJet.msoftdrop)
        
        mreco_g = mreco_g[~ak.is_none(mreco_g)]
        fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = "nominal")
        fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco_g, systematic = "nominal")
        return self.hists

    def postprocess(self, accumulator):
        hname_list = ["ptjet_mjet_u_reco", 'ptjet_mjet_g_reco']
        sumw = accumulator["sumw"]

        for hname in hname_list:
            h = accumulator[hname]
            for ds in h.axes['dataset']:
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    continue
                else:
                    xs = 6077.22  # DY NLO
                    sw = sumw[ds]
                    self.lumi_fb = 59.74  # 2018 lumi
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue

                    scale = (xs * self.lumi_fb * 1000) / sw
                    for i, name in enumerate(h.axes["dataset"]):
                        h.view(flow=True)[i] *= scale
            accumulator[hname] = h
        return accumulator

                    
            