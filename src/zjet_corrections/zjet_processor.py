import awkward as ak
import numpy as np
import time
import uproot
import hist
import vector
import time
import os
import pandas as pd
import time
from coffea import util, processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import gc
import tokenize as tok
import re
import copy
import os



from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .hist_utils import *
from .smp_utils import *


from .corrections import *


# get_herwig_weight_g = PtRhoWeighter('correctionFiles/spline_groomed.npz')
# get_herwig_weight_u = PtRhoWeighter('correctionFiles/spline_ungroomed.npz')

class Log:
    def __init__(self, mode="info"):
        self.mode = mode
    def info(self, msg):
        if self.mode in ["info", "debug"]:
            print("[INFO]", msg)
    def debug(self, msg):
        if self.mode == "debug":
            print("[DEBUG]", msg)
        

class QJetMassProcessor(processor.ProcessorABC):
    '''
    Processor to run a Z+jets jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, do_gen = True, mode = "minimal",  debug = False, jet_systematics = None, systematics = None):
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
        self._do_reweight = False
        self._do_jk = False

        if mode == "minimal":
            self._do_jk = False
        elif mode == "reweight_pythia" or mode == "reweight_pythia_rho":
            self._do_reweight = True
        elif mode == "jk_mc" or mode == "jk_data" or mode == "rho_jk" or mode == "mass_jk":
            self._do_jk = True
        elif mode == "full":
            self._do_jk = False
        else:
            self._do_jk = False
        
        
        if jet_systematics == None:
            self.jet_systematics = ['nominal', 'JERUp', 'JERDown', 'JMSUp', 'JMSDown', 'JMRUp', 'JMRDown']
            jes_systematics = ['JES_AbsoluteMPFBiasUp', 'JES_AbsoluteMPFBiasDown', 'JES_AbsoluteScaleUp', 'JES_AbsoluteScaleDown',
                'JES_AbsoluteStatUp', 'JES_AbsoluteStatDown', 'JES_FlavorQCDUp', 'JES_FlavorQCDDown', 'JES_FragmentationUp',
                'JES_FragmentationDown', 'JES_PileUpDataMCUp', 'JES_PileUpDataMCDown', 'JES_PileUpPtBBUp', 'JES_PileUpPtBBDown',
                'JES_PileUpPtEC1Up', 'JES_PileUpPtEC1Down', 'JES_PileUpPtEC2Up', 'JES_PileUpPtEC2Down', 'JES_PileUpPtHFUp', 'JES_PileUpPtHFDown', 
                'JES_PileUpPtRefUp', 'JES_PileUpPtRefDown', 'JES_RelativeFSRUp', 'JES_RelativeFSRDown', 'JES_RelativeJEREC1Up',
                'JES_RelativeJEREC1Down', 'JES_RelativeJEREC2Up', 'JES_RelativeJEREC2Down', 'JES_RelativeJERHFUp', 'JES_RelativeJERHFDown',
                'JES_RelativePtBBUp', 'JES_RelativePtBBDown', 'JES_RelativePtEC1Up', 'JES_RelativePtEC1Down', 'JES_RelativePtEC2Up', 'JES_RelativePtEC2Down',
                'JES_RelativePtHFUp', 'JES_RelativePtHFDown', 'JES_RelativeBalUp', 'JES_RelativeBalDown', 'JES_RelativeSampleUp', 'JES_RelativeSampleDown', 
                'JES_RelativeStatECUp', 'JES_RelativeStatECDown', 'JES_RelativeStatFSRUp', 'JES_RelativeStatFSRDown', 'JES_RelativeStatHFUp', 'JES_RelativeStatHFDown',
                'JES_SinglePionECALUp', 'JES_SinglePionECALDown', 'JES_SinglePionHCALUp', 'JES_SinglePionHCALDown', 'JES_TimePtEtaUp', 'JES_TimePtEtaDown']
            self.jet_systematics = self.jet_systematics + jes_systematics
        else:
            self.jet_systematics = jet_systematics
        
        if systematics == None:
            self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 
                                    'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp', 'murecoDown', 
                                    'muidUp', 'muidDown', 'mutrigUp', 'mutrigDown', 'muisoUp', 'muisoDown',
                                    'pdfUp', 'pdfDown', 'q2Up', 'q2Down', 'isrUp', 'isrDown', 'fsrUp', 'fsrDown',
                                    'l1prefiringUp', 'l1prefiringDown'] 
        else:
            self.systematics = systematics
        

        self.lepptcuts = [40,29] # [ele, mu]
        
        if self._debug:
            self.logging = Log(mode="debug")
        else:
            self.logging = Log(mode="info")

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
        mreco_over_pt_axis = binning.mreco_over_pt_axis
        mgen_over_pt_axis = binning.mgen_over_pt_axis
        y_axis = binning.y_axis
        ptlong_axis = binning.ptlong_axis
        
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

        if self._mode == "minimal" or self._mode == "reweight_pythia"  :
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis,channel_axis, ptreco_axis, mreco_axis, syst_axis])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_u", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "response_matrix_g", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis,channel_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis,channel_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptz_mz_reco" , [dataset_axis, zmass_axis, pt_axis])

        if self._mode == "minimal_rho" or self._mode == "reweight_pythia_rho":
            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            #register_hist(self.hists, "ptjet_rhojet_g_reco2", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_rho_u", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                register_hist(self.hists, "response_matrix_rho_g", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_u_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_g_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, syst_axis])
                

                #register_hist(self.hists, 'm_u_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
                #register_hist(self.hists, 'm_g_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis])
        if self._mode == "validation":
            register_hist(self.hists, "pt_mupos", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_mupos", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_mupos", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "pt_muneg", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_muneg", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_muneg", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "y_mupos", [dataset_axis, y_axis, syst_axis])
            register_hist(self.hists, "y_muneg", [dataset_axis, y_axis, syst_axis])


            
            register_hist(self.hists, "pt_elpos", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_elpos", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_elpos", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "pt_elneg", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_elneg", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_elneg", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "y_elpos", [dataset_axis, y_axis, syst_axis])
            register_hist(self.hists, "y_elneg", [dataset_axis, y_axis, syst_axis])
            
            register_hist(self.hists, "nJets", [dataset_axis, n_axis, syst_axis])
            
            register_hist(self.hists, "pt_Z", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "eta_Z", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_Z", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "mass_Z", [dataset_axis, zmass_axis, syst_axis])
            register_hist(self.hists, "y_Z", [dataset_axis, y_axis, syst_axis])
            
            
            register_hist(self.hists, "pt_jet0", [dataset_axis, pt_axis, syst_axis])
            register_hist(self.hists, "pt_flavor_jet0_gen", [dataset_axis, pt_axis, n_axis])
            register_hist(self.hists, "eta_jet0", [dataset_axis, eta_axis, syst_axis])
            register_hist(self.hists, "phi_jet0", [dataset_axis, phi_axis, syst_axis])
            register_hist(self.hists, "mass_jet0", [dataset_axis, mass_axis, syst_axis])
            register_hist(self.hists, "y_jet0", [dataset_axis, y_axis, syst_axis])
            register_hist(self.hists, "eta_phi_jet_reco", [dataset_axis, eta_axis, phi_axis])

            register_hist(self.hists, "ptasym_presel", [dataset_axis, frac_axis])
            register_hist(self.hists, "ptasym", [dataset_axis, frac_axis, syst_axis])
            register_hist(self.hists, "dr", [dataset_axis, dr_axis, syst_axis])
            register_hist(self.hists, "dphi", [dataset_axis, dphi_axis, syst_axis])

            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])
            
            #register_hist(self.hists, "ht", [dataset_axis, ptlong_axis, ht_axis ])
            #register_hist(self.hists, "eta_phi_jet_reco", [dataset_axis, eta_axis, phi_axis])
            
            # register_hist(self.hists, "ht_AK4", [dataset_axis, pt_axis, ht_axis ])
            # register_hist(self.hists, "ht_reco", [dataset_axis, pt_axis, ht_axis ])
            # register_hist(self.hists, "ht_reco_AK4", [dataset_axis, pt_axis, ht_axis ])
            # register_hist(self.hists, "pt_mu0", [dataset_axis, pt_axis, syst_axis])
            # register_hist(self.hists, "pt_Z", [dataset_axis, pt_axis, syst_axis])
            # register_hist(self.hists, "pt_el0", [dataset_axis, pt_axis, syst_axis])
  
            
            

            

                
        if self._mode == "rho_jk":
            register_hist(self.hists, "ptjet_rhojet_u_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis,  binning.jackknife_axis , syst_axis,])
            register_hist(self.hists, "ptjet_rhojet_g_reco", [dataset_axis, ptreco_axis, mreco_over_pt_axis,  binning.jackknife_axis , syst_axis, ])
            #register_hist(self.hists, "ptjet_rhojet_g_reco2", [dataset_axis, ptreco_axis, mreco_over_pt_axis, syst_axis ])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_rho_u", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis,])
                register_hist(self.hists, "response_matrix_rho_g", [dataset_axis, ptreco_axis, mreco_over_pt_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis,])
                register_hist(self.hists, "ptjet_rhojet_u_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_rhojet_g_gen", [dataset_axis, ptgen_axis, mgen_over_pt_axis, binning.jackknife_axis, syst_axis])
                

                register_hist(self.hists, 'm_u_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis, binning.jackknife_axis, syst_axis,])
                register_hist(self.hists, 'm_g_jet_reco_over_gen', [dataset_axis, ptgen_axis, mgen_axis, frac_axis, binning.jackknife_axis, syst_axis,])

        if self._mode == "mass_jk":
            register_hist(self.hists, "ptjet_mjet_u_reco", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, binning.jackknife_axis, syst_axis])
            register_hist(self.hists, "ptjet_mjet_g_reco", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, binning.jackknife_axis, syst_axis])

            if self._do_gen:
                register_hist(self.hists, "response_matrix_u", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "response_matrix_g", [dataset_axis, channel_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis, channel_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis, channel_axis, ptgen_axis, mgen_axis, binning.jackknife_axis, syst_axis])


                
        if self._mode == "jk_mc":
            register_hist(self.hists, "jk_response_matrix_u", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_response_matrix_g", [dataset_axis, ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis])
        if self._mode == "jk_data":
            register_hist(self.hists, "jk_ptjet_mjet_g_reco", [dataset_axis,ptreco_axis, mreco_axis, binning.jackknife_axis])
            register_hist(self.hists, "jk_ptjet_mjet_u_reco", [dataset_axis,ptreco_axis, mreco_axis, binning.jackknife_axis])

        
        self.hists["sumw"] = processor.defaultdict_accumulator(float)
        self.hists["nev"] = processor.defaultdict_accumulator(int)
        self.hists["cutflow"] = processor.defaultdict_accumulator(int)
        self.logging.debug(f"Registered Histograms {self.hists.keys()}")

    @property
    def accumulator(self):
        return self.hists

    def process(self, events_all):
        self.logging.debug(f"Systematics {self.systematics}")
        self.logging.debug(f"Current Mode {self._mode}")
        t0 = time.time()
        dataset = events_all.metadata["dataset"]
        filename = events_all.metadata['filename']
        self.logging.info(f"Starting processing for dataset: {dataset} and file: {filename}")

       

        self.logging.debug(f"Total events in chunk: {len(events_all)}")

        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'UL2016APV', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        self.IOV = IOV

        self.hists["cutflow"][f"{dataset}_all"] += len(events_all)
        self.hists["nev"][dataset] += len(events_all)
        if self._do_gen:
            self.hists["sumw"][dataset] += ak.sum(events_all.genWeight)
        else:
            self.hists["sumw"][dataset] += len(events_all)

        index_list = np.arange(len(events_all))

        #print(f"Binning : {events_all.Generator.binvar}")
        for jk_index in range(0, 10): ## loops from 0 to 9 in case do_jk flag is enabled, otherwise breaks at 0
            if not self._do_jk:
                events1 = events_all
                self.logging.debug("Jackknife resampling not enabled, processing all events together.")
            if self._do_jk:
                jk_sel = ak.where( (index_list % 10) == jk_index, False, True)
                #self.logging.debug("JK index ", jk_index, " events dropped ", ak.sum(~jk_sel) )
                events1 = events_all[jk_sel]
                del jk_sel

                
 
            sel = PackedSelection(dtype='uint64')
            
            if (self._do_gen):
                era = None
                firstidx = filename.find( "store/mc/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                year = fname_toks[ fname_toks.index("mc") + 1]
                ht_bin = fname_toks[ fname_toks.index("mc") + 2]
                ht_bin_tokens = ht_bin.split("_")
                try:
                    ht_bin = ht_bin_tokens[ht_bin_tokens.index("DYJetsToLL")+2]
                except:
                    ht_bin = 'inclusive'
                

                ## Flag used for number of events
                herwig = False
                if 'herwig' in filename: herwig = True
                self.herwig = herwig
            if not self._do_gen:
                firstidx = filename.find( "store/data/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                channel = fname_toks[ fname_toks.index('NANOAOD') - 1]
                ht_bin = 'all'
                self.lumimasks = getLumiMaskRun2()
                lumi_mask = self.lumimasks[IOV](events1.run, events1.luminosityBlock)

                events1 = events1[lumi_mask]
            if self._do_gen:
                self.logging.info(f"year: {year}, ht_bin: {ht_bin}, herwig: {herwig}")
            else:
                self.logging.info(f"channel: {channel}")
            # year = '2018'


            # ht_bin = 'all'
            # herwig = False
            #channel = 'SingleMuon'
                # lumi_mask = self.lumimasks[IOV](events_all.run, events_all.luminosityBlock)
                # # print("RUN", events_all.run )
                # # print("Lumi block", events_all.luminosityBlock )
                # # print("lumi_mask", lumi_mask)
                # print("Len of evnets after mask", len(events_all[lumi_mask]) )
                # events_all = events_all[lumi_mask]

            # Trigger selection
        
            if not self._do_gen:
                self.logging.info("Channel {}".format(channel))
                if "UL2016" in dataset: 
                    if channel == "SingleMuon":
                        trigsel = events1.HLT.IsoMu24  
                    else:
                        self.logging.info("Doing {channel} in {dataset}")
                        trigsel = events1.HLT.Ele27_WPTight_Gsf | events1.HLT.Photon175
                elif "UL2017" in dataset:
                    if channel == "SingleMuon":
                        trigsel = events1.HLT.IsoMu27  
                    else:
                        self.logging.info(f"Doing {channel} in {dataset}")
                        trigsel = events1.HLT.Ele35_WPTight_Gsf | events1.HLT.Photon200
                elif "UL2018" in dataset:
                    if channel == "SingleMuon":
                        trigsel = events1.HLT.IsoMu24  
                    else:
                        self.logging.info("Doing {channel} in {dataset}")
                        trigsel = events1.HLT.Ele32_WPTight_Gsf | events1.HLT.Photon200
                else:
                    raise Exception("Dataset is incorrect, should have 2016, 2017, 2018: ", dataset)
                sel.add("trigsel", trigsel)    

                self.logging.debug(f"Trigger Selection {ak.sum(sel.require(trigsel = True))}")


            events0 = events1 # fix because hassle to change all events1 to events0 below

            weights = Weights(size = len(events0), storeIndividual = True) #initialize weights class
            self.logging.debug("Weights initialized")
            if len(events0) <1:
                return self.hists
            # Store GEN weights or ones based on Simulation/Data
            if self._do_gen:
                
                
                weights.add("genWeight", events0.genWeight)
                if "LHE" in events0.fields:
                    ht_value = events0.LHE.HT
                    print("ht bin", ht_bin)
                    fill_hist(self.hists, 'ht', dataset = dataset, pt = ht_value, ht_bin = ht_bin, weight = weights.weight())

                    # ht_value_ak4 = ak.sum(events0.GenJet[events0.GenJet.pt > 10].pt, axis = 1)
                    
                    # fill_hist(self.hists, 'ht_AK4', dataset = dataset, pt = ht_value_ak4, ht_bin = ht_bin, weight = weights.weight())

                    del ht_value, #ht_value_ak4
            else:
                weights = Weights(size = len(events0), storeIndividual = True)
                weights.add("unity", np.ones(len(events0)))

            


            
            #####################################
            ### MET Filters ####################
            #####################################

            self.logging.info("Applying MET filters")

            MET_filters = {'2016APV':["goodVertices",
                                "globalSuperTightHalo2016Filter",
                                "HBHENoiseFilter",
                                "HBHENoiseIsoFilter",
                                "EcalDeadCellTriggerPrimitiveFilter",
                                "BadPFMuonFilter",
                                "BadPFMuonDzFilter",
                                "eeBadScFilter",
                                "hfNoisyHitsFilter"],
                    '2016'   :["goodVertices",
                                "globalSuperTightHalo2016Filter",
                                "HBHENoiseFilter",
                                "HBHENoiseIsoFilter",
                                "EcalDeadCellTriggerPrimitiveFilter",
                                "BadPFMuonFilter",
                                "BadPFMuonDzFilter",
                                "eeBadScFilter",
                                "hfNoisyHitsFilter"],
                    '2017'   :["goodVertices",
                                "globalSuperTightHalo2016Filter",
                                "HBHENoiseFilter",
                                "HBHENoiseIsoFilter",
                                "EcalDeadCellTriggerPrimitiveFilter",
                                "BadPFMuonFilter",
                                "BadPFMuonDzFilter",
                                "hfNoisyHitsFilter",
                                "eeBadScFilter",
                                "ecalBadCalibFilter"],
                    '2018'   :["goodVertices",
                                "globalSuperTightHalo2016Filter",
                                "HBHENoiseFilter",
                                "HBHENoiseIsoFilter",
                                "EcalDeadCellTriggerPrimitiveFilter",
                                "BadPFMuonFilter",
                                "BadPFMuonDzFilter",
                                "hfNoisyHitsFilter",
                                "eeBadScFilter",
                                "ecalBadCalibFilter"]}
            

            if self._do_gen:
                pu_nom, pu_up, pu_down = get_pu_weights(events0, IOV)
                self.logging.debug(f"PU weights (nom, up, down) : {pu_nom[:10]}")
                
                weights.add(name = "pu", weight = pu_nom, weightUp = pu_up, weightDown = pu_down)
                pdf_nom, pdf_up, pdf_down = GetPDFweights(events0)
                self.logging.debug(f"pdf weights (nom, up, down) : {pdf_nom[:10]}")
                weights.add(name = "pdf", weight = pdf_nom, weightUp = pdf_up, weightDown = pdf_down)
                l1prefire_nom, l1prefire_up, l1prefire_down = GetL1PreFiringWeight(IOV, events0)
                self.logging.debug(f"L1 prefiring weights (nom, up, down) : {l1prefire_nom[:10]}")
                weights.add(name = "l1prefiring", weight = l1prefire_nom, weightUp = l1prefire_up, weightDown = l1prefire_down)
                q2_nom, q2_up, q2_down = GetQ2weights(events0)
                self.logging.debug(f"Q2 weights (nom, up, down) : {q2_nom[:10]}")
                weights.add(name = "q2", weight = q2_nom, weightUp = q2_up, weightDown = q2_down)

                
                isr_nom, isr_up, isr_down = GetPSweights(events0, "ISR")
                fsr_nom, fsr_up, fsr_down = GetPSweights(events0, "FSR")
                try:
                    weights.add(name = "isr", weight = isr_nom, weightUp = isr_up, weightDown = isr_down)
                    weights.add(name = "fsr", weight = fsr_nom, weightUp = fsr_up, weightDown = fsr_down)
                except:
                    pass

                
            
            # GEN Selection
            if self._do_gen:
                self.logging.info("Entering GEN selection")

                # Apply selection on GEN particles
                # Apply different pt cuts for electrons and muons
                is_electron = np.abs(events0.GenDressedLepton.pdgId) == 11
                is_muon = np.abs(events0.GenDressedLepton.pdgId) == 13

                pt_cut = (
                    (is_electron & (events0.GenDressedLepton.pt > self.lepptcuts[0])) |
                    (is_muon & (events0.GenDressedLepton.pt > self.lepptcuts[1]))
                )
                
                eta_cut = np.abs(events0.GenDressedLepton.eta) < 2.4
                
                events0 = ak.with_field(
                    events0,
                    events0.GenDressedLepton[pt_cut & eta_cut],
                    "GenDressedLepton"
                )

                ## Adding rapidity
                events0 = ak.with_field(
                    events0,
                    ak.with_field(
                        events0.GenJetAK8,
                        getRapidity(events0.GenJetAK8),
                        "rapidity"
                    ),
                    "GenJetAK8"
                )
                
                events0 = ak.with_field(
                                    events0,
                                    events0.GenJetAK8[(events0.GenJetAK8.pt > 0)
                                                & (np.abs(events0.GenJetAK8.rapidity) < 2.4)
                                    ],
                                    "GenJetAK8"
                )
                
                genjets_clean = apply_lepton_separation_gen(
                                events0.GenJetAK8,
                                events0.GenDressedLepton,
                                dr_cut=0.4
                            )
                events0 = ak.with_field(
                            events0,
                            genjets_clean[
                                (genjets_clean.pt > 0)
                                & (np.abs(genjets_clean.rapidity) < 2.4)
                            ],
                            "GenJetAK8"
                        )


                sel.add("oneGenJet", 
                        ak.sum( (events0.GenJetAK8.pt > 0) & (np.abs(events0.GenJetAK8.rapidity) < 2.4), axis=1 ) >= 1
                    )
                sel.add("oneGenJet_pt200", 
                        ak.sum( (events0.GenJetAK8.pt > 200) & (np.abs(events0.GenJetAK8.rapidity) < 2.4), axis=1 ) >= 1
                    )


                z_gen = get_z_gen_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
                z_ptcut_gen = sel.all("twoGen_leptons") & ak.fill_none(z_gen.pt > 90.0, False)
                z_mcut_gen = sel.all("twoGen_leptons") & ak.fill_none(
                    (z_gen.mass > 71.0) & (z_gen.mass < 111.0),
                    False,
                )

                sel.add("z_ptcut_gen", z_ptcut_gen)
                sel.add("z_mcut_gen", z_mcut_gen)

                ######## dr between two leptons ########
                twoGen_ee_sel = sel.require(twoGen_ee = True)
                twoGen_mm_sel = sel.require(twoGen_mm = True)


                gen_jet, z_jet_dphi_gen = get_dphi( z_gen, events0.GenJetAK8 )
                z_jet_dr_gen = gen_jet.delta_r(z_gen)

                gensubjets = events0.SubGenJetAK8
                groomed_gen_jet, groomedgensel = get_groomed_jet(gen_jet, gensubjets, False)

                #####################################
                ### Gen event topology selection
                #####################################        
                z_pt_asym_gen = np.abs(z_gen.pt - gen_jet.pt) / (z_gen.pt + gen_jet.pt)
                z_pt_frac_gen = gen_jet.pt / z_gen.pt
                z_pt_asym_sel_gen =  z_pt_asym_gen < 0.3
                z_jet_dphi_sel_gen = z_jet_dphi_gen > 1.57 #2.8 #np.pi * 0.5

                sel.add("z_jet_dphi_sel_gen", z_jet_dphi_sel_gen)
                sel.add("z_pt_asym_sel_gen", z_pt_asym_sel_gen)

                # sel.add("z_jet_dphi_sel_gen_seq", sel.all("z_mcut_gen_seq", "z_jet_dphi_sel_gen") )
                # sel.add( "z_pt_asym_sel_gen_seq", sel.all("z_jet_dphi_sel_gen_seq", "z_pt_asym_sel_gen") )

                kinsel_gen = sel.require(twoGen_leptons=True,oneGenJet=True,z_ptcut_gen=True,z_mcut_gen=True)
                sel.add("kinsel_gen", kinsel_gen)
                toposel_gen = sel.require( z_pt_asym_sel_gen=True, z_jet_dphi_sel_gen=True)
                sel.add("toposel_gen", toposel_gen)

            

            # ----------------- RECO Selection ---------------------------
            self.logging.info("Entering RECO selection")
            
                
            # npv cut
            sel.add('npv', events0.PV.npvsGood >0)

            selectEvents = np.array([events0.Flag[MET_filters[IOV][i]] for i in range(len(MET_filters[IOV]))
                            if MET_filters[IOV][i] in events0.Flag.fields])
            selectEvents = np.logical_and.reduce(selectEvents, axis=0) ## a passing event should pass "ALL" the MET filters

            self.logging.debug("MET Filter applied")

            if ak.sum(selectEvents) < 1 :
                print("No event passed the MET filters.")
                return self.hists

            sel.add("MET", selectEvents)
            sel.add("MET_seq", sel.all('npv', 'MET')) # This is RECO only

            eta = np.abs(events0.Electron.eta)
            
            events0 = ak.with_field(
                events0,
                events0.Electron[
                    (events0.Electron.pt > self.lepptcuts[0])
                    & (eta < 2.4)
                    & ((eta < 1.422) | (eta > 1.566))      # exclude ECAL crack
                    & (events0.Electron.pfRelIso03_all < 0.25)  # suppressing isolation cut here
                    & (events0.Electron.cutBased > 3)      # tight: 4, medium: 3
                    & (np.abs(events0.Electron.dz) < 0.5)
                    & (np.abs(events0.Electron.dxy) < 0.2)
                ],
                "Electron",
            )


                
            events0 = ak.with_field(
                    events0,
                    events0.Muon[(events0.Muon.pt > self.lepptcuts[1]) 
                                &(np.abs(events0.Muon.eta) < 2.4)
                                &(events0.Muon.pfIsoId > 3) #tight iso, pfIso04 < 0.2 , 2 = loose, 3 = medium # supressing isolation cut here
                                #&(events0.Muon.miniIsoId > 1)
                                &(events0.Muon.tightId	 == True)
                                & (np.abs(events0.Muon.dz) < 0.5) ## dz < 0.2 cm to prevent pileup
                                & (np.abs(events0.Muon.dxy) < 0.2)
                                #&(events0.Muon.looseId	 == True)
                    
                    ],
                    "Muon"
                )
            self.logging.debug("Leptons Selected")

            
            z_reco = get_z_reco_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)



            # Guard: only evaluate Z kinematics where a valid dilepton pair exists
            has_z = sel.require(twoReco_leptons=True)
            z_ptcut_reco  = has_z & (z_reco.pt > 90)
            z_mcut_reco   = has_z & (z_reco.mass > 71.) & (z_reco.mass < 111.)
            
            sel.add("z_ptcut_reco", z_ptcut_reco)
            sel.add("z_mcut_reco",  z_mcut_reco)
            # z_ptcut_reco = z_reco.pt > 90
            # z_mcut_reco = (z_reco.mass > 71.) & (z_reco.mass < 111.)
            # sel.add("z_ptcut_reco", z_ptcut_reco & (sel.require(twoReco_leptons = True) ))
            # sel.add("z_mcut_reco", z_mcut_reco & (sel.require(twoReco_leptons = True) ))

            self.logging.debug("Z Object Created")
            #### dr reco plots ###
            sel_nominal = copy.deepcopy(sel)
            
            twoReco_ee_sel = sel.require(twoReco_ee = True)
            twoReco_mm_sel = sel.require(twoReco_mm = True)
            twoReco_ll_sel = sel.require(twoReco_leptons = True)

            ## Storing jec variation in the main 'Events' class
            
            corr_jets = GetJetCorrections(events0.FatJet, events0, era, IOV, isData = not self._do_gen, mode='AK8')  ###### correcting FatJet.mass
            corr_jets = corr_jets[corr_jets.subJetIdx1 > -1]

            ## Storing the jec variations for the AK4 subjets
            corr_subjets = GetJetCorrections(events0.SubJet, events0, era, IOV, isData = not self._do_gen, mode = 'AK4')
            #corr_jets['msoftdrop_orig'] = corr_jets.msoftdrop
            corr_jets['msoftdrop'] =   (corr_subjets[corr_jets.subJetIdx1] + corr_subjets[corr_jets.subJetIdx2]).mass 

            self.logging.debug("Jet Corrections Applied")
            self.logging.debug(f"Available Jet systematics {self.jet_systematics}")
            
            
            
            for jet_syst in self.jet_systematics: # Start loop over jet systematics
                self.logging.debug(f"Processing jet systematic: {jet_syst}")
                        # Apply jet corrections and show leading changes for quick inspection
                if jet_syst != 'nominal':
                    del sel
                sel = copy.deepcopy(sel_nominal)

                # corr_fatjets = GetJetCorrections(
                #     events0.FatJet,
                #     events0,
                #     era=dataset,
                #     IOV=IOV,
                #     isData=not self._do_gen,
                #     mode='AK8',
                # )
                
                # fatjet_pt_after = ak.to_numpy(ak.flatten(corr_fatjets.pt, axis=None))
                self.logging.debug(f"FatJet pt before correction: {events0[ak.num(events0.FatJet, axis = 1) > 0].FatJet.pt}" )
                # self.logging.debug(f"FatJet pt after correction: {fatjet_pt_after[:5]}" )
                

                if jet_syst == "nominal":
                    if self._do_gen:
                        events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV, corr_jets)), "FatJet")
                    else:
                        events_j = ak.with_field(events0, corr_jets , "FatJet")
                    self.logging.debug(f"FatJet pt after correction: {events_j[ak.num(events_j.FatJet, axis = 1) > 0].FatJet.pt}" )
                elif jet_syst == "JERUp":
                    corr_jets_obj = corr_jets.JER.up
                    corr_jets_obj['msoftdrop'] = (corr_subjets.JER.up[corr_jets.subJetIdx1] + corr_subjets.JER.up[corr_jets.subJetIdx2]).mass
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                elif jet_syst == "JERDown":
                    corr_jets_obj = corr_jets.JER.up
                    corr_jets_obj['msoftdrop'] = (corr_subjets.JER.down[corr_jets.subJetIdx1] + corr_subjets.JER.down[corr_jets.subJetIdx2]).mass
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                elif jet_syst == "JMSUp":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "up")) , "FatJet")
                elif jet_syst == "JMSDown":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets, var = "down")) , "FatJet")
                elif jet_syst == "JMRUp":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "up") , "FatJet")
                elif jet_syst == "JMRDown":
                    events_j = ak.with_field(events0,  jmrsf(IOV, jmssf(IOV,corr_jets), var = "down") , "FatJet")
                
                elif (jet_syst[-2:]=="Up" and "JES" in jet_syst):
                    #print(jet_syst)
                    field = jet_syst[:-2]
                    #print(field)
                    corr_jets_obj = corr_jets[field].up
                    corr_jets_obj['msoftdrop'] = (corr_subjets[field].up[corr_jets.subJetIdx1] + corr_subjets[field].up[corr_jets.subJetIdx2]).mass
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                    
                elif (jet_syst[-4:]=="Down" and "JES" in jet_syst):
                    field = jet_syst[:-4]
                    corr_jets_obj = corr_jets[field].down
                    corr_jets_obj['msoftdrop'] = (corr_subjets[field].down[corr_jets.subJetIdx1] + corr_subjets[field].down[corr_jets.subJetIdx2]).mass
                    
                    events_j = ak.with_field(events0, jmrsf(IOV, jmssf(IOV,corr_jets_obj)), "FatJet")
                    del corr_jets_obj
                
                else:
                    print("{} is not considered".format(jet_syst))

                ####### ADDIND RAPIDITY ############


                events_j = ak.with_field(
                    events_j,
                    ak.with_field(
                        events_j.FatJet,
                        getRapidity(events_j.FatJet),
                        "rapidity"
                    ),
                    "FatJet"
                )

                
                events_j = ak.with_field(
                    events_j,
                    ak.with_field(
                        events_j.Muon,
                        getRapidity(events_j.Muon),
                        "rapidity"
                    ),
                    "Muon"
                )
                
                events_j = ak.with_field(
                    events_j,
                    ak.with_field(
                        events_j.Electron,
                        getRapidity(events_j.Electron),
                        "rapidity"
                    ),
                    "Electron"
                )
                #events_j.FatJet['y'] = compute_rapidity(events_j.FatJet)


                # events_j = ak.with_field(
                #             events_j,
                #             ak.with_field(events_j.FatJet, compute_rapidity(events_j.FatJet), "y"),
                #             "FatJet"
                #         )
                # events_j = ak.with_field(
                #             events_j,
                #             ak.with_field(events_j.Electron, compute_rapidity(events_j.Electron), "y"),
                #             "Electron"
                #         )
                
                
                presort_recojets = events_j.FatJet[
                    (events_j.FatJet.mass > 0) 
                    &(events_j.FatJet.pt > 0) 
                    &(np.abs(events_j.FatJet.rapidity) < 2.4) 
                    &(events_j.FatJet.jetId == 6) 
                ]



                ## Sorting jets by pt
                index = ak.argsort(presort_recojets.pt, ascending=False)
                
                recojets  = presort_recojets[index]
                #recojets = presort_recojets
                events_j = ak.with_field(events_j, recojets, "FatJet")

                ## adding lepton separation
                recojets = apply_lepton_separation(
                                events_j.FatJet,
                                events_j.Muon,
                                events_j.Electron,
                                dr_cut=0.4,
                            )
                #recojets = events_j.FatJet
                # if self._do_gen:
                #     hem_weight = HEMVeto(FatJets, runs, isMC=self._do_gen, year = IOV)
                # hem_sel = HEMVeto(recojets, events_j.run) #is not in HEM 15/16 region
                
                if self._do_gen and IOV == "2018":
                    hem_weight = HEMVeto(events_j.FatJet, events_j.run, isMC=True)
                    weights.add("HEM", hem_weight)
                    hem_sel = ak.ones_like(events_j.event)
                else:
                    hem_sel = HEMVeto(recojets, events_j.run) #is not in HEM 15/16 region
                
                # sel.add("oneRecoJet", 
                #             ak.sum( (events_j.FatJet.pt > 0) & (np.abs(events_j.FatJet.eta) < 2.5)  & (events_j.FatJet.jetId == 6) & hem_sel, axis=1 ) >= 1
                #         )

                sel.add("oneRecoJet",
                    ak.sum((recojets.pt > 0) & (np.abs(recojets.eta) < 2.5) & (recojets.jetId == 6) & hem_sel, axis=1) >= 1 )


                reco_jet, z_jet_dphi_reco = get_dphi( z_reco, recojets )
                
            
                #reco_jet = ak.firsts(recojets)


                ## Muon correction
                rocc = get_rocc_corrections()
                rng = np.random.default_rng(seed=12345)
                
                if self._do_gen:
                    k = rocc.ak_kSmearMC(events_j.Muon, events_j.Muon.nTrackerLayers, rng=rng)
                else:
                    
                    k = rocc.ak_kScaleDT(events_j.Muon)
                
                mu_corr = ak.with_field(events_j.Muon, events_j.Muon.pt * k, "pt")
                events_j = ak.with_field(events_j, mu_corr, "Muon")

                ### Sorting by pt
                mu = events_j.Muon

                index = ak.argsort(mu.pt, axis=1, ascending=False)
                mu_sorted = mu[index]
                
                # Replace the Muon collection in events_j
                events_j = ak.with_field(events_j, mu_sorted, "Muon")


            
                #print("Special event reco_jet object pt", reco_jet[sel_spl].pt)
                #print("Special event FatJets object pt, eta, jetid", events0[sel_spl].FatJet.pt, events0[sel_spl].FatJet.eta, events0[sel_spl].FatJet.jetId)

                #print("Special event FatJets object pt, eta, jetid", recojets[sel_spl].pt, recojets[sel_spl].eta, recojets[sel_spl].jetId)
                #reco_jet, dr = find_closest_dr( gen_jet, recojets )
                z_jet_dr_reco = reco_jet.delta_r(z_reco)
                z_jet_dphi_reco_values = z_jet_dphi_reco

                ####### MAKE PRESEL PLOTS ######
                #filter_sel = sel.all('npv', 'oneRecoJet')
                reco_exists = ~ak.is_none(reco_jet.mass)

                
                #####################################
                ### Reco event topology sel
                #####################################
                z_jet_dphi_sel_reco = (z_jet_dphi_reco > 1.57)  #& (sel.require(twoReco_leptons = True))#np.pi * 0.5
                z_pt_asym_reco = np.abs(z_reco.pt - reco_jet.pt) / (z_reco.pt + reco_jet.pt)
                z_pt_frac_reco = reco_jet.pt / z_reco.pt
                z_pt_asym_sel_reco = (z_pt_asym_reco < 0.3) #& (sel.require(twoReco_leptons = True))
                sel.add("z_jet_dphi_sel_reco", z_jet_dphi_sel_reco)
                sel.add("z_pt_asym_sel_reco", z_pt_asym_sel_reco)


                
                kinsel_reco = sel.require(twoReco_leptons=True,oneRecoJet=True,z_ptcut_reco=True,z_mcut_reco=True)#, jetid = True)
                sel.add("kinsel_reco", kinsel_reco)
                #print("Leading RECO Jet matched muon ID", reco_jet.muonIdx3SJ)
                #print("Z-Jet dphi cut ",  sel.require(kinsel_reco= True, z_jet_dphi_sel_reco= True,trigsel= True ).sum())
                #print("Z-Jet pt-asymmetry cut ",  sel.require(kinsel_reco= True,z_jet_dphi_sel_reco= True, z_pt_asym_sel_reco= True,trigsel= True ).sum())

                
                toposel_reco = sel.require( z_pt_asym_sel_reco=True, z_jet_dphi_sel_reco=True)
                sel.add("toposel_reco", toposel_reco)
                if jet_syst == "nominal":
                    ## N-1 for pt asym cut
                    if self._do_gen:
                        sel_n1_ptasym = sel.all("npv" , "MET" , "kinsel_reco", "z_jet_dphi_sel_reco") 
                    else:
                        sel_n1_ptasym = sel.all("npv" , "MET" , "kinsel_reco","trigsel", "z_jet_dphi_sel_reco" ) 
    
                    if ak.sum(sel_n1_ptasym)>0:
                        self.logging.debug(f"Sum of this sel is {ak.sum(sel_n1_ptasym)}")
                        self.logging.debug(f"Len? {len(z_pt_asym_reco[sel_n1_ptasym])}  ")
                        
                        #print(z_pt_asym_reco[sel_n1_ptasym])
                        fill_hist(self.hists, "ptasym_presel", dataset = dataset, frac = z_pt_asym_reco[sel_n1_ptasym], weight = weights.weight()[sel_n1_ptasym])
                
                
                if self._do_gen:
                    import matplotlib.pyplot as plt
                    #plt.hist(reco_jet.delta_r(gen_jet), bins = 100, range = (0,1))
                    #plt.show()
                    is_matched_reco = reco_jet.delta_r(gen_jet) < 0.4
                    sel.add("is_matched_reco", is_matched_reco)

                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco")#, "is_matched_reco" )
                    #plt.hist(reco_jet[allsel_reco].delta_r(gen_jet[allsel_reco]), bins = 1000, range = (0,4))
                    #plt.show()
                    sel.add("allsel_reco", allsel_reco)
                    is_matched_gen = gen_jet.delta_r(reco_jet) < 0.4
                    sel.add("is_matched_gen", is_matched_gen)
                    allsel_gen = sel.all("kinsel_gen", "toposel_gen" , "is_matched_gen" )
                    sel.add("allsel_gen", allsel_gen)
                    #sel.add("fakes", sel.require(allsel_reco = True, allsel_gen = False))
                else:
                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco", "trigsel" )
                    sel.add("allsel_reco", allsel_reco)

            




                if (self._do_gen) and (jet_syst == "nominal"):

                    self.logging.debug("Padded Electron/Muon collections to minimum length 2 per event")
                    add_lepton_weights(events_j, twoReco_ee_sel, twoReco_mm_sel, weights, IOV)  

                    self.logging.info("Lepton weights added")      
                sel_reco = sel.require(allsel_reco = True)
                if jet_syst == "nominal":
                    fill_hist(self.hists, "ptz_mz_reco", dataset =  dataset, mass = z_reco[sel_reco].mass, pt = z_reco[sel_reco].pt, weight = weights.weight()[sel_reco]) 

                #####-###-----------PLOTTING EVENTS ----___##


                
                # ------------------------------------------------------------
                # Small helpers
                # ------------------------------------------------------------
                # def _to_1d_numpy(x):
                #     if x is None:
                #         return np.array([])
                #     return ak.to_numpy(ak.flatten(ak.fill_none(x, []), axis=None))
                
                # def _wrap_phi(phi):
                #     """Wrap phi into [0, 2pi) for polar plotting."""
                #     phi = np.asarray(phi)
                #     return np.mod(phi, 2 * np.pi)
                
                # def _event_collection(ev, name):
                #     return getattr(ev, name) if name in ev.fields else None
                
                # def _plot_collection_polar(
                #     ax,
                #     coll,
                #     label,
                #     color=None,
                #     marker="o",
                #     open_marker=False,
                #     annotate_pt=False,
                #     max_r=None,
                #     lw=1.5,
                #     alpha=0.95,
                # ):
                #     """
                #     Plot a single-event collection on polar plane:
                #       theta = phi
                #       r     = pt
                #     """
                #     if coll is None or len(coll) == 0:
                #         return
                
                #     pt  = _to_1d_numpy(coll.pt)
                #     phi = _wrap_phi(_to_1d_numpy(coll.phi))
                
                #     if len(pt) == 0:
                #         return
                
                #     # scatter
                #     if open_marker:
                #         ax.scatter(phi, pt, marker=marker, s=65, facecolors="none",
                #                    edgecolors=color, linewidths=lw, alpha=alpha, label=label)
                #     else:
                #         ax.scatter(phi, pt, marker=marker, s=45, color=color,
                #                    linewidths=lw, alpha=alpha, label=label)
                
                #     # radial lines from origin
                #     for p, ph in zip(pt, phi):
                #         ax.plot([ph, ph], [0, p], color=color, alpha=0.35, lw=1.2)
                
                #     if annotate_pt:
                #         for p, ph in zip(pt, phi):
                #             ax.text(ph, p + 0.02 * (max_r if max_r is not None else np.max(pt) + 1),
                #                     f"{p:.0f}", fontsize=7, ha="center", va="bottom")
                
                # def _plot_single_object_polar(
                #     ax,
                #     obj,
                #     label,
                #     color=None,
                #     marker="D",
                #     open_marker=False,
                #     lw=2.0,
                #     alpha=1.0,
                # ):
                #     """
                #     Plot a single object (not a collection) that has pt, phi fields.
                #     Works for ak.Record or one-element arrays.
                #     """
                #     if obj is None:
                #         return
                
                #     try:
                #         pt = ak.to_numpy(ak.atleast_1d(obj.pt))
                #         phi = _wrap_phi(ak.to_numpy(ak.atleast_1d(obj.phi)))
                #     except Exception:
                #         return
                
                #     if len(pt) == 0:
                #         return
                
                #     p = float(pt[0])
                #     ph = float(phi[0])
                
                #     ax.plot([ph, ph], [0, p], color=color, alpha=0.5, lw=1.5)
                
                #     if open_marker:
                #         ax.scatter([ph], [p], marker=marker, s=70, facecolors="none",
                #                    edgecolors=color, linewidths=lw, alpha=alpha, label=label)
                #     else:
                #         ax.scatter([ph], [p], marker=marker, s=60, color=color,
                #                    linewidths=lw, alpha=alpha, label=label)
                
                # def _compute_event_z_from_leptons(coll):
                #     """
                #     Approximate Z direction from first two leptons in a collection using vector sum in transverse plane.
                #     Returns (pt, phi) or None.
                #     """
                #     if coll is None or len(coll) < 2:
                #         return None
                
                #     try:
                #         pt  = ak.to_numpy(coll.pt)
                #         phi = ak.to_numpy(coll.phi)
                #     except Exception:
                #         return None
                
                #     if len(pt) < 2:
                #         return None
                
                #     px = pt[0] * np.cos(phi[0]) + pt[1] * np.cos(phi[1])
                #     py = pt[0] * np.sin(phi[0]) + pt[1] * np.sin(phi[1])
                
                #     z_pt = np.hypot(px, py)
                #     z_phi = np.arctan2(py, px)
                #     return {"pt": z_pt, "phi": z_phi}
                
                # def _plot_z_candidate(ax, zcand, label, color, ls="--", marker="p"):
                #     if zcand is None:
                #         return
                #     ph = _wrap_phi(np.array([zcand["phi"]]))[0]
                #     pt = zcand["pt"]
                #     ax.plot([ph, ph], [0, pt], color=color, alpha=0.7, lw=1.6, ls=ls)
                #     ax.scatter([ph], [pt], marker=marker, s=70, facecolors="none",
                #                edgecolors=color, linewidths=1.8, label=label)
                # def _mask_value(mask, iev):
                #     if mask is None:
                #         return "NA"
                #     try:
                #         return bool(mask[iev])
                #     except Exception:
                #         return "NA"
                
                # # ------------------------------------------------------------
                # # Main plotting function
                # # ------------------------------------------------------------

                # def add_event_debug_panel(
                #     ax,
                #     iev,
                #     allsel_reco,
                #     allsel_gen,
                #     npv_mask,
                #     MET_mask,
                #     kinsel_reco_mask,
                #     toposel_reco_mask,
                #     kinsel_gen_mask,
                #     toposel_gen_mask,
                #     is_matched_gen,
                #     reco_jet,
                #     gen_jet,
                #     z_reco,
                #     z_gen,
                # ):
                #     lines = []
                
                #     # --- selection bits ---
                #     lines.append("Selection:")
                #     lines.append(f" npv           : {_mask_value(npv_mask, iev)}")
                #     lines.append(f" MET           : {_mask_value(MET_mask, iev)}")
                #     lines.append(f" kinsel_reco   : {_mask_value(kinsel_reco_mask, iev)}")
                #     lines.append(f" toposel_reco  : {_mask_value(toposel_reco_mask, iev)}")
                #     lines.append(f" kinsel_gen    : {_mask_value(kinsel_gen_mask, iev)}")
                #     lines.append(f" toposel_gen   : {_mask_value(toposel_gen_mask, iev)}")
                #     lines.append(f" matched       : {_mask_value(is_matched_gen_mask, iev)}")
                #     lines.append(f" allsel_reco   : {_mask_value(allsel_reco, iev)}")
                #     lines.append(f" allsel_gen    : {_mask_value(allsel_gen, iev)}")
                
                #     # --- object info ---
                #     try:
                #         rj = reco_jet[iev]
                #         gj = gen_jet[iev]
                #         zr = z_reco[iev]
                #         zg = z_gen[iev]
                
                #         lines.append("Reco jet:")
                #         lines.append(f" pt = {float(rj.pt):.1f}")
                #         lines.append(f" eta= {float(rj.eta):.2f}")
                #         lines.append(f" phi= {float(rj.phi):.2f}")
                #         lines.append("")
                
                #         lines.append("Gen jet:")
                #         lines.append(f" pt = {float(gj.pt):.1f}")
                #         lines.append(f" eta= {float(gj.eta):.2f}")
                #         lines.append(f" phi= {float(gj.phi):.2f}")
                #         lines.append("")
                
                #         lines.append("Reco Z:")
                #         lines.append(f" pt = {float(zr.pt):.1f}")
                #         lines.append(f" m  = {float(zr.mass):.1f}")
                #         lines.append("")
                
                #         lines.append("Gen Z:")
                #         lines.append(f" pt = {float(zg.pt):.1f}")
                #         lines.append(f" m  = {float(zg.mass):.1f}")
                #         lines.append("")
                
                #         lines.append("Derived:")
                #         lines.append(f" dphi Z_reco-jet = {abs(float(zr.delta_phi(rj))):.3f}")
                #         lines.append(f" dphi Z_gen-jet  = {abs(float(zg.delta_phi(gj))):.3f}")
                #         lines.append(f" dR reco-gen jet = {float(rj.delta_r(gj)):.3f}")
                #         lines.append(f" pt balance reco = {float(rj.pt/zr.pt):.3f}")
                #         lines.append(f" pt balance gen  = {float(gj.pt/zg.pt):.3f}")
                
                #     except:
                #         lines.append("Object info missing")
                
                #     text = "\n".join(lines)
                
                #     ax.text(
                #         0.0,
                #         1.0,
                #         text,
                #         fontsize=12,
                #         verticalalignment="top",
                #         horizontalalignment="left",
                #         family="monospace",
                #     )
                # def plot_reco_not_gen_polar_events(
                #     events_j,
                #     allsel_reco,
                #     allsel_gen,
                #     reco_jet=None,
                #     gen_jet=None,
                #     z_reco=None,
                #     z_gen=None,
                #     npv_mask=None,
                #     MET_mask=None,
                #     kinsel_reco_mask=None,
                #     toposel_reco_mask=None,
                #     kinsel_gen_mask=None,
                #     toposel_gen_mask=None,
                #     is_matched_gen_mask=None,
                #     outdir="outputs/plots/eventImage",
                #     max_events=50,
                # ):
                #     os.makedirs(outdir, exist_ok=True)
                
                #     mask = ak.to_numpy(~allsel_reco & allsel_gen)
                #     idxs = np.flatnonzero(mask)[:max_events]
                
                #     print(f"Found {mask.sum()} events with allsel_reco=True and allsel_gen=False")
                #     print(f"Saving first {len(idxs)} event displays to {outdir}")
                
                #     for i, iev in enumerate(idxs):
                #         ev = events_j[iev]
                
                #         # estimate a sensible radial max from all visible objects
                #         pts_all = []
                
                #         for name in ["FatJet", "GenJetAK8", "Muon", "Electron", "GenDressedLepton"]:
                #             if name in ev.fields and len(getattr(ev, name)) > 0:
                #                 pts_all.extend(list(_to_1d_numpy(getattr(ev, name).pt)))
                
                #         if reco_jet is not None:
                #             try:
                #                 pts_all.append(float(ak.to_numpy(ak.atleast_1d(reco_jet[iev].pt))[0]))
                #             except Exception:
                #                 pass
                
                #         if gen_jet is not None:
                #             try:
                #                 pts_all.append(float(ak.to_numpy(ak.atleast_1d(gen_jet[iev].pt))[0]))
                #             except Exception:
                #                 pass
                
                #         rmax = max(50, 1.15 * max(pts_all)) if len(pts_all) else 100
                
                #         fig = plt.figure(figsize=(14, 8))
                #         #gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.9], wspace=0.15)
                #         gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.45, 0.9], wspace=0.1)
                #         ax      = fig.add_subplot(gs[0, 0], projection="polar")
                #         ax_leg  = fig.add_subplot(gs[0, 1])
                #         ax_info = fig.add_subplot(gs[0, 2])
                #         ax_leg.axis("off")
                #         ax_info.axis("off")
                        

                
                #         # --- Plot full collections ---
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "GenDressedLepton"),
                #             label="Gen Dressed Lepton", color="royalblue",
                #             marker="o", open_marker=True, annotate_pt=False, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "Electron"),
                #             label="Reco Electron", color="tab:green",
                #             marker="s", open_marker=True, annotate_pt=False, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "Muon"),
                #             label="Reco Muon", color="tab:red",
                #             marker="^", open_marker=True, annotate_pt=False, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "GenJetAK8"),
                #             label="Gen Jet", color="brown",
                #             marker="D", open_marker=False, annotate_pt=True, max_r=rmax
                #         )
                
                #         _plot_collection_polar(
                #             ax, _event_collection(ev, "FatJet"),
                #             label="Reco Jet", color="coral",
                #             marker="D", open_marker=True, annotate_pt=True, max_r=rmax
                #         )
                
                #         # --- Overlay selected reco/gen jet if provided ---
                #         # if reco_jet is not None:
                #         #     try:
                #         #         _plot_single_object_polar(
                #         #             ax, reco_jet[iev],
                #         #             label="Selected Reco Jet", color="darkorange",
                #         #             marker="D", open_marker=True, lw=2.6
                #         #         )
                #         #     except Exception:
                #         #         pass
                
                #         # if gen_jet is not None:
                #         #     try:
                #         #         _plot_single_object_polar(
                #         #             ax, gen_jet[iev],
                #         #             label="Selected Gen Jet", color="darkred",
                #         #             marker="D", open_marker=False, lw=2.6
                #         #         )
                #         #     except Exception:
                #         #         pass
                #         if reco_jet is not None:
                #             try:
                #                 _plot_single_object_polar(
                #                     ax, reco_jet[iev],
                #                     label="Selected Reco Jet",
                #                     color="gold",
                #                     marker="X",
                #                     open_marker=False,
                #                     lw=3.5,
                #                     alpha=1.0,
                #                 )
                #             except Exception:
                #                 pass
                        
                #         if gen_jet is not None:
                #             try:
                #                 _plot_single_object_polar(
                #                     ax, gen_jet[iev],
                #                     label="Selected Gen Jet",
                #                     color="black",
                #                     marker="*",
                #                     open_marker=False,
                #                     lw=3.5,
                #                     alpha=1.0,
                #                 )
                #             except Exception:
                #                 pass
                
                #         # --- Approximate Z from reco leptons ---
                #         reco_z = None
                #         if "Muon" in ev.fields and len(ev.Muon) >= 2:
                #             reco_z = _compute_event_z_from_leptons(ev.Muon)
                #         elif "Electron" in ev.fields and len(ev.Electron) >= 2:
                #             reco_z = _compute_event_z_from_leptons(ev.Electron)
                
                #         gen_z = _compute_event_z_from_leptons(_event_collection(ev, "GenDressedLepton"))
                
                #         # _plot_z_candidate(ax, reco_z,  label="Gen Z",  color="magenta", ls="-.", marker="p")
                #         # _plot_z_candidate(ax, gen_z, label="Reco Z", color="hotpink", ls="--", marker="p")

                #         reco_z = None
                #         gen_z  = None
                        
                #         if z_reco is not None:
                #             try:
                #                 reco_z = z_reco[iev]
                #             except:
                #                 pass
                        
                #         if z_gen is not None:
                #             try:
                #                 gen_z = z_gen[iev]
                #             except:
                #                 pass
                        
                #         _plot_single_object_polar(
                #             ax, reco_z,
                #             label="Reco Z",
                #             color="hotpink",
                #             marker="p",
                #             open_marker=True,
                #             lw=2.5,
                #         )
                        
                #         _plot_single_object_polar(
                #             ax, gen_z,
                #             label="Gen Z",
                #             color="magenta",
                #             marker="p",
                #             open_marker=False,
                #             lw=2.5,
                #         )
                
                #         # --- Optional Δφ between selected reco/gen jet ---
                #         title = f"Event {iev}"
                #         if reco_jet is not None and gen_jet is not None:
                #             try:
                #                 dphi = float(abs(reco_jet[iev].delta_phi(gen_jet[iev])))
                #                 title += f"   Δ(φ) = {dphi:.4f}"
                #             except Exception:
                #                 pass
                
                #         ax.set_title(title, pad=28, fontsize=16)
                #         ax.set_rmax(rmax)
                #         ax.set_rlabel_position(22.5)
                #         ax.set_xlabel(r"$\phi$ [rad]", labelpad=18)
                #         ax.set_ylabel(r"$\rho = p_T$ [GeV]", labelpad=25)
                #         ax.grid(True, alpha=0.65)
                
                #         handles, labels = ax.get_legend_handles_labels()
                #         ax_leg.legend(
                #             handles, labels,
                #             loc="upper left",
                #             fontsize=10,
                #             frameon=True,
                #         )

                #         add_event_debug_panel(
                #             ax_info,
                #             iev,
                #             allsel_reco,
                #             allsel_gen,
                #             npv_mask,
                #             MET_mask,
                #             kinsel_reco_mask,
                #             toposel_reco_mask,
                #             kinsel_gen_mask,
                #             toposel_gen_mask,
                #             is_matched_gen,
                #             reco_jet,
                #             gen_jet,
                #             z_reco,
                #             z_gen,
                #         )
                
                #         fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.08)
                #         fout = os.path.join(outdir, f"event_{i:03d}_entry_{iev}.png")
                #         plt.savefig(fout, dpi=160)
                #         plt.close(fig)
                
                #     return idxs
                # npv_mask          = sel.require(npv=True)
                # MET_mask          = sel.require(MET=True)
                # kinsel_reco_mask  = sel.require(kinsel_reco=True)
                # toposel_reco_mask = sel.require(toposel_reco=True)
                # kinsel_gen_mask   = sel.require(kinsel_gen=True)
                # toposel_gen_mask  = sel.require(toposel_gen=True)
                # is_matched_gen_mask = sel.require(is_matched_gen=True)

                # bad_idx = plot_reco_not_gen_polar_events(
                #     events_j=events_j,
                #     allsel_reco=allsel_reco,
                #     allsel_gen=allsel_gen,
                #     reco_jet=reco_jet,
                #     gen_jet=gen_jet,
                #     z_reco=z_reco,
                #     z_gen=z_gen,
                #     npv_mask=npv_mask,
                #     MET_mask=MET_mask,
                #     kinsel_reco_mask=kinsel_reco_mask,
                #     toposel_reco_mask=toposel_reco_mask,
                #     kinsel_gen_mask=kinsel_gen_mask,
                #     toposel_gen_mask=toposel_gen_mask,
                #     is_matched_gen_mask=is_matched_gen_mask,
                #     outdir="outputs/plots/eventImage",
                #     max_events=50,
                # )

                



                ######-----------_#############
                
                # Fill histograms
                # GEN level histograms
                self.logging.debug(f"Total reco events passing all selection: {sel.require(allsel_reco = True).sum()}",  )
                if self._do_gen:
                    self.logging.debug(f"Total gen events passing all selection: {sel.require(allsel_gen = True).sum()}")
                    self.logging.debug(f"Total events passing both reco and gen selections: {sel.require(allsel_reco = True, allsel_gen = True).sum()}")
                self.logging.debug(f"Total reco events (ee channel) passing all selection: {sel.require(allsel_reco = True, twoReco_ee = True).sum()}",  )
                self.logging.debug(f"Total reco events (mm channel) passing all selection: {sel.require(allsel_reco = True, twoReco_mm = True).sum()}",  )
                self.logging.debug(f"Weights sample: {weights.weight()[sel_reco][:10]}" )
                channels = ['mm', 'ee']
                for channel in channels:
                    
                    if channel == 'ee':
                        print("Now doing ee")
                        if self._do_gen:
                            sel_both = sel.require(allsel_reco = True, allsel_gen = True, twoGen_ee = True, twoReco_ee = True)
                            sel_gen = sel.require(allsel_gen = True, twoGen_ee = True)
                        sel_reco = sel.require(allsel_reco = True, twoReco_ee = True)
                    else:
                        print("Now doing mm")
                        if self._do_gen:
                            sel_both = sel.require(allsel_reco = True, allsel_gen = True, twoGen_mm = True, twoReco_mm = True)
                            sel_gen = sel.require(allsel_gen = True, twoGen_mm = True)
                        sel_reco = sel.require(allsel_reco = True, twoReco_mm = True)
                    jetR = 0.8
                    print(f"JET SYST {jet_syst}")
                    if jet_syst == "nominal":
                        self.logging.debug(f"list of systematics {self.systematics}")
                        for syst in self.systematics:
                            self.logging.debug(f"Processing systematic {syst}")
                            if syst == "nominal":
                                if self._do_gen:
                                    
                                    weights_gen =  weights.partial_weight(include=['genWeight'])[sel_gen]
                                    weights_both = weights.weight()[sel_both]
                                weights_reco = weights.weight()[sel_reco]
                                
                                
                            else:
                                weights_reco = weights.weight(modifier=syst)[sel_reco]
                                if self._do_gen:
                                    weights_both = weights.weight(modifier=syst)[sel_both]
                            if self._do_gen:
                                gen_jet_truth = gen_jet[sel_gen]
                                groomed_gen_jet_truth = groomed_gen_jet[sel_gen]
                                weights_gen = weights.weight( )[sel_gen]
                                

                                ptgen = gen_jet_truth.pt
                                ptgen = ptgen[~ak.is_none(ptgen)]
                                mgen = gen_jet_truth.mass
                                mgen = mgen[~ak.is_none(mgen)]
                                mgen_g = groomed_gen_jet_truth.mass
                                mgen_g = mgen_g[~ak.is_none(mgen_g)]
                                weights_gen = weights_gen[~ak.is_none(mgen)]
                                mass_jk_fill = {"jk": jk_index} if self._mode == "mass_jk" else {}

                                fill_hist(
                                    self.hists,
                                    "ptjet_mjet_u_gen",
                                    dataset=dataset,
                                    channel=channel,
                                    ptgen=ptgen,
                                    mgen=mgen,
                                    weight=weights_gen,
                                    systematic=syst,
                                    **mass_jk_fill,
                                )
                                fill_hist(
                                    self.hists,
                                    "ptjet_mjet_g_gen",
                                    dataset=dataset,
                                    channel=channel,
                                    ptgen=ptgen,
                                    mgen=mgen_g,
                                    weight=weights_gen,
                                    systematic=syst,
                                    **mass_jk_fill,
                                )

                                mpt_gen = 2*np.log10(mgen/(ptgen*jetR))
                                mpt_gen_g = 2*np.log10(mgen_g/(ptgen*jetR))

                                herwig_weight_g = get_herwig_weight_g().weight_array(ptgen, mpt_gen_g)
                                herwig_weight_u = get_herwig_weight_u().weight_array(ptgen, mpt_gen)    

                                if self._do_reweight:
                                    self.logging.debug("We are doing reweight right?")
                                    weights_gen_u = weights_gen * herwig_weight_u
                                    weights_gen_g = weights_gen * herwig_weight_g
                                    #self.logging.debug(f"And the weights are not false? {weights_gen_g}")

                                    fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = mpt_gen, weight = weights_gen_u, systematic = syst)
                                    
                                    fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = mpt_gen_g, weight = weights_gen_g, systematic = syst)
                                elif self._do_jk:
                                    fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen/(ptgen*jetR)), weight = weights_gen, jk = jk_index, systematic = syst)
                                    
                                    fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen_g/(ptgen*jetR)), weight = weights_gen, jk = jk_index, systematic = syst)
        



                                else:
                                    #self.logging.debug(f"No of GEN JET {len(mgen)}")  
                                    partonFlavour = gen_jet_truth.partonFlavour
                                    abs_pdg = abs(partonFlavour)
    
                                    is_quark = (
                                        (abs_pdg == 1) |
                                        (abs_pdg == 2) |
                                        (abs_pdg == 3) |
                                        (abs_pdg == 4) |
                                        (abs_pdg == 5)
                                    )
                                    
                                    is_gluon = (gen_jet_truth.partonFlavour == 21)
    
                                    parton_flavor = ak.where(is_quark, 1,
                                      ak.where(is_gluon, 2, 0))

                                    fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen/(ptgen*jetR)), weight = weights_gen, systematic = syst)
                                    
                                    fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                              mpt_gen = 2*np.log10(mgen_g/(ptgen*jetR)), weight = weights_gen, systematic = syst)

                                    fill_hist(self.hists, "pt_flavor_jet0_gen", dataset = dataset, pt = ptgen, n = parton_flavor )

                                    


                                        
                                # if (not self._do_jk) or (not self._do_reweight):
                                #     fill_hist(self.hists, "ptjet_rhojet_u_gen", dataset = dataset, ptgen = ptgen,
                                #               mpt_gen = 2*np.log10(mgen/(ptgen*jetR)), weight = weights_gen, systematic = syst)
                                    
                                #     fill_hist(self.hists, "ptjet_rhojet_g_gen", dataset = dataset, ptgen = ptgen,
                                #               mpt_gen = 2*np.log10(mgen_g/(ptgen*jetR)), weight = weights_gen, systematic = syst)

                                # else:## Rho jk
                                    

                                
                                


                                
                                
                                gen_jet_both = gen_jet[sel_both]
                                reco_jet_both = reco_jet[sel_both]
                                groomed_gen_jet_both = groomed_gen_jet[sel_both]
        
                                ptgen_both = gen_jet_both.pt
                                ptgen_both = ptgen_both[~ak.is_none(ptgen_both)]
                                mgen_both = gen_jet_both.mass
                                mgen_both = mgen_both[~ak.is_none(mgen_both)]
                                mgen_both_g = groomed_gen_jet_both.mass

                                mpt_gen_both = 2*np.log10(mgen_both/(ptgen_both*jetR))
                                mpt_gen_both_g = 2*np.log10(mgen_both_g/(ptgen_both*jetR))
        
                                #self.logging.debug(f"No of GEN JET also passing RECO {len(mgen_both)}")
                                
                                ptreco_both = reco_jet_both.pt
                                ptreco_both = ptreco_both[~ak.is_none(ptreco_both)]
                                mreco_both = reco_jet_both.mass
                                mreco_both = mreco_both[~ak.is_none(mreco_both)]

                                ptreco_both_g = reco_jet_both.pt
                                ptreco_both_g = ptreco_both_g[~ak.is_none(ptreco_both_g)]
                                mreco_both_g = reco_jet_both.msoftdrop
                                mreco_both_g = mreco_both_g[~ak.is_none(mreco_both_g)]
                                weights_both_g = weights_both[~ak.is_none(mreco_both_g)]

                                mpt_reco_both = 2*np.log10(np.abs(mreco_both/(ptreco_both*jetR)))
                                mpt_reco_both_g = 2*np.log10(np.abs(mreco_both_g/(ptreco_both_g*jetR)))


                                
                                jetR = 0.8
                                fill_hist(self.hists, "response_matrix_u", dataset = dataset, channel = channel, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, systematic = syst, weight = weights_both, **mass_jk_fill)

                                fill_hist(self.hists, "response_matrix_g", dataset = dataset, channel = channel, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g,systematic = syst, weight = weights_both, **mass_jk_fill)

                                # fill_hist(self.hists, 'm_u_jet_reco_over_gen', dataset = dataset, ptgen = ptgen_both, mgen = mgen_both, 
                                #           frac = (mreco_both-mgen_both) /mgen_both, weight = weights_both)

                                # fill_hist(self.hists, 'm_g_jet_reco_over_gen', dataset = dataset, ptgen = ptgen_both, mgen = mgen_both_g, 
                                #           frac = (mreco_both_g-mgen_both_g) /mgen_both_g, weight = weights_both)
                                

                                # import matplotlib.pyplot as plt
                                # plt.hist(2*np.log10(mreco_both/(ptreco_both*jetR)))
                                # plt.show()
                                


                                
                                if self._do_jk:### Rho jk
                                    fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both/(ptgen_both*jetR)), 
                                              weight = weights_both, jk = jk_index, systematic = syst)
                                        
                                    fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both_g/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both_g/(ptgen_both*jetR)), 
                                              weight = weights_both, jk = jk_index, systematic = syst)
                                elif self._do_reweight:

                                    herwig_weight_both_g = get_herwig_weight_g().weight_array(ptreco_both_g, mpt_reco_both_g)
                                    herwig_weight_both_g = ak.nan_to_num(herwig_weight_both_g, 0)
                                    herwig_weight_both_u = get_herwig_weight_u().weight_array(ptreco_both, mpt_reco_both)
                                    self.logging.debug(f"herwig WEights both g? {herwig_weight_both_g}" )
                                    weights_both_g = weights_both_g * herwig_weight_both_g
                                    weights_both_u = weights_both * herwig_weight_both_u
                                    # self.logging.debug(f"WEights both g? {weights_both_g}" )
                                    # self.logging.debug(f"How many nan in weights_both_g? {ak.sum(ak.is_none(weights_both_g))}")
                                    # self.logging.debug(f"How many nan in mpt_reco? {ak.sum(ak.is_none(mpt_reco_both))}")
                                    # self.logging.debug(f"How many nan in mpt_gen? {ak.sum(ak.is_none(mpt_gen_both))}")
                                    # self.logging.debug(f"how many nan in ptreco_both? {}")
                                    # self.logging.debug(f"how many nan in ptgen_both? {}")

                                    #self.logging.debug(f"WEights both g? {weights_both_g}")
                                    # self.logging.debug(
                                    #     f"How many NaN in weights_both_g? {ak.sum(np.isnan(herwig_weight_both_g))}"
                                    # )
                                    # self.logging.debug(
                                    #     f"How many NaN in weights_both_g? {ak.sum(np.isnan(weights_both_g))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in mpt_reco_both_g? {ak.sum(np.isnan(mpt_reco_both_g))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in mpt_gen_both? {ak.sum(np.isnan(mpt_gen_both_g))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in ptreco_both? {ak.sum(np.isnan(ptreco_both))}"
                                    # )
                                    
                                    # self.logging.debug(
                                    #     f"How many NaN in ptgen_both? {ak.sum(np.isnan(ptgen_both))}"
                                    # )
                                    fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = mpt_reco_both, ptgen = ptgen_both,
                                              mpt_gen = mpt_gen_both, systematic = syst, weight = weights_both_u)
                                    
                                    fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = mpt_reco_both_g, ptgen = ptgen_both,
                                              mpt_gen = mpt_gen_both_g, systematic = syst, weight = weights_both_g)
                                
                                else:
                                    fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both/(ptgen_both*jetR)), systematic = syst, weight = weights_both)
                                    
                                    fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                              mpt_reco = 2*np.log10(mreco_both_g/(ptreco_both*jetR)), ptgen = ptgen_both,
                                              mpt_gen = 2*np.log10(mgen_both_g/(ptgen_both*jetR)), systematic = syst, weight = weights_both)
                                
                                
                                fill_hist(self.hists, "jk_response_matrix_u",dataset = dataset, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, jk = jk_index, weight = weights_both)
                                fill_hist(self.hists, "jk_response_matrix_g", dataset = dataset, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, jk = jk_index, weight = weights_both)

                
                            reco_jet_meas = reco_jet[sel_reco]
                            ptreco = reco_jet_meas.pt
                            ptreco_g = reco_jet_meas.pt
                            mreco = reco_jet_meas.mass
                            mreco_g = reco_jet_meas.msoftdrop
                            #mreco_g2 = reco_jet_meas.msoftdrop_orig

                            ptreco = ptreco[~ak.is_none(mreco)]
                            mreco = mreco[~ak.is_none(mreco)]
                            mreco_g = mreco_g[~ak.is_none(mreco_g)]
                            #mreco_g2 = mreco_g2[~ak.is_none(mreco_g2)]
                            ptreco_g = ptreco_g[~ak.is_none(mreco_g)]
                            weights_reco_g = weights_reco[~ak.is_none(mreco_g)]
                            weights_reco = weights_reco[~ak.is_none(mreco)]

                            mpt_reco = 2*np.log10(mreco/(ptreco*jetR))
                            mpt_reco_g = 2*np.log10(mreco_g/(ptreco*jetR))

                            events_j_meas = events_j[sel_reco]
                            z_reco_meas = z_reco[sel_reco]
                            print("Fatjet y ", reco_jet_meas.rapidity)
                            print("Fatjet eta ", reco_jet_meas.eta)
                            # print(
                            #     "energy:", reco_jet_meas.p4.energy,
                            #     "\npz:", reco_jet_meas.p4.pz,
                            #     "\np4.energy:", reco_jet_meas.p4.energy,
                            #     "\np4.pz:", reco_jet_meas.p4.pz
                            # )
                            if channel == "mm":
                                pt_mu0  = events_j_meas.Muon[:, 0].pt
                                eta_mu0 = events_j_meas.Muon[:, 0].eta
                                phi_mu0 = events_j_meas.Muon[:, 0].phi
                            
                                pt_mu1  = events_j_meas.Muon[:, 1].pt
                                eta_mu1 = events_j_meas.Muon[:, 1].eta
                                phi_mu1 = events_j_meas.Muon[:, 1].phi
                                
                                y_mu0 = events_j_meas.Muon[:,0].rapidity
                                y_mu1 = events_j_meas.Muon[:,1].rapidity
                                
                                                         
                                q0 = events_j_meas.Muon[:, 0].charge
                                q1 = events_j_meas.Muon[:, 1].charge
                            
                                pt_mupos  = ak.where(q0 > 0, pt_mu0, pt_mu1)
                                pt_muneg  = ak.where(q0 < 0, pt_mu0, pt_mu1)
                            
                                eta_mupos = ak.where(q0 > 0, eta_mu0, eta_mu1)
                                eta_muneg = ak.where(q0 < 0, eta_mu0, eta_mu1)
                            
                                phi_mupos = ak.where(q0 > 0, phi_mu0, phi_mu1)
                                phi_muneg = ak.where(q0 < 0, phi_mu0, phi_mu1)

                                y_mupos = ak.where(q0 > 0, y_mu0, y_mu1)
                                y_muneg = ak.where(q0 < 0, y_mu0, y_mu1)   
                            
                            
                            if channel == "ee":
                                pt_el0  = events_j_meas.Electron[:, 0].pt
                                eta_el0 = events_j_meas.Electron[:, 0].eta
                                phi_el0 = events_j_meas.Electron[:, 0].phi
                            
                                pt_el1  = events_j_meas.Electron[:, 1].pt
                                eta_el1 = events_j_meas.Electron[:, 1].eta
                                phi_el1 = events_j_meas.Electron[:, 1].phi
                                y_el0 = events_j_meas.Electron[:,0].rapidity
                                y_el1 = events_j_meas.Electron[:,1].rapidity
                                
                                q0 = events_j_meas.Electron[:, 0].charge
                                q1 = events_j_meas.Electron[:, 1].charge

                                
                                y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                                y_elneg = ak.where(q0 < 0, y_el0, y_el1)
                                pt_elpos  = ak.where(q0 > 0, pt_el0, pt_el1)
                                pt_elneg  = ak.where(q0 < 0, pt_el0, pt_el1)
                            
                                eta_elpos = ak.where(q0 > 0, eta_el0, eta_el1)
                                eta_elneg = ak.where(q0 < 0, eta_el0, eta_el1)
                            
                                phi_elpos = ak.where(q0 > 0, phi_el0, phi_el1)
                                phi_elneg = ak.where(q0 < 0, phi_el0, phi_el1)

                                y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                                y_elneg = ak.where(q0 < 0, y_el0, y_el1)


                            
                            pt_Z = z_reco_meas.pt
                            eta_Z = z_reco_meas.eta
                            phi_Z = z_reco_meas.phi
                            mass_Z = z_reco_meas.mass
                            nJet = ak.num(events_j_meas.FatJet, axis = 1)
                            
                            dr = z_jet_dr_reco[sel_reco]
                            dphi = z_jet_dphi_reco[sel_reco]
                            ptasym = z_pt_asym_reco[sel_reco]
                            
                            


                            
                            
                    
                            #self.logging.debug(f"No of RECO JET {len(mreco)}")
                            if syst == "nominal":
                                self.logging.debug(f"Len of ptreco {len(ptreco)} mreco {len(mreco)} syst {syst} channel {channel} dataset {dataset}")
                                self.logging.debug(f"ptreco sample {ptreco[:10]}")
                                self.logging.debug(f"mreco sample {mreco[:10]}")
                                self.logging.debug(f"mreco_g sample {mreco_g[:10]}")
                            fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = syst, weight = weights_reco, **mass_jk_fill)
                            fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = syst, weight = weights_reco_g, **mass_jk_fill)


                                


                            if self._do_jk:## Rho jk
                                fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                      mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), jk = jk_index, weight = weights_reco, systematic = syst)
                            
                                fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                          mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), jk = jk_index, weight = weights_reco_g, systematic = syst)
                            elif self._do_reweight:
                                herwig_weight_reco_g = get_herwig_weight_g().weight_array(ptreco, mpt_reco_g)
                                herwig_weight_reco_u = get_herwig_weight_u().weight_array(ptreco, mpt_reco)
                                weights_reco_g = weights_reco_g * herwig_weight_reco_g
                                weights_reco_u = weights_reco * herwig_weight_reco_u
                                fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                      mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), systematic = syst, weight = weights_reco_u)
                            
                                fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                          mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), systematic = syst, weight = weights_reco_g)
                            else: # regular cases
                                
                                ### Required input distributions
                                fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                      mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), systematic = syst, weight = weights_reco)
                            
                                fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                          mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), systematic = syst, weight = weights_reco_g)

                                ### Required for data/MC validation
                                # weights_gen_reco = weights.partial_weight('genWeight')[sel_reco]
                                # print(weights_gen_reco)
                                ## Filling validation plots ##
                                if channel == "mm":
                                    fill_hist(self.hists, "pt_mupos",  dataset=dataset, pt=pt_mupos,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_mupos", dataset=dataset, eta=eta_mupos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_mupos", dataset=dataset, phi=phi_mupos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_mupos",   dataset=dataset, y=y_mupos,     systematic=syst, weight=weights_reco)
                                
                                    fill_hist(self.hists, "pt_muneg",  dataset=dataset, pt=pt_muneg,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_muneg", dataset=dataset, eta=eta_muneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_muneg", dataset=dataset, phi=phi_muneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_muneg",   dataset=dataset, y=y_muneg,     systematic=syst, weight=weights_reco)

                                    del (
                                        pt_mu0, pt_mu1,
                                        eta_mu0, eta_mu1,
                                        phi_mu0, phi_mu1,
                                        y_mu0, y_mu1,
                                        q0, q1,
                                        pt_mupos, pt_muneg,
                                        eta_mupos, eta_muneg,
                                        phi_mupos, phi_muneg,
                                        y_mupos, y_muneg
                                    )
                                
                                
                                # electrons (pos/neg)
                                if channel == "ee":
                                    fill_hist(self.hists, "pt_elpos",  dataset=dataset, pt=pt_elpos,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_elpos", dataset=dataset, eta=eta_elpos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_elpos", dataset=dataset, phi=phi_elpos, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_elpos",   dataset=dataset, y=y_elpos,     systematic=syst, weight=weights_reco)
                                
                                    fill_hist(self.hists, "pt_elneg",  dataset=dataset, pt=pt_elneg,   systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "eta_elneg", dataset=dataset, eta=eta_elneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "phi_elneg", dataset=dataset, phi=phi_elneg, systematic=syst, weight=weights_reco)
                                    fill_hist(self.hists, "y_elneg",   dataset=dataset, y=y_elneg,     systematic=syst, weight=weights_reco)

                                    del (
                                        pt_el0, pt_el1,
                                        eta_el0, eta_el1,
                                        phi_el0, phi_el1,
                                        y_el0, y_el1,
                                        q0, q1,
                                        pt_elpos, pt_elneg,
                                        eta_elpos, eta_elneg,
                                        phi_elpos, phi_elneg,
                                        y_elpos, y_elneg
                                    )
                                
                                ## nJets
                                fill_hist(self.hists, "nJets", dataset = dataset, n = nJet, systematic = syst, weight = weights_reco )
                                
                                ## Z object
                                fill_hist(self.hists, "pt_Z", dataset = dataset, pt = pt_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "eta_Z", dataset = dataset, eta = eta_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "phi_Z", dataset = dataset, phi = phi_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "mass_Z", dataset = dataset, mass = mass_Z, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "y_Z", dataset = dataset, y = getRapidity(z_reco_meas), systematic = syst, weight = weights_reco)

                                
                                ## Jet
                                fill_hist(self.hists, "pt_jet0", dataset = dataset, pt = ptreco, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "eta_jet0", dataset = dataset, eta = reco_jet_meas.eta, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "phi_jet0", dataset = dataset, phi = reco_jet_meas.phi, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "mass_jet0", dataset = dataset, mass = reco_jet_meas.mass, systematic = syst, weight = weights_reco )
                                fill_hist(self.hists, "y_jet0", dataset = dataset, y = reco_jet_meas.rapidity, systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "eta_phi_jet_reco", dataset = dataset, eta = reco_jet_meas.eta, phi = reco_jet_meas.phi, weight = weights_reco)

                                ## Combo
                                fill_hist(self.hists, "ptasym", dataset = dataset, frac = ptasym, systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "dr", dataset = dataset, dr = dr, systematic = syst, weight = weights_reco)
                                fill_hist(self.hists, "dphi", dataset = dataset, dphi = dphi, systematic = syst, weight = weights_reco)
                                
                            

                            # fill_hist(self.hists, "ptjet_rhojet_g_reco2", dataset = dataset, ptreco = ptreco_g,
                            #           mpt_reco = 2*np.log10(mreco_g2/(ptreco*jetR)), systematic = syst, weight = weights_reco_g)
                            
                            fill_hist( self.hists, "jk_ptjet_mjet_u_reco", dataset = dataset,ptreco = ptreco, mreco = mreco, systematic = syst, weight = weights_reco, jk = jk_index)
                            fill_hist( self.hists, "jk_ptjet_mjet_g_reco", dataset = dataset,ptreco = ptreco_g, mreco = mreco_g, systematic = syst, weight = weights_reco_g, jk = jk_index)
                            
                            if not self._do_gen:
                                break # Break on nominal when running over data




                    else: # jet syst is not nominal
                        print("ARE WE ENTERING THIS SECTION OR NO?")
                        if self._do_gen:
                            weights_gen =  weights.partial_weight(include=['genWeight'])[sel_gen]
                            weights_both = weights.weight()[sel_both]
                        weights_reco = weights.weight()[sel_reco]
                        mass_jk_fill = {"jk": jk_index} if self._mode == "mass_jk" else {}
                        

                        reco_jet_meas = reco_jet[sel_reco]
                        ptreco = reco_jet_meas.pt
                        ptreco_g = reco_jet_meas.pt
                        mreco = reco_jet_meas.mass
                        mreco_g = reco_jet_meas.msoftdrop

                        ptreco = ptreco[~ak.is_none(mreco)]
                        mreco = mreco[~ak.is_none(mreco)]
                        mreco_g = mreco_g[~ak.is_none(mreco_g)]
                        ptreco_g = ptreco_g[~ak.is_none(mreco_g)]
                        weights_reco_g = weights_reco[~ak.is_none(mreco_g)]
                        weights_reco = weights_reco[~ak.is_none(mreco)]


                        events_j_meas = events_j[sel_reco]
                        z_reco_meas = z_reco[sel_reco]
                        if channel == "mm":
                            pt_mu0  = events_j_meas.Muon[:, 0].pt
                            eta_mu0 = events_j_meas.Muon[:, 0].eta
                            phi_mu0 = events_j_meas.Muon[:, 0].phi
                        
                            pt_mu1  = events_j_meas.Muon[:, 1].pt
                            eta_mu1 = events_j_meas.Muon[:, 1].eta
                            phi_mu1 = events_j_meas.Muon[:, 1].phi
                            
                            y_mu0 = events_j_meas.Muon[:,0].rapidity
                            y_mu1 = events_j_meas.Muon[:,1].rapidity
                            
                                                     
                            q0 = events_j_meas.Muon[:, 0].charge
                            q1 = events_j_meas.Muon[:, 1].charge
                        
                            pt_mupos  = ak.where(q0 > 0, pt_mu0, pt_mu1)
                            pt_muneg  = ak.where(q0 < 0, pt_mu0, pt_mu1)
                        
                            eta_mupos = ak.where(q0 > 0, eta_mu0, eta_mu1)
                            eta_muneg = ak.where(q0 < 0, eta_mu0, eta_mu1)
                        
                            phi_mupos = ak.where(q0 > 0, phi_mu0, phi_mu1)
                            phi_muneg = ak.where(q0 < 0, phi_mu0, phi_mu1)

                            y_mupos = ak.where(q0 > 0, y_mu0, y_mu1)
                            y_muneg = ak.where(q0 < 0, y_mu0, y_mu1)   
                        
                        
                        if channel == "ee":
                            pt_el0  = events_j_meas.Electron[:, 0].pt
                            eta_el0 = events_j_meas.Electron[:, 0].eta
                            phi_el0 = events_j_meas.Electron[:, 0].phi
                        
                            pt_el1  = events_j_meas.Electron[:, 1].pt
                            eta_el1 = events_j_meas.Electron[:, 1].eta
                            phi_el1 = events_j_meas.Electron[:, 1].phi
                            y_el0 = events_j_meas.Electron[:,0].rapidity
                            y_el1 = events_j_meas.Electron[:,1].rapidity
                            q0 = events_j_meas.Electron[:, 0].charge
                            q1 = events_j_meas.Electron[:, 1].charge
                            
                            y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                            y_elneg = ak.where(q0 < 0, y_el0, y_el1)
                        

                        
                            pt_elpos  = ak.where(q0 > 0, pt_el0, pt_el1)
                            pt_elneg  = ak.where(q0 < 0, pt_el0, pt_el1)
                        
                            eta_elpos = ak.where(q0 > 0, eta_el0, eta_el1)
                            eta_elneg = ak.where(q0 < 0, eta_el0, eta_el1)
                        
                            phi_elpos = ak.where(q0 > 0, phi_el0, phi_el1)
                            phi_elneg = ak.where(q0 < 0, phi_el0, phi_el1)

                            y_elpos = ak.where(q0 > 0, y_el0, y_el1)
                            y_elneg = ak.where(q0 < 0, y_el0, y_el1)


                        
                        pt_Z = z_reco_meas.pt
                        eta_Z = z_reco_meas.eta
                        phi_Z = z_reco_meas.phi
                        mass_Z = z_reco_meas.mass
                        nJet = ak.num(events_j_meas.FatJet, axis = 1)
                        
                        dr = z_jet_dr_reco[sel_reco]
                        dphi = z_jet_dphi_reco[sel_reco]
                        ptasym = z_pt_asym_reco[sel_reco]

                        #self.logging.debug(f"No of RECO JET {len(mreco)}")
                            
                        fill_hist(self.hists, "ptjet_rhojet_u_reco", dataset = dataset, ptreco = ptreco,
                                  mpt_reco = 2*np.log10(mreco/(ptreco*jetR)), systematic = jet_syst, weight = weights_reco)
                        
                        fill_hist(self.hists, "ptjet_rhojet_g_reco", dataset = dataset, ptreco = ptreco_g,
                                  mpt_reco = 2*np.log10(mreco_g/(ptreco*jetR)), systematic = jet_syst, weight = weights_reco_g)
                        print("HOW IS DOJK TRUE?")
                        print(self._do_jk)
                        ## Filling Rho
                        if not self._do_jk:
                            print("ARE WE FILLING THIS OR NO?")
                            fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = jet_syst, weight = weights_reco_g)

                            ### Required for data/MC validation
                            if channel == "mm":
                                fill_hist(self.hists, "pt_mupos",  dataset=dataset, pt=pt_mupos,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_mupos", dataset=dataset, eta=eta_mupos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_mupos", dataset=dataset, phi=phi_mupos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_mupos",   dataset=dataset, y=y_mupos,     systematic=jet_syst, weight=weights_reco)
                            
                                fill_hist(self.hists, "pt_muneg",  dataset=dataset, pt=pt_muneg,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_muneg", dataset=dataset, eta=eta_muneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_muneg", dataset=dataset, phi=phi_muneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_muneg",   dataset=dataset, y=y_muneg,     systematic=jet_syst, weight=weights_reco)

                                del (
                                    pt_mu0, pt_mu1,
                                    eta_mu0, eta_mu1,
                                    phi_mu0, phi_mu1,
                                    y_mu0, y_mu1,
                                    q0, q1,
                                    pt_mupos, pt_muneg,
                                    eta_mupos, eta_muneg,
                                    phi_mupos, phi_muneg,
                                    y_mupos, y_muneg
                                )
                            
                            
                            # electrons (pos/neg)
                            if channel == "ee":
                                fill_hist(self.hists, "pt_elpos",  dataset=dataset, pt=pt_elpos,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_elpos", dataset=dataset, eta=eta_elpos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_elpos", dataset=dataset, phi=phi_elpos, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_elpos",   dataset=dataset, y=y_elpos,     systematic=jet_syst, weight=weights_reco)
                            
                                fill_hist(self.hists, "pt_elneg",  dataset=dataset, pt=pt_elneg,   systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "eta_elneg", dataset=dataset, eta=eta_elneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "phi_elneg", dataset=dataset, phi=phi_elneg, systematic=jet_syst, weight=weights_reco)
                                fill_hist(self.hists, "y_elneg",   dataset=dataset, y=y_elneg,     systematic=jet_syst, weight=weights_reco)

                                del (
                                    pt_el0, pt_el1,
                                    eta_el0, eta_el1,
                                    phi_el0, phi_el1,
                                    y_el0, y_el1,
                                    q0, q1,
                                    pt_elpos, pt_elneg,
                                    eta_elpos, eta_elneg,
                                    phi_elpos, phi_elneg,
                                    y_elpos, y_elneg
                                )
                            
                            ## nJets
                            fill_hist(self.hists, "nJets", dataset = dataset, n = nJet, systematic = jet_syst, weight = weights_reco )
                            
                            ## Z object
                            fill_hist(self.hists, "pt_Z", dataset = dataset, pt = pt_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "eta_Z", dataset = dataset, eta = eta_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "phi_Z", dataset = dataset, phi = phi_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "mass_Z", dataset = dataset, mass = mass_Z, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "y_Z", dataset = dataset, y = getRapidity(z_reco_meas), systematic = jet_syst, weight = weights_reco)
                            ## Jet
                            fill_hist(self.hists, "pt_jet0", dataset = dataset, pt = ptreco, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "eta_jet0", dataset = dataset, eta = reco_jet_meas.eta, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "phi_jet0", dataset = dataset, phi = reco_jet_meas.phi, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "mass_jet0", dataset = dataset, mass = reco_jet_meas.mass, systematic = jet_syst, weight = weights_reco )
                            fill_hist(self.hists, "y_jet0", dataset = dataset, y = reco_jet_meas.rapidity, systematic = jet_syst, weight = weights_reco)

                            ## Combo
                            fill_hist(self.hists, "ptasym", dataset = dataset, frac = ptasym, systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "dr", dataset = dataset, dr = dr, systematic = jet_syst, weight = weights_reco)
                            fill_hist(self.hists, "dphi", dataset = dataset, dphi = dphi, systematic = jet_syst, weight = weights_reco)

                        else:
                            fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = jet_syst, weight = weights_reco, **mass_jk_fill)
                            fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = jet_syst, weight = weights_reco_g, **mass_jk_fill)
                            

                        if self._do_gen:
                            gen_jet_both = gen_jet[sel_both]
                            reco_jet_both = reco_jet[sel_both]
                            groomed_gen_jet_both = groomed_gen_jet[sel_both]

                            ptgen_both = gen_jet_both.pt
                            ptgen_both = ptgen_both[~ak.is_none(ptgen_both)]
                            mgen_both = gen_jet_both.mass
                            mgen_both = mgen_both[~ak.is_none(mgen_both)]
                            mgen_both_g = groomed_gen_jet_both.mass

                            #self.logging.debug(f"No of GEN JET also passing RECO {len(mgen_both)}")
                            
                            ptreco_both = reco_jet_both.pt
                            ptreco_both = ptreco_both[~ak.is_none(ptreco_both)]
                            mreco_both = reco_jet_both.mass
                            mreco_both = mreco_both[~ak.is_none(mreco_both)]
                            weights_both = weights_both[~ak.is_none(mreco_both)]
                            
                            fill_hist(self.hists, "response_matrix_u", dataset = dataset, channel = channel, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, systematic = jet_syst, weight = weights_both, **mass_jk_fill)
                            ptreco_both_g = reco_jet_both.pt
                            ptreco_both_g = ptreco_both_g[~ak.is_none(ptreco_both_g)]
                            mreco_both_g = reco_jet_both.msoftdrop
                            mreco_both_g = mreco_both_g[~ak.is_none(mreco_both_g)]
                            weights_both_g = weights_both[~ak.is_none(mreco_both_g)]
                            fill_hist(self.hists, "response_matrix_g", dataset = dataset, channel = channel, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, systematic = jet_syst, weight = weights_both_g, **mass_jk_fill)

                            ### Rho filling

                            fill_hist(self.hists, "response_matrix_rho_u", dataset = dataset,  ptreco = ptreco_both,
                                      mpt_reco = 2*np.log10(mreco_both/(ptreco_both*jetR)), ptgen = ptgen_both,
                                      mpt_gen = 2*np.log10(mgen_both/(ptgen_both*jetR)), systematic = jet_syst,
                                      weight = weights_both)
                                
                            fill_hist(self.hists, "response_matrix_rho_g", dataset = dataset,  ptreco = ptreco_both,
                                      mpt_reco = 2*np.log10(mreco_both_g/(ptreco_both*jetR)), ptgen = ptgen_both,
                                      mpt_gen = 2*np.log10(mgen_both_g/(ptgen_both*jetR)), systematic = jet_syst,
                                      weight = weights_both)


                            
                            
                            #register_hist(self.hists, "jk_response_matrix_u", [ ptreco_axis, mreco_axis, ptgen_axis, mgen_axis, binning.jackknife_axis])
                            fill_hist(self.hists, "jk_response_matrix_u",dataset = dataset, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, jk = jk_index, weight = weights_both)
                            fill_hist(self.hists, "jk_response_matrix_g",dataset = dataset, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, jk = jk_index, weight = weights_both_g)

                            
                            # End of channels loop
                
                if not self._do_gen:
                    self.logging.debug("This break condition is true")
                    break  # Exit the jet_syst loop if not doing GEN analysis

            if not self._do_jk:
                break  # Exit the JK loop if not doing jackknife resampling

        for name in sel.names:
            self.hists["cutflow"][f"{dataset}_{name}"] += sel.all(name).sum()
        

        return self.hists

    def postprocess(self, accumulator):
        hname_list = [key for key in accumulator.keys() if key not in ("cutflow", "nev", "sumw")]
        lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
        #hname_list = ["ptjet_mjet_u_reco", 'ptjet_mjet_g_reco', "ptjet_mjet_u_gen", "ptjet_mjet_g_gen", "response_matrix_u", "response_matrix_g"]
        sumw = accumulator["sumw"]
        
        for hname in hname_list:
            
            h = accumulator[hname]
            for i,ds in enumerate(h.axes['dataset']):
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    continue
                elif 'pythia' in ds:
                    xsdb = {
                        'HT-70to100': 140.0	,
                        'HT-100to200': 139.2,
                        'HT-200to400': 38.4,
                        'HT-400to600': 5.174,
                        'HT-600to800': 1.258,
                        'HT-800to1200': 0.5598,
                        'HT-1200to2500': 0.1305,
                        'HT-2500toInf': 0.002997,  
                        
                    }
                    
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    ht_bin = ds.split('_')[-1]
                    iov = ds.split('_')[-2]
                    xs = xsdb[ht_bin]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    
                    

                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    
                    scale = (xs * lumi_fb * 1000) / sw
                    k_factor = 1.1297638966
                    scale = scale*k_factor
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")


                

                    
                elif 'herwig' in ds:
                    xs = 5.036
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-2]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif 'powheg' in ds:
                    xs = 1976.0
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-2]
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('st' in ds) or ('ST' in ds):
                    xsdb = {'st_tW_antitop': 34.97,
                                'st_tW_top': 34.91	,
                                'ST_t-channel_antitop': 67.93,
                                'ST_t-channel_top':   69.09}
                    if ds.startswith('st_tW_antitop'):
                        xs = xsdb['st_tW_antitop']
                    elif ds.startswith('st_tW_top'):
                        xs = xsdb['st_tW_top']
                    elif ds.startswith('ST_t-channel_antitop'):
                        xs = xsdb['ST_t-channel_antitop']
                    else:
                        xs = xsdb['ST_t-channel_top']
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('ww' in ds):
                    xs = 75.95
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('wz' in ds):
                    xs = 27.6
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('zz' in ds):
                    xs = 12.17
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                elif ('ttjets' in ds):
                    xs = 471.7
                    lumi_db = {'UL16NanoAODv9':19.52 , 'UL16NanoAODAPVv9': 16.81 ,'UL17NanoAODv9': 41.48 , 'UL18NanoAODv9': 59.83}
                    iov = ds.split('_')[-1]
                    # self.logging.debug(f"{iov}")
                    lumi_fb = lumi_db[iov]
                    sw = sumw[ds]
                    if xs is None:
                        print(f"[postprocess] WARNING: missing XS_PB for dataset '{name}'. Skipping normalization.")
                        continue
                    if sw == 0.0:
                        print(f"[postprocess] WARNING: sumw==0 for dataset '{name}'. Skipping normalization.")
                        continue
                    scale = (xs * lumi_fb * 1000) / sw
                    h.view(flow=True)[i] *= scale
                    if i==0:
                        self.logging.info(f"Scaled {hname} for dataset {ds} by {scale:.6f} = {xs} * {lumi_fb*1000} / {sw}")
                else:
                    self.logging.info(f"Hist {hname} is not listed, no scaling done")
                    
            grouping = defaultdict(list)
            
            for ds in h.axes["dataset"]:
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    grouping[ds].append(ds)
                    continue
    
                
                if "pythia" in ds:
                    iov = ds.split("_")[-2]
                    new_key = f"pythia_{iov}"
                elif "herwig" in ds:
                    iov = ds.split("_")[-2]
                    new_key = f"herwig_{iov}"
                elif ("st" in ds) or ("ST" in ds):
                    iov = ds.split("_")[-1]
                    new_key = f"ST_{iov}"
                elif "ww" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"ww_{iov}"
                elif "wz" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"wz_{iov}"
                elif "zz" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"zz_{iov}"
                elif "ttjets" in ds:
                    iov = ds.split("_")[-1]
                    new_key = f"ttjets_{iov}"
                else:
                    iov = ds.split("_")[-1]
                    new_key = f"MC_{iov}"
                grouping[new_key].append(ds)
        
                # 3) Merge with the no-growth workaround (preserves axis order)
            h = group(h, oldname="dataset", newname="dataset", grouping=dict(grouping))
            accumulator[hname] = h

        return accumulator

                    
            
