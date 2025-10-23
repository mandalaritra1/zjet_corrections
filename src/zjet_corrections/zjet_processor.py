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
#from coffea.analysis_tools import PackedSelection
from collections import defaultdict
import gc
import tokenize as tok
import re


from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .weight_class import Weights, PackedSelection

from .hist_utils import *
from .smp_utils import *

from .corrections import *

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

        if mode == "minimal":
            self._do_jk = False
        elif mode == "jk_mc" or mode == "jk_data":
            self._do_jk = True
        elif mode == "full":
            self._do_jk = False
        else:
            self._do_jk = False
        

        if jet_systematics == None:
            self.jet_systematics = ["nominal"]
        else:
            self.jet_systematics = jet_systematics

        if systematics == None:
            self.systematics = ['nominal', 'puUp', 'puDown' , 'elerecoUp', 'elerecoDown', 
                                    'eleidUp', 'eleidDown', 'eletrigUp', 'eletrigDown', 'murecoUp', 'murecoDown', 
                                    'muidUp', 'muidDown', 'mutrigUp', 'mutrigDown', 'muisoUp', 'muisoDown',
                                    'pdfUp', 'pdfDown', 'q2Up', 'q2Down',
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
                register_hist(self.hists, "ptjet_mjet_u_gen", [dataset_axis,channel_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptjet_mjet_g_gen", [dataset_axis,channel_axis, ptgen_axis, mgen_axis, syst_axis])
                register_hist(self.hists, "ptz_mz_reco" , [dataset_axis, zmass_axis, pt_axis])

        
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
        for jk_index in range(0, 10): ## loops from 0 to 9 in case do_jk flag is enabled, otherwise breaks at 0
            if not self._do_jk:
                events1 = events_all
                self.logging.debug("Jackknife resampling not enabled, processing all events together.")
            if self._do_jk:
                jk_sel = ak.where( (index_list % 10) == jk_index, False, True)
                self.logging.debug("JK index ", jk_index, " events dropped ", ak.sum(~jk_sel) )
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
                lumi_mask = self.lumimasks[IOV](events_all.run, events_all.luminosityBlock)

                #events_all = events_all[lumi_mask]
            self.logging.info(f"year: {year}, ht_bin: {ht_bin}, herwig: {herwig}")
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
            
            # Store GEN weights or ones based on Simulation/Data
            if self._do_gen:
                
                
                weights.add("genWeight", events0.genWeight)
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

                eta_cut = np.abs(events0.GenDressedLepton.eta) < 2.5

                events0 = ak.with_field(
                    events0,
                    events0.GenDressedLepton[pt_cut & eta_cut],
                    "GenDressedLepton"
                )

                events0 = ak.with_field(
                                    events0,
                                    events0.GenJetAK8[(events0.GenJetAK8.pt > 0)
                                                & (np.abs(events0.GenJetAK8.eta) < 2.5)
                                    ],
                                    "GenJetAK8"
                )

                sel.add("oneGenJet", 
                        ak.sum( (events0.GenJetAK8.pt > 0) & (np.abs(events0.GenJetAK8.eta) < 2.5), axis=1 ) >= 1
                    )
                sel.add("oneGenJet_pt200", 
                        ak.sum( (events0.GenJetAK8.pt > 200) & (np.abs(events0.GenJetAK8.eta) < 2.5), axis=1 ) >= 1
                    )


                z_gen = get_z_gen_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
                z_ptcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  z_gen.pt > 90., False )
                z_mcut_gen = ak.where( sel.all("twoGen_leptons") & ~ak.is_none(z_gen),  (z_gen.mass > 71.) & (z_gen.mass < 111), False )

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

            

            # RECO Selection
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

            events0 = ak.with_field(
                    events0,
                    events0.Electron[(events0.Electron.pt > self.lepptcuts[0]) 
                                    & (np.abs(events0.Electron.eta) < 2.5) 
                                    & (events0.Electron.pfRelIso03_all < 0.25)  # supressing isolation cut here
                                    & (events0.Electron.cutBased > 3) ## TightId == 4 , mediumId == 3, looseId == 2
                                    & (events0.Electron.dz < 0.2) ## dz < 0.2 cm to prevent pileup
                    ],
                    "Electron"
                )

                

                

                
            events0 = ak.with_field(
                    events0,
                    events0.Muon[(events0.Muon.pt > self.lepptcuts[1]) 
                                &(np.abs(events0.Muon.eta) < 2.5)
                                &(events0.Muon.pfIsoId > 1) #medium iso, pfIso04 < 0.2 , 2 = loose, 3 = medium # supressing isolation cut here
                                #&(events0.Muon.miniIsoId > 1)
                                &(events0.Muon.mediumId	 == True)
                                & (events0.Muon.dz < 0.2) ## dz < 0.2 cm to prevent pileup
                                #&(events0.Muon.looseId	 == True)
                    
                    ],
                    "Muon"
                )
            self.logging.debug("Leptons Selected")
            z_reco = get_z_reco_selection(events0, sel, self.lepptcuts[0], self.lepptcuts[1], None, None)
            z_ptcut_reco = z_reco.pt > 90
            z_mcut_reco = (z_reco.mass > 71.) & (z_reco.mass < 111.)
            sel.add("z_ptcut_reco", z_ptcut_reco & (sel.require(twoReco_leptons = True) ))
            sel.add("z_mcut_reco", z_mcut_reco & (sel.require(twoReco_leptons = True) ))

            self.logging.debug("Z Object Created")
            #### dr reco plots ###

            twoReco_ee_sel = sel.require(twoReco_ee = True)
            twoReco_mm_sel = sel.require(twoReco_mm = True)
            twoReco_ll_sel = sel.require(twoReco_leptons = True)
            corr_jets = GetJetCorrections(events1.FatJet, events1, era, IOV, isData = not self._do_gen, mode='AK8')  ###### correcting FatJet.mass
            self.logging.debug("Jet Corrections Applied")

            for jet_syst in self.jet_systematics: # Start loop over jet systematics
                self.logging.debug(f"Processing jet systematic: {jet_syst}")
                        # Apply jet corrections and show leading changes for quick inspection

                # corr_fatjets = GetJetCorrections(
                #     events0.FatJet,
                #     events0,
                #     era=dataset,
                #     IOV=IOV,
                #     isData=not self._do_gen,
                #     mode='AK8',
                # )
                
                # fatjet_pt_after = ak.to_numpy(ak.flatten(corr_fatjets.pt, axis=None))
                # self.logging.debug(f"FatJet pt before correction: {fatjet_pt_before[:5]}" )
                # self.logging.debug(f"FatJet pt after correction: {fatjet_pt_after[:5]}" )
                

                if jet_syst == "nominal":
                    events_j = ak.with_field(events0, corr_jets, "FatJet")
                elif jet_syst == "JERUp":
                    events_j = ak.with_field(events0, corr_jets.JER.up, "FatJet")
                elif jet_syst == "JERDown":
                    events_j = ak.with_field(events0, corr_jets.JER.down, "FatJet")

                presort_recojets = events_j.FatJet[
                    (events_j.FatJet.mass > 0) 
                    &(events_j.FatJet.pt > 0) 
                    &(np.abs(events_j.FatJet.eta) < 2.5) 
                    &(events_j.FatJet.jetId == 6) 
                ]

                ## Sorting jets by pt
                index = ak.argsort(presort_recojets.pt, ascending=False)
                
                recojets  = presort_recojets[index]
                #recojets = presort_recojets
                events_j = ak.with_field(events_j, recojets, "FatJet")


                recojets = events_j.FatJet

                sel.add("oneRecoJet", 
                            ak.sum( (events_j.FatJet.pt > 0) & (np.abs(events_j.FatJet.eta) < 2.5)  & (events_j.FatJet.jetId == 6), axis=1 ) >= 1
                        )




                reco_jet, z_jet_dphi_reco = get_dphi( z_reco, recojets )
            
            
                reco_jet = ak.firsts(recojets)

            
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
                if self._do_gen:
                    is_matched_reco = reco_jet.delta_r(gen_jet) < 0.4
                    sel.add("is_matched_reco", is_matched_reco)

                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco", "is_matched_reco" )
                    sel.add("allsel_reco", allsel_reco)
                    is_matched_gen = gen_jet.delta_r(reco_jet) < 0.4
                    sel.add("is_matched_gen", is_matched_gen)
                    allsel_gen = sel.all("kinsel_gen", "toposel_gen" , "is_matched_gen" )
                    sel.add("allsel_gen", allsel_gen)
                    sel.add("fakes", sel.require(allsel_reco = True, allsel_gen = False))
                else:
                    allsel_reco = sel.all("npv", "MET", "kinsel_reco", "toposel_reco" )
                    sel.add("allsel_reco", allsel_reco)

            




                if (self._do_gen) and (jet_syst == "nominal"):

                    self.logging.debug("Padded Electron/Muon collections to minimum length 2 per event")
                    add_lepton_weights(events_j, twoReco_ee_sel, twoReco_mm_sel, weights, IOV)  

                    self.logging.info("Lepton weights added")      
                sel_reco = sel.require(allsel_reco = True)
                if jet_syst == "nominal":
                    fill_hist(self.hists, "ptz_mz_reco", dataset =  dataset, mass = z_reco[sel_reco].mass, pt = z_reco[sel_reco].pt, weight = weights.weight()[sel_reco]) 

                # Fill histograms
                # GEN level histograms
                channels = ['ee', 'mm']
                self.logging.debug(f"Total reco events passing all selection: {sel.require(allsel_reco = True).sum()}",  )
                self.logging.debug(f"Total reco events (ee channel) passing all selection: {sel.require(allsel_reco = True, twoReco_ee = True).sum()}",  )
                self.logging.debug(f"Total reco events (mm channel) passing all selection: {sel.require(allsel_reco = True, twoReco_mm = True).sum()}",  )
                self.logging.debug(f"Weights sample: {weights.weight()[sel_reco][:10]}" )
                for channel in channels:
                    if channel == 'ee':
                        if self._do_gen:
                            sel_both = sel.require(allsel_reco = True, allsel_gen = True, twoGen_ee = True, twoReco_ee = True)
                            sel_gen = sel.require(allsel_gen = True, twoGen_ee = True)
                        sel_reco = sel.require(allsel_reco = True, twoReco_ee = True)
                    else:
                        if self._do_gen:
                            sel_both = sel.require(allsel_reco = True, allsel_gen = True, twoGen_mm = True, twoReco_mm = True)
                            sel_gen = sel.require(allsel_gen = True, twoGen_mm = True)
                        sel_reco = sel.require(allsel_reco = True, twoReco_mm = True)

                    if jet_syst == "nominal":
                        for syst in self.systematics:
                            if syst == "nominal":
                                if self._do_gen:
                                    weights_gen =  weights.partial_weight('genWeight')[sel_gen]
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

                                #self.logging.debug(f"No of GEN JET {len(mgen)}")           

                                fill_hist(self.hists, "ptjet_mjet_u_gen", dataset = dataset, channel = channel, ptgen = ptgen, mgen = mgen, weight = weights_gen, systematic = syst)
                                fill_hist(self.hists, "ptjet_mjet_g_gen", dataset = dataset, channel = channel, ptgen = ptgen, mgen = mgen_g, weight = weights_gen, systematic = syst)
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
                                fill_hist(self.hists, "response_matrix_u", dataset = dataset, channel = channel, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, systematic = syst, weight = weights_both)
                                ptreco_both_g = reco_jet_both.pt
                                ptreco_both_g = ptreco_both_g[~ak.is_none(ptreco_both_g)]
                                mreco_both_g = reco_jet_both.msoftdrop
                                mreco_both_g = mreco_both_g[~ak.is_none(mreco_both_g)]
                                fill_hist(self.hists, "response_matrix_g", dataset = dataset, channel = channel, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, systematic = syst, weight = weights_both)

                
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
                            
                    
                            #self.logging.debug(f"No of RECO JET {len(mreco)}")
                            if syst == "nominal":
                                self.logging.debug(f"Len of ptreco {len(ptreco)} mreco {len(mreco)} syst {syst} channel {channel} dataset {dataset}")
                                self.logging.debug(f"ptreco sample {ptreco[:10]}")
                                self.logging.debug(f"mreco sample {mreco[:10]}")
                                self.logging.debug(f"mreco_g sample {mreco_g[:10]}")
                            fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = syst, weight = weights_reco)
                            fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = syst, weight = weights_reco_g)
                            if not self._do_gen:
                                break # Break on nominal when running over data




                    else: # jet syst is not nominal
                        if self._do_gen:
                            weights_gen =  weights.partial_weight(include=['genWeight'])[sel_gen]
                            weights_both = weights.weight()[sel_both]
                        weights_reco = weights.weight()[sel_reco]
                        

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
                        

                        #self.logging.debug(f"No of RECO JET {len(mreco)}")
                            
                        fill_hist(self.hists, "ptjet_mjet_u_reco", dataset = dataset, channel = channel, ptreco = ptreco, mreco = mreco, systematic = jet_syst, weight = weights_reco)
                        fill_hist(self.hists, "ptjet_mjet_g_reco", dataset = dataset, channel = channel, ptreco = ptreco_g, mreco = mreco_g, systematic = jet_syst, weight = weights_reco_g)

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
                            
                            fill_hist(self.hists, "response_matrix_u", dataset = dataset, channel = channel, ptreco = ptreco_both, mreco = mreco_both, ptgen = ptgen_both, mgen = mgen_both, systematic = jet_syst, weight = weights_both)
                            ptreco_both_g = reco_jet_both.pt
                            ptreco_both_g = ptreco_both_g[~ak.is_none(ptreco_both_g)]
                            mreco_both_g = reco_jet_both.msoftdrop
                            mreco_both_g = mreco_both_g[~ak.is_none(mreco_both_g)]
                            weights_both_g = weights_both[~ak.is_none(mreco_both_g)]
                            fill_hist(self.hists, "response_matrix_g", dataset = dataset, channel = channel, ptreco = ptreco_both_g, mreco = mreco_both_g, ptgen = ptgen_both, mgen = mgen_both_g, systematic = jet_syst, weight = weights_both_g)
                            # End of channels loop
                if not self._do_gen:
                    break  # Exit the jet_syst loop if not doing GEN analysis

            if not self._do_jk:
                break  # Exit the JK loop if not doing jackknife resampling

        for name in sel.names:
            self.hists["cutflow"][f"{dataset}_{name}"] += sel.all(name).sum()
        

        return self.hists

    def postprocess(self, accumulator):
        hname_list = [key for key in accumulator.keys() if key not in ("cutflow", "nev", "sumw")]

        #hname_list = ["ptjet_mjet_u_reco", 'ptjet_mjet_g_reco', "ptjet_mjet_u_gen", "ptjet_mjet_g_gen", "response_matrix_u", "response_matrix_g"]
        sumw = accumulator["sumw"]
        
        for hname in hname_list:
            
            h = accumulator[hname]
            for i,ds in enumerate(h.axes['dataset']):
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    continue
                elif 'pythia' in ds:
                    xsdb = {
                        'HT-100to200': 139.2,
                        'HT-1200to2500': 0.1305,
                        'HT-200to400': 38.4,
                        'HT-2500toInf': 0.002997,
                        'HT-600to800': 1.258,
                        'HT-400to600': 5.174,
                        'HT-70to100': 140.0	,
                        'HT-800to1200': 0.5598
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
            grouping = defaultdict(list)
            
            for ds in h.axes["dataset"]:
                if ds.startswith("SingleMuon") or ds.startswith("EGamma") or ds.startswith("SingleElectron"):
                    grouping[ds].append(ds)
                    continue
    
                iov = ds.split("_")[-2]
                if "pythia" in ds:
                    new_key = f"pythia_{iov}"
                elif "herwig" in ds:
                    new_key = f"herwig_{iov}"
                else:
                    new_key = f"MC_{iov}"
                grouping[new_key].append(ds)
        
                # 3) Merge with the no-growth workaround (preserves axis order)
            h = group(h, oldname="dataset", newname="dataset", grouping=dict(grouping))
            accumulator[hname] = h

        return accumulator

                    
            
