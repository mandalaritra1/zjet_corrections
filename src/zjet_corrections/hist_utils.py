# src/zjet_corrections/hist_utils.py



def fill_hist(hdict, name, **kwargs):
    """
    Safely fills a histogram only if it exists in the dictionary.
    """
    if name in hdict:
        hdict[name].fill(**kwargs)


def register_hist(hdict, name, axes, label="Counts"):
    """
    Registers a histogram if not already present in the dictionary.
    """
    from hist import Hist
    if name not in hdict:
        hdict[name] = Hist(*axes, storage="weight", label=label)

class util_binning :
    '''
    Class to implement the binning schema for jet mass and pt 2d unfolding. The gen-level mass is twice as fine. 
    '''
    def __init__(self):
        #self.ptreco_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")   
        
        #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 13000], name="mgen", label=r"Mass (GeV)")

        ### Original version
        #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 150, 200, 13000], name="mgen", label=r"Mass (GeV)")
        
        #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 6200, 13000], name="mreco", label=r"$m_{RECO}$ (GeV)")
        ############

        #self.mgen_axis = hist.axis.Variable([0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 200, 13000], name="mgen", label=r"Mass (GeV)")
        #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 , 140, 150, 160, 180, 200, 250, 300, 350, 400, 500, 1000, 13000], name="mreco", label=r"$m_{RECO}$ (GeV)")

        #self.mgen_axis = hist.axis.Regular(100,0,200, name="mgen", label=r"Mass (GeV)")
        #self.mreco_axis = hist.axis.Regular(100,0,200, name="mreco", label=r"$m_{RECO}$")

        ### Only for making the response matrix for Herwig TOY MC
        #self.mgen_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 13000], name="mgen", label=r"Mass (GeV)")

        
        #self.mreco_axis = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")
        # Original bins
        # self.mgen_axis = hist.axis.Variable([0, 5, 10, 20, 40, 60, 80, 100, 120, 140,160, 180, 200, 13000], name="mgen", label=r"Mass (GeV)")
        # self.mreco_axis = hist.axis.Variable([0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 500, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")

        self.mgen_axis = hist.axis.Variable([0, 5, 10, 20, 40, 60, 80, 100, 150, 200, 13000], name="mgen", label=r"Mass (GeV)")
        self.mreco_axis = hist.axis.Variable([0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 500, 13000] , name="mreco", label=r"$m_{RECO}$ (GeV)")

        self.mgen_axis = hist.axis.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                                12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
                                                32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 
                                                52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 
                                                72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 
                                                94, 96, 98, 100, 102, 104, 106, 108, 110, 
                                                112, 114, 116, 118, 120, 122, 124, 126, 
                                                128, 130, 132, 134, 136, 138, 140, 142, 
                                                144, 146, 148, 150, 152, 154, 156, 158, 
                                                160, 162, 164, 166, 168, 170, 172, 174, 
                                                176, 178, 180, 182, 184, 186, 188, 190, 
                                                192, 194, 196, 198, 200, 1000], name="mgen", label=r"Mass (GeV)")

        self.mreco_axis = hist.axis.Variable([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 
                                                    5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 
                                                    10, 11.0, 12, 13.0, 14, 15.0, 16, 17.0, 18, 19.0, 
                                                    20, 21.0, 22, 23.0, 24, 25.0, 26, 27.0, 28, 29.0, 
                                                    30, 31.0, 32, 33.0, 34, 35.0, 36, 37.0, 38, 39.0, 
                                                    40, 41.0, 42, 43.0, 44, 45.0, 46, 47.0, 48, 49.0, 
                                                    50, 51.0, 52, 53.0, 54, 55.0, 56, 57.0, 58, 59.0, 
                                                    60, 61.0, 62, 63.0, 64, 65.0, 66, 67.0, 68, 69.0, 
                                                    70, 71.0, 72, 73.0, 74, 75.0, 76, 77.0, 78.0, 79.0, 80, 81.0, 
                                                    82, 83.0, 84, 85.0, 86, 87.0, 88, 89.0, 90, 91.0, 
                                                    92, 93.0, 94, 95.0, 96, 97.0, 98, 99.0, 100, 101.0, 
                                                    102, 103.0, 104, 105.0, 106, 107.0, 108, 109.0, 110, 111.0, 
                                                    112, 113.0, 114, 115.0, 116, 117.0, 118, 119.0, 120, 121.0, 
                                                    122, 123.0, 124, 125.0, 126, 127.0, 128, 129.0, 130, 131.0, 
                                                    132, 133.0, 134, 135.0, 136, 137.0, 138, 139.0, 140, 141.0, 
                                                    142, 143.0, 144, 145.0, 146, 147.0, 148, 149.0, 150, 151.0, 
                                                    152, 153.0, 154, 155.0, 156, 157.0, 158, 159.0, 160, 161.0, 
                                                    162, 163.0, 164, 165.0, 166, 167.0, 168, 169.0, 170, 171.0, 
                                                    172, 173.0, 174, 175.0, 176, 177.0, 178, 179.0, 180, 181.0, 
                                                    182, 183.0, 184, 185.0, 186, 187.0, 188, 189.0, 190, 191.0, 
                                                    192, 193.0, 194, 195.0, 196, 197.0, 198, 199.0, 200, 500, 1000], name="mreco", label=r"$m_{RECO}$ (GeV)")

        # self.mgen_over_pt_axis = hist.axis.Variable([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10], name = 'mpt_gen', label = r'$\rho$ ')
        # self.mreco_over_pt_axis = hist.axis.Variable([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5 , 7.5, 10], name = 'mpt_reco', label = r'$\rho$ (Detector)')


        # Negative counterpart of mgen_over_pt_axis
        self.mgen_over_pt_axis = hist.axis.Variable(
            [-10, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
            name='mpt_gen', label=r'$-\rho$'
        )
        
        # Negative counterpart of mreco_over_pt_axis
        self.mreco_over_pt_axis = hist.axis.Variable(
            [-10, -7.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3,
             -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
            name='mpt_reco', label=r'$-\rho$ (Detector)'
        )
        #self.ptgen_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptgen", label=r"p_{T,RECO} (GeV)")   

        #self.ptgen_axis = hist.axis.Variable([140, 200, 260, 330, 408, 13000], name="ptgen", label=r"p_{T,GEN} (GeV)")  
        #self.ptreco_axis = hist.axis.Variable([140, 200, 260, 330, 408, 13000], name="ptreco", label=r"p_{T,RECO} (GeV)")

        
        # self.ptgen_axis = hist.axis.Variable([  200.,   260.,   350.,   460., 13000.], name="ptgen", label=r"$p_{T,GEN}$ (GeV)")  
        # self.ptreco_axis = hist.axis.Variable([ 200.,   260.,   350.,   460., 13000.], name="ptreco", label=r"$p_{T,RECO}$ (GeV)")
        self.ptgen_axis = hist.axis.Variable([  200.,   290.,   400.,    13000.], name="ptgen", label=r"$p_{T,GEN}$ (GeV)")  
        self.ptreco_axis = hist.axis.Variable([ 200.,   290.,   400.,    13000.], name="ptreco", label=r"$p_{T,RECO}$ (GeV)")

        self.mcut_reco_u_axis = hist.axis.Variable([ 0, 20, 10000], name="mreco", label=r"Mass (GeV)" )
        self.mcut_reco_g_axis = hist.axis.Variable([ 0, 10, 10000], name="mreco", label=r"Mass (GeV)" )

        self.mcut_gen_u_axis = hist.axis.Variable([ 0, 20, 10000], name="mgen", label=r"Mass (GeV)" )
        self.mcut_gen_g_axis = hist.axis.Variable([ 0, 10, 10000], name="mgen", label=r"Mass (GeV)" )

        self.ptgen_axis_fine = hist.axis.Variable([  200., 210,  220, 230, 240, 260., 280, 300, 320, 340,  350., 370, 390, 410, 430, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 13000.], name="ptgen_fine", label=r"$p_{T,GEN}$ (GeV)")  
        
        
        #self.mgen_axis = hist.axis.Variable( [0,2.5,5,7.5,10,15,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,325,350,1000], name="mgen", label=r"Mass [GeV]")
        
        self.gen_binning = tunfold_binning( self.mgen_axis.edges, self.ptgen_axis.edges, False, False, False, False )
        self.reco_binning = tunfold_binning( self.mreco_axis.edges, self.ptreco_axis.edges, False, False, False, False )
        
        self.gen_axis = hist.axis.Regular(self.gen_binning.total_nbin, 0, self.gen_binning.total_nbin, name = "bin_gen", label = "Generator")
        self.reco_axis =  hist.axis.Regular(self.reco_binning.total_nbin, 0, self.reco_binning.total_nbin, name = "bin_reco", label = "Detector")
        
        self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        self.lep_axis = hist.axis.StrCategory(["ee", "mm"], name="lep")
        self.n_axis = hist.axis.Regular(10, 0, 10, name="n", label=r"Number")
        self.mass_axis = hist.axis.Regular(100, 0, 1000, name="mass", label=r"$m$ [GeV]")
        self.diff_axis = hist.axis.Regular(100, -20, 20, name="diff", label=r"$\Delta$ [GeV]")
        self.diff_axis_large = hist.axis.Regular(100, -50, 50, name="diff", label=r"$\Delta$ [GeV]")
        self.zmass_axis = hist.axis.Regular(100, 80, 100, name="mass", label=r"$m$ [GeV]")
        self.pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")                
        self.frac_axis = hist.axis.Regular(200, -1.5, 1.5, name="frac", label=r"Fraction")                
        self.dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        self.dr_fine_axis = hist.axis.Regular(150, 0, 1.5, name="dr", label=r"$\Delta R$")
        self.dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
        self.eta_axis = hist.axis.Regular(40, -5, 5, name="eta", label=r"$ \eta$")
        self.phi_axis = hist.axis.Regular(40, -5, 5, name="phi", label=r"$ \phi$")
        self.ptfine_axis = hist.axis.Regular(20, 200, 500, name="pt", label=r"p_{T,RECO} (GeV)")
        self.jackknife_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife categories" )
        
        self.syst_axis=hist.axis.StrCategory([],growth = True, name = "systematic", label = "Systematic Uncertainty")