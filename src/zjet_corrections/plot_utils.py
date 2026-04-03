import pickle as pkl
import matplotlib.pyplot as plt
import hist
from . import hep_plot as hplot
import matplotlib.patches as patches

HT_BINS = [
    "HT-100to200",
    "HT-200to400",
    "HT-400to600",
    "HT-600to800",
    "HT-800to1200",
    "HT-1200to2500",
    "HT-2500toInf",
]

ERAS = ["2016APV", "2016", "2017", "2018"]

datasets = {
    "2016": [
        "SingleElectron_UL2016",
        "SingleMuon_UL2016",
    ],
    "2016APV": [
        "SingleElectron_UL2016APV",
        "SingleMuon_UL2016APV",
    ],
    "2017": [
        "SingleElectron_UL2017",
        "SingleMuon_UL2017",
    ],
    "2018": [
        "SingleMuon_UL2018",
        "EGamma_UL2018",
    ],
}

def load_out(era, template="outputs/ht_validation_pythia_{era}.pkl"):
    with open(template.format(era=era), "rb") as f:
        return pkl.load(f)


def plot_ht(era, template="outputs/ht_validation_pythia_{era}.pkl"):
    out = load_out(era, template)

    hplot.setup(era=era)
    hplot.quick_label(
        data=False,
        xlabel=r"$H_T$ [GeV]",
        ylabel=r"#Events",
    )
    hplot.set_plot_name("ht")

    out["ht"].project("ht_bin", "pt")[HT_BINS, hist.loc(0):].plot(
        stack=False,
        histtype="fill",
    )

    plt.yscale("log")
    plt.xlim(100, 4000)
    plt.legend()
    hplot.show()


def plot_ht_all_eras(template="outputs/ht_validation_pythia_{era}.pkl"):
    for era in ERAS:
        plt.figure()
        plot_ht(era, template=template)

def plot_etaphijet_data(era, template = "outputs/etaphijet_validation_data.pkl"):
    with open(template, "rb") as f:
        out = pkl.load(f)

    hplot.setup(era = era)
    hplot.set_plot_name("etaphi_jet0")
    hplot.quick_label(data = True)




    out["eta_phi_jet_reco"][datasets[era], ...].project('eta', 'phi').plot2d(norm = 'log')

    if era == '2018':
        # HEM veto region (approximate CMS convention)
        hem_eta_min, hem_eta_max = -3.0, -1.3
        hem_phi_min, hem_phi_max = -1.57, -0.87
        
        ax = plt.gca()
        
        # draw rectangle around HEM veto area
        hem_box = patches.Rectangle(
            (hem_eta_min, hem_phi_min),                  # bottom-left corner
            hem_eta_max - hem_eta_min,                   # width in eta
            hem_phi_max - hem_phi_min,                   # height in phi
            linewidth=2.5,
            edgecolor='red',
            facecolor='none',
            linestyle='--'
        )
        
        ax.add_patch(hem_box)
    plt.ylim(-3.2, 3.2)
    cbar = plt.gcf().axes[-1]   # last axis is the colorbar
    cbar.set_ylabel("# Events")
    hplot.show()


def plot_etaphijet_data_all(template="outputs/etaphijet_validation_data.pkl"):
    for era in ERAS:
        try:
            print(f"Plotting era {era}")
            plt.figure()
            plot_etaphijet_data(era, template=template)
        except Exception as e:
            print(f"Skipping {era}: {e}")


def plot_etaphijet_mc(era, template = "outputs/etaphijet_validation_pythia_{era}.pkl" ):
    out = load_out(era, template)
    hplot.setup(era = era)
    hplot.set_plot_name("etaphi_jet0_mc")
    hplot.quick_label(data = False)

    out["eta_phi_jet_reco"].project('eta', 'phi').plot2d(norm = 'log')
    plt.ylim(-3.2, 3.2)
    cbar = plt.gcf().axes[-1]   # last axis is the colorbar
    cbar.set_ylabel("# Events")
    hplot.show()

def plot_etaphijet_mc_all(template = "outputs/etaphijet_validation_pythia_{era}.pkl" ):
    for era in ERAS:
        try:
            print(f"Plotting era {era}")
            plt.figure()
            plot_etaphijet_mc(era, template=template)
        except Exception as e:
            print(f"Skipping {era}: {e}")
    
        
            
        