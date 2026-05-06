import pathlib
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zjet_corrections.zjet_processor import QJetMassProcessor


def test_reco_mass_diagnostic_fields_keep_nanoaod_and_raw_ungroomed_values():
    fatjets = ak.Array(
        [
            [
                {"mass": 100.0, "msoftdrop": 80.0, "rawFactor": 0.10},
                {"mass": 50.0, "msoftdrop": 55.0, "rawFactor": 0.20},
            ],
            [{"mass": 40.0, "msoftdrop": 30.0, "rawFactor": 0.00}],
        ]
    )

    out = QJetMassProcessor._with_reco_mass_diagnostic_fields(fatjets)

    assert ak.to_list(out.mass_nanoaod) == [[100.0, 50.0], [40.0]]
    assert ak.to_list(out.msoftdrop_nanoaod) == [[80.0, 55.0], [30.0]]
    assert ak.to_list(out.mass_raw_diagnostic) == [[90.0, 40.0], [40.0]]


def test_raw_softdrop_mass_is_built_from_raw_subjets():
    fatjets = ak.Array(
        [
            [
                {
                    "subJetIdx1": 0,
                    "subJetIdx2": 1,
                    "pt": 100.0,
                    "eta": 0.0,
                    "phi": 0.0,
                    "mass": 50.0,
                    "msoftdrop": 40.0,
                    "rawFactor": 0.0,
                }
            ]
        ]
    )
    subjets = ak.Array(
        [
            [
                {"pt": 50.0, "eta": 0.0, "phi": 0.0, "mass": 10.0, "rawFactor": 0.10},
                {"pt": 50.0, "eta": 0.0, "phi": 0.0, "mass": 20.0, "rawFactor": 0.20},
            ]
        ]
    )

    out = QJetMassProcessor._with_raw_softdrop_mass_from_subjets(fatjets, subjets)

    expected_mass = np.sqrt((np.sqrt(45.0**2 + 9.0**2) + np.sqrt(40.0**2 + 16.0**2)) ** 2 - 85.0**2)
    assert np.isclose(ak.to_numpy(out.msoftdrop_raw_diagnostic)[0, 0], expected_mass)


def test_raw_mass_diagnostics_registered_for_mass_modes():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="minimal",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )

    assert "m_g_over_m_u_raw_reco" in proc.hists
    assert "m_g_vs_m_u_raw_reco" in proc.hists
    assert proc.hists["m_g_vs_m_u_raw_reco"].axes.name == (
        "dataset",
        "channel",
        "m_u_reco",
        "m_g_reco",
        "systematic",
    )


def test_raw_mass_diagnostics_registered_for_mass_jk_mode():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="mass_jk",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )

    assert proc.hists["m_g_vs_m_u_raw_reco"].axes.name == (
        "dataset",
        "channel",
        "m_u_reco",
        "m_g_reco",
        "jk",
        "systematic",
    )


def test_mass_diagnostic_ntuple_mode_books_only_ntuple_and_metadata():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="mass_diagnostic_ntuple",
        jet_systematics=["nominal", "JERUp"],
        systematics=["nominal", "puUp"],
    )

    assert set(proc.hists) == {"reco_jet_ntuple", "sumw", "nev", "cutflow"}
    assert proc.jet_systematics == ["nominal"]
    assert proc.systematics == ["nominal"]
    assert {"mass_raw", "msoftdrop_raw", "mass_nanoaod", "msoftdrop_nanoaod"} <= set(
        proc.hists["reco_jet_ntuple"]
    )


def test_mass_diagnostic_ntuple_postprocess_skips_ntuple_accumulator():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="mass_diagnostic_ntuple",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )

    out = proc.postprocess(proc.hists)

    assert out["reco_jet_ntuple"] is proc.hists["reco_jet_ntuple"]


def test_raw_mass_ratio_fill_rejects_none_zero_and_nonfinite_values():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="minimal",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )

    proc._fill_groomed_over_ungroomed_reco(
        dataset="sample",
        channel="mm",
        ptreco=ak.Array([250.0, 260.0, 270.0, 280.0, 290.0]),
        ungroomed_mass=ak.Array([100.0, 0.0, None, np.inf, 50.0]),
        groomed_mass=ak.Array([80.0, 10.0, 20.0, 30.0, 75.0]),
        weight=ak.Array([1.0, 2.0, 3.0, 4.0, 5.0]),
        systematic="nominal",
        mass_ratio_name="m_g_over_m_u_raw_reco",
        mass_2d_name="m_g_vs_m_u_raw_reco",
    )

    ratio_total = proc.hists["m_g_over_m_u_raw_reco"].values().sum()
    map_total = proc.hists["m_g_vs_m_u_raw_reco"].values().sum()

    assert ratio_total == 6.0
    assert map_total == 6.0
