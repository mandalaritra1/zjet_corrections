import inspect
import pathlib
import sys

import awkward as ak
import numpy as np
from coffea import processor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zjet_corrections.zjet_processor import QJetMassProcessor


def test_reco_mass_diagnostic_fields_keep_nanoaod_and_raw_ungroomed_values():
    fatjets = ak.Array(
        [
            [
                {"pt": 200.0, "mass": 100.0, "msoftdrop": 80.0, "rawFactor": 0.10},
                {"pt": 120.0, "mass": 50.0, "msoftdrop": 55.0, "rawFactor": 0.20},
            ],
            [{"pt": 90.0, "mass": 40.0, "msoftdrop": 30.0, "rawFactor": 0.00}],
        ]
    )

    out = QJetMassProcessor._with_reco_mass_diagnostic_fields(fatjets)

    assert ak.to_list(out.pt_nanoaod) == [[200.0, 120.0], [90.0]]
    assert ak.to_list(out.mass_nanoaod) == [[100.0, 50.0], [40.0]]
    assert ak.to_list(out.msoftdrop_nanoaod) == [[80.0, 55.0], [30.0]]
    assert ak.to_list(out.pt_raw_diagnostic) == [[180.0, 96.0], [90.0]]
    assert ak.to_list(out.mass_raw_diagnostic) == [[90.0, 40.0], [40.0]]
    assert ak.to_list(out.msoftdrop_raw_fatjet_diagnostic) == [[72.0, 44.0], [30.0]]


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


def test_ak8_scaled_softdrop_mass_uses_ungroomed_mass_scale():
    fatjets = ak.Array(
        [
            [
                {"mass": 100.0, "mass_raw_diagnostic": 80.0, "msoftdrop_raw_diagnostic": 40.0},
                {"mass": 120.0, "mass_raw_diagnostic": 0.0, "msoftdrop_raw_diagnostic": 30.0},
                {"mass": np.inf, "mass_raw_diagnostic": 60.0, "msoftdrop_raw_diagnostic": 30.0},
            ],
            [
                {"mass": 45.0, "mass_raw_diagnostic": 30.0, "msoftdrop_raw_diagnostic": None},
            ],
        ]
    )

    out = QJetMassProcessor._with_ak8_scaled_softdrop_mass(fatjets)
    values = ak.to_numpy(ak.flatten(out.msoftdrop, axis=None))

    assert np.isclose(values[0], 50.0)
    assert np.isnan(values[1])
    assert np.isnan(values[2])
    assert np.isnan(values[3])


def test_ak8_scaled_softdrop_mass_preserves_raw_groomed_ungroomed_ratio():
    fatjets = ak.Array(
        [
            [
                {"mass": 100.0, "mass_raw_diagnostic": 80.0, "msoftdrop_raw_diagnostic": 40.0},
                {"mass": 60.0, "mass_raw_diagnostic": 40.0, "msoftdrop_raw_diagnostic": 20.0},
            ]
        ]
    )

    out = QJetMassProcessor._with_ak8_scaled_softdrop_mass(fatjets)
    corrected_ratio = ak.to_numpy(ak.flatten(out.msoftdrop / out.mass, axis=None))
    raw_ratio = ak.to_numpy(
        ak.flatten(out.msoftdrop_raw_diagnostic / out.mass_raw_diagnostic, axis=None)
    )

    assert np.allclose(corrected_ratio, raw_ratio)


def test_ak8_scaled_softdrop_mass_can_use_nominal_raw_reference():
    varied_fatjets = ak.Array([[{"mass": 110.0}]])
    nominal_fatjets = ak.Array(
        [[{"mass_raw_diagnostic": 80.0, "msoftdrop_raw_diagnostic": 40.0}]]
    )

    out = QJetMassProcessor._with_ak8_scaled_softdrop_mass(varied_fatjets, nominal_fatjets)

    assert np.isclose(ak.to_numpy(out.msoftdrop)[0, 0], 55.0)


def test_jer_down_uses_down_varied_ak8_mass_scale():
    source = inspect.getsource(QJetMassProcessor.process)
    jer_down_block = source.split('elif jet_syst == "JERDown":', 1)[1].split(
        'elif jet_syst == "JMSUp":',
        1,
    )[0]

    assert "corr_jets.JER.down" in jer_down_block
    assert "corr_jets.JER.up" not in jer_down_block


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
    assert {
        "mass_raw",
        "msoftdrop_raw",
        "msoftdrop_raw_fatjet",
        "mass_nanoaod",
        "msoftdrop_nanoaod",
    } <= set(proc.hists["reco_jet_ntuple"])


def test_mass_diagnostic_ntuple_postprocess_skips_ntuple_accumulator():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="mass_diagnostic_ntuple",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )

    out = proc.postprocess(proc.hists)

    assert out["reco_jet_ntuple"] is proc.hists["reco_jet_ntuple"]


def test_mass_diagnostic_ntuple_postprocess_scales_mc_weights():
    proc = QJetMassProcessor(
        do_gen=True,
        mode="mass_diagnostic_ntuple",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )
    dataset = "pythia_UL16NanoAODv9_HT-100to200"
    ntuple = proc.hists["reco_jet_ntuple"]
    ntuple["dataset"] += processor.column_accumulator(np.array([dataset, dataset], dtype=object))
    ntuple["weight"] += processor.column_accumulator(np.array([1.0, 2.0], dtype=np.float64))
    proc.hists["sumw"][dataset] = 1000.0

    out = proc.postprocess(proc.hists)

    expected_scale = 139.2 * 19.52 * 1000.0 * 1.1297638966 / 1000.0
    assert np.allclose(out["reco_jet_ntuple"]["weight"].value, [expected_scale, 2.0 * expected_scale])


def test_mass_diagnostic_ntuple_postprocess_keeps_data_weights_unscaled():
    proc = QJetMassProcessor(
        do_gen=False,
        mode="mass_diagnostic_ntuple",
        jet_systematics=["nominal"],
        systematics=["nominal"],
    )
    ntuple = proc.hists["reco_jet_ntuple"]
    ntuple["dataset"] += processor.column_accumulator(
        np.array(["SingleMuon_UL2016", "SingleElectron_UL2016"], dtype=object)
    )
    ntuple["weight"] += processor.column_accumulator(np.array([1.0, 1.0], dtype=np.float64))

    out = proc.postprocess(proc.hists)

    assert np.allclose(out["reco_jet_ntuple"]["weight"].value, [1.0, 1.0])


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
