import pathlib
import sys

import awkward as ak
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zjet_corrections.zjet_processor import QJetMassProcessor


def test_raw_reco_mass_fields_use_uncorrected_fatjet_values():
    fatjets = ak.Array(
        [
            [
                {"mass": 100.0, "msoftdrop": 80.0, "rawFactor": 0.10},
                {"mass": 50.0, "msoftdrop": 55.0, "rawFactor": 0.20},
            ],
            [{"mass": 40.0, "msoftdrop": 30.0, "rawFactor": 0.00}],
        ]
    )

    out = QJetMassProcessor._with_raw_reco_mass_fields(fatjets)

    assert ak.to_list(out.mass_raw_ungroomed) == [[90.0, 40.0], [40.0]]
    assert ak.to_list(out.mass_raw_groomed) == [[80.0, 55.0], [30.0]]


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
