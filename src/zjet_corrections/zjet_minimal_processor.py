"""
ZJetMinimalProcessor
====================
Bare-bones Z+jet processor for diagnosing purity/stability.

NO corrections, NO systematics, NO reweighting.
Goal: produce a 2D response matrix (reco jet mass vs gen jet mass) in a given
jet pT bin using only the minimal cuts needed to select Z+jet events.

If purity/stability look bad here, the problem is in the event selection or
binning — not in the corrections stack.
"""

import awkward as ak
import numpy as np
import hist
from coffea import processor
from coffea.nanoevents import NanoAODSchema


class ZJetMinimalProcessor(processor.ProcessorABC):
    """
    Minimal Z(->mumu)+jet processor.

    Parameters
    ----------
    pt_lo, pt_hi : float
        Reco jet pT range to select (GeV).  Default: 290–400 GeV.
    mass_bins : list[float]
        Bin edges for the jet mass axis (applied to both reco and gen).
    """

    def __init__(
        self,
        pt_lo: float = 290.0,
        pt_hi: float = 400.0,
        mass_bins: list | None = None,
    ):
        self.pt_lo = pt_lo
        self.pt_hi = pt_hi
        self.mass_bins = mass_bins or [
            20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
        ]

    # ------------------------------------------------------------------
    def process(self, events):
        NanoAODSchema.warn_missing_crossrefs = False

        # ---- 1. Muon selection (tight Z→μμ) --------------------------
        mu = events.Muon
        good_mu = mu[
            (mu.pt > 20.0)
            & (np.abs(mu.eta) < 2.4)
            & mu.tightId
            & (mu.pfRelIso04_all < 0.15)
        ]

        # Require ≥2 muons with opposite charge summing to zero
        has_os = (ak.num(good_mu) >= 2) & (ak.sum(good_mu.charge, axis=1) == 0)
        events   = events[has_os]
        good_mu  = good_mu[has_os]

        # Build Z candidate from the two leading muons
        mu1 = good_mu[:, 0]
        mu2 = good_mu[:, 1]
        Z   = mu1 + mu2  # NanoAOD vector add

        # Z mass window: 71–111 GeV
        z_ok    = (Z.mass > 71.0) & (Z.mass < 111.0)
        events  = events[z_ok]
        Z       = Z[z_ok]
        mu1     = mu1[z_ok]
        mu2     = mu2[z_ok]

        # ---- 2. Reco jet selection ------------------------------------
        jets = events.Jet
        good_jets = jets[
            (jets.pt > 30.0)
            & (np.abs(jets.eta) < 2.4)
            & (jets.jetId >= 2)           # tight jet ID
        ]

        # Remove jets overlapping with either selected muon (ΔR > 0.4)
        good_jets = good_jets[
            ak.all(good_jets.metric_table(mu1) > 0.4, axis=2)
            & ak.all(good_jets.metric_table(mu2) > 0.4, axis=2)
        ]

        # Require ≥1 jet; take the leading one
        has_jet  = ak.num(good_jets) >= 1
        events   = events[has_jet]
        Z        = Z[has_jet]
        mu1      = mu1[has_jet]
        mu2      = mu2[has_jet]
        good_jets = good_jets[has_jet]
        lead_jet = good_jets[:, 0]

        # Back-to-back with Z: |ΔΦ(jet, Z)| > 2.7
        dphi_ok  = np.abs(lead_jet.delta_phi(Z)) > 2.7
        events   = events[dphi_ok]
        Z        = Z[dphi_ok]
        lead_jet = lead_jet[dphi_ok]

        # Jet pT bin selection
        pt_ok    = (lead_jet.pt > self.pt_lo) & (lead_jet.pt < self.pt_hi)
        events   = events[pt_ok]
        Z        = Z[pt_ok]
        lead_jet = lead_jet[pt_ok]

        # ---- 3. Histograms -------------------------------------------
        mass_axis_reco = hist.axis.Variable(
            self.mass_bins, name="reco_mass", label="Reco Jet Mass [GeV]"
        )
        mass_axis_gen = hist.axis.Variable(
            self.mass_bins, name="gen_mass", label="Gen Jet Mass [GeV]"
        )

        response_hist = hist.Hist(mass_axis_reco, mass_axis_gen)
        reco_mass_1d  = hist.Hist(mass_axis_reco)
        gen_mass_1d   = hist.Hist(
            hist.axis.Variable(self.mass_bins, name="gen_mass", label="Gen Jet Mass [GeV]")
        )

        is_mc = hasattr(events, "GenJet")

        if is_mc:
            # Use NanoAOD built-in genJetIdx cross-reference
            matched_gen = lead_jet.matched_gen  # None when unmatched
            has_match   = ~ak.is_none(matched_gen)

            matched_reco  = lead_jet[has_match]
            matched_gen_j = matched_gen[has_match]
            w_matched     = np.sign(ak.to_numpy(events.genWeight[has_match]))

            response_hist.fill(
                reco_mass=ak.to_numpy(matched_reco.mass),
                gen_mass=ak.to_numpy(matched_gen_j.mass),
                weight=w_matched,
            )

            w_all = np.sign(ak.to_numpy(events.genWeight))
            reco_mass_1d.fill(
                reco_mass=ak.to_numpy(lead_jet.mass),
                weight=w_all,
            )

            # Gen 1D: only matched jets
            gen_mass_vals = ak.to_numpy(
                ak.fill_none(matched_gen.mass, -999.0)
            )
            gen_mask = ak.to_numpy(~ak.is_none(matched_gen))
            gen_mass_1d.fill(
                gen_mass=gen_mass_vals[gen_mask],
                weight=w_all[gen_mask],
            )

        else:
            # Data: reco only
            reco_mass_1d.fill(reco_mass=ak.to_numpy(lead_jet.mass))

        return {
            "response":  response_hist,
            "reco_mass": reco_mass_1d,
            "gen_mass":  gen_mass_1d,
            "n_events":  len(events),
        }

    def postprocess(self, accumulator):
        return accumulator
