#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Terminal-runnable version of your notebook script, with interactive confirmation.

Key defaults:
- casa=True, test=True, data=False (MC), dataset="pythia"
- use_dask=False, workers=1, chunksize=400000, maxchunks=1
- out_file="test.pkl"
- mode="minimal"
- systematics/jet_systematics:
    * default: ['nominal'] / ['nominal']
    * --use-systematics => None / None (full systematics)
    * --minimal-systematics => ['nominal'] / ['nominal','JERUp']
    * users may override with --systematics and/or --jet-systematics; explicit overrides win.

Example usage:
  ./run_qjetmass.py --casa --test --dataset pythia
  ./run_qjetmass.py --use-systematics --systematics nominal --jet-systematics nominal,JESUp
  ./run_qjetmass.py --minimal-systematics --full --use-dask
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path

import coffea
from coffea import processor
from coffea.nanoevents import NanoAODSchema


# -------------------------
# Utilities
# -------------------------

def format_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def try_add_src_to_path(extra_paths=None):
    """
    Add likely repo ./src to PYTHONPATH if present.
    Also add any user-provided extra paths.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "src",
        here.parent / "src",
        Path.cwd() / "src",
        Path.cwd().parent / "src",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            sys.path.insert(0, str(c))
            break

    if extra_paths:
        for p in extra_paths:
            sys.path.insert(0, os.path.abspath(p))


def make_runner(
    use_dask: bool = False,
    client=None,
    workers: int = 1,
    chunksize: int = 200_000,
    maxchunks: int | None = 1,
    skipbadfiles: bool = True,
    retries: int = 3,
    treereduction: int = 4,
):
    """
    If use_dask=True, 'client' must be an existing distributed.Client.
    Otherwise falls back to FuturesExecutor(workers=...).
    """
    if use_dask:
        if client is None:
            raise ValueError("use_dask=True but no Dask 'client' was provided.")
        executor = processor.DaskExecutor(
            client=client,
            status=True,
            retries=retries,
            treereduction=treereduction,
        )
    else:
        executor = processor.FuturesExecutor(
            workers=workers,
            status=True,
            compression=None,
        )

    return processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=chunksize,
        maxchunks=maxchunks,
        skipbadfiles=skipbadfiles,
    )


def build_fileset(args):
    """
    Re-implements your fileset logic with minimal semantic changes.
    """
    prependstr = args.prepend

    ht_bins = args.ht_bins.split(",") if args.ht_bins else [
        "100to200", "200to400", "400to600", "600to800",
        "800to1200", "1200to2500", "2500toInf"
    ]

    fileset = {}

    if args.data:
        # Data mode
        if args.casa:
            sample_path = ["SingleMuon_UL2018.txt", "EGamma_UL2018.txt"]
        else:
            if not args.sample_list:
                raise FileNotFoundError(
                    "For --no-casa --data, please provide --sample-list <txt>."
                )
            sample_path = [args.sample_list]

        for path in sample_path:
            sample = Path(path).stem
            file_to_open = Path("samples") / path if args.casa else Path(path)
            with open(file_to_open) as f:
                lines = f.readlines()

            files = [prependstr + line.strip() for line in lines if line.strip()]
            fileset[sample] = files

    else:
        # MC mode
        if not args.casa and args.sample_list:
            sample_path = [args.sample_list]
            for path in sample_path:
                sample = Path(path).stem
                with open(path) as f:
                    lines = f.readlines()
                files = [prependstr + line.strip() for line in lines if line.strip()]
                fileset[sample] = files
            return fileset

        if args.dataset == "pythia":
            sample_path = ["pythia_UL16NanoAODv9.txt"]
            for path in sample_path:
                sample = Path(path).stem
                file_to_open = Path("samples_mc") / path
                with open(file_to_open) as f:
                    lines = f.readlines()

                for ht_bin in ht_bins:
                    files = [
                        prependstr + line.strip()
                        for line in lines
                        if ht_bin in line and line.strip()
                    ]
                    fileset[f"{sample}_HT-{ht_bin}"] = files

        elif args.dataset == "herwig":
            sample_path = [
                "herwig7_UL16NanoAODAPVv9_inclusive.txt",
                "herwig7_UL16NanoAODv9_inclusive.txt",
                "herwig7_UL17NanoAODv9_inclusive.txt",
                "herwig7_UL18NanoAODv9_inclusive.txt",
            ]
            for path in sample_path:
                sample = Path(path).stem
                file_to_open = Path("samples_mc") / path
                with open(file_to_open) as f:
                    lines = f.readlines()

                files = [prependstr + line.strip() for line in lines if line.strip()]
                fileset[sample] = files

        elif args.dataset == "powheg":
            sample_path = ["powheg_UL18NanoAODv9_inclusive.txt"]
            for path in sample_path:
                sample = Path(path).stem
                file_to_open = Path("samples_mc") / path
                with open(file_to_open) as f:
                    lines = f.readlines()

                files = [prependstr + line.strip() for line in lines if line.strip()]
                fileset[sample] = files

        elif args.dataset == "st":
            sample_path = [
                "st_tW_antitop_UL16NanoAODv9.txt",
                "st_tW_antitop_UL16NanoAODAPVv9.txt",
                "st_tW_antitop_UL17NanoAODv9.txt",
                "st_tW_antitop_UL18NanoAODv9.txt",
                "st_tW_top_UL16NanoAODv9.txt",
                "st_tW_top_UL16NanoAODAPVv9.txt",
                "st_tW_top_UL17NanoAODv9.txt",
                "st_tW_top_UL18NanoAODv9.txt",
                "ST_t-channel_antitop_4f_InclusiveDecays_UL16NanoAODv9.txt",
                "ST_t-channel_antitop_4f_InclusiveDecays_UL16NanoAODAPVv9.txt",
                "ST_t-channel_antitop_4f_InclusiveDecays_UL17NanoAODv9.txt",
                "ST_t-channel_antitop_4f_InclusiveDecays_UL18NanoAODv9.txt",
                "ST_t-channel_top_4f_InclusiveDecays_UL16NanoAODv9.txt",
                "ST_t-channel_top_4f_InclusiveDecays_UL16NanoAODAPVv9.txt",
                "ST_t-channel_top_4f_InclusiveDecays_UL17NanoAODv9.txt",
                "ST_t-channel_top_4f_InclusiveDecays_UL18NanoAODv9.txt",
            ]
            for path in sample_path:
                sample = Path(path).stem
                file_to_open = Path("samples_mc") / path
                with open(file_to_open) as f:
                    lines = f.readlines()

                files = [prependstr + line.strip() for line in lines if line.strip()]
                fileset[sample] = files

        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    return fileset


def pick_test_fileset(fileset):
    """
    Mimic your test selection:
      fileset_test = {list(fileset.keys())[2]: [fileset[list(fileset.keys())[3]][3]]}
    If not enough entries, fallback to first file.
    """
    keys = list(fileset.keys())
    if len(keys) >= 4 and len(fileset[keys[3]]) >= 4:
        return {keys[2]: [fileset[keys[3]][3]]}
    else:
        k0 = keys[0]
        return {k0: [fileset[k0][0]]}


def resolve_systematics(args):
    """
    Determine systematics lists based on flags and user overrides.
    Priority:
      1) explicit overrides via --systematics or --jet-systematics
      2) --use-systematics
      3) --minimal-systematics
      4) default nominal-only
    """
    # Defaults based on mode flags
    if args.use_systematics:
        sys_default = None
        jet_default = None
        mode_tag = "full-systematics"
    elif args.minimal_systematics:
        sys_default = ["nominal"]
        jet_default = ["nominal", "JERUp"]
        mode_tag = "minimal-systematics"
    else:
        sys_default = ["nominal"]
        jet_default = ["nominal"]
        mode_tag = "nominal-only"

    # Apply user overrides if provided
    if args.systematics is not None:
        if args.systematics.strip().lower() == "none":
            sys_list = None
        else:
            sys_list = [s.strip() for s in args.systematics.split(",") if s.strip()]
        mode_tag += " + user-override(systematics)"
    else:
        sys_list = sys_default

    if args.jet_systematics is not None:
        if args.jet_systematics.strip().lower() == "none":
            jet_list = None
        else:
            jet_list = [s.strip() for s in args.jet_systematics.split(",") if s.strip()]
        mode_tag += " + user-override(jet_systematics)"
    else:
        jet_list = jet_default

    return sys_list, jet_list, mode_tag


def print_run_config(args, fileset_to_run, systematics, jet_systematics, syst_mode_tag):
    """
    Pretty-print the effective configuration before running.
    """
    print("\n================ RUN CONFIGURATION ================")
    print(f"casa:              {args.casa}")
    print(f"test mode:         {args.test}")
    print(f"data:              {args.data}")
    print(f"dataset:           {args.dataset}")
    print(f"prepend:           {args.prepend}")
    print(f"sample_list:       {args.sample_list}")
    print(f"ht_bins:           {args.ht_bins}")
    print(f"use_dask:          {args.use_dask}")
    print(f"workers:           {args.workers}")
    print(f"chunksize:         {args.chunksize}")
    print(f"maxchunks:         {None if args.maxchunks == 0 else args.maxchunks}")
    print(f"retries:           {args.retries}")
    print(f"treereduction:     {args.treereduction}")
    print(f"mode:              {args.mode}")
    print(f"out_file:          {args.out_file}")
    print(f"syst_mode:         {syst_mode_tag}")
    print(f"systematics:       {systematics}")
    print(f"jet_systematics:   {jet_systematics}")
    print("samples to run:")
    for k, v in fileset_to_run.items():
        print(f"  - {k}: {len(v)} files")
    print("===================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run QJetMassProcessor from terminal.")

    # Environment & dataset flags
    parser.add_argument("--casa", dest="casa", action="store_true", default=True,
                        help="Assume coffea-casa environment & sample lists under samples*/ (default: True).")

    parser.add_argument("--debug", dest="debug", action="store_true", default=False, help="Enables debug mode")
    parser.add_argument("--no-casa", dest="casa", action="store_false",
                        help="Do not assume casa; sample-list should be direct path.")

    parser.add_argument("--test", dest="test", action="store_true", default=True,
                        help="Run on tiny 1–2 file test mode (default: True).")
    parser.add_argument("--full", dest="test", action="store_false",
                        help="Run full fileset.")

    parser.add_argument("--data", dest="data", action="store_true", default=False,
                        help="Run on data.")
    parser.add_argument("--mc", dest="data", action="store_false",
                        help="Run on MC (default).")

    parser.add_argument("--dataset", choices=["pythia", "herwig", "powheg", "st"],
                        default="pythia", help="MC dataset model.")
    parser.add_argument("--prepend", default="root://xcache/",
                        help="Prefix for each file path.")
    parser.add_argument("--sample-list", default=None,
                        help="Direct txt list of files (needed for --no-casa in some modes).")
    parser.add_argument("--ht-bins", default=None,
                        help="Comma-separated HT bins to use for pythia.")

    # Execution flags
    parser.add_argument("--use-dask", action="store_true", default=False,
                        help="Use DaskExecutor instead of FuturesExecutor.")
    parser.add_argument("--workers", type=int, default=1, help="Workers for FuturesExecutor.")
    parser.add_argument("--chunksize", type=int, default=400000, help="Chunk size.")
    parser.add_argument("--maxchunks", type=int, default=1,
                        help="Max chunks. Use 0 for None (all chunks).")
    parser.add_argument("--retries", type=int, default=3, help="Dask retries.")
    parser.add_argument("--treereduction", type=int, default=4, help="Dask treereduction.")
    parser.add_argument("--extra-path", action="append", default=None,
                        help="Extra path(s) to add to PYTHONPATH, can be repeated.")

    # Systematics flags
    parser.add_argument("--use-systematics", action="store_true", default=False,
                        help="Enable full systematics (sets systematics=None, jet_systematics=None unless overridden).")
    parser.add_argument("--minimal-systematics", action="store_true", default=False,
                        help="Minimal systematics: systematics=['nominal'], jet_systematics=['nominal','JERUp'] unless overridden.")
    parser.add_argument("--systematics", default=None,
                        help="Override systematics list (comma-separated) or 'None'. If set, overrides flags.")
    parser.add_argument("--jet-systematics", default=None,
                        help="Override jet_systematics list (comma-separated) or 'None'. If set, overrides flags.")

    # Processor flags
    parser.add_argument("--mode", default="minimal", help="Processor mode. Default: minimal.")
    parser.add_argument("--out-file", default="test.pkl", help="Pickle output to this file (default: test.pkl).")

    args = parser.parse_args()

    # interpret maxchunks 0 as None
    maxchunks = None if args.maxchunks == 0 else args.maxchunks

    try_add_src_to_path(args.extra_path)

    # Imports that need PYTHONPATH set
    import importlib
    import zjet_corrections
    import zjet_corrections.zjet_processor
    importlib.reload(zjet_corrections.zjet_processor)
    from zjet_corrections.zjet_processor import QJetMassProcessor

    NanoAODSchema.warn_missing_crossrefs = False

    # Dask client if requested
    client = None
    if args.use_dask:
        from dask.distributed import Client
        if args.casa:
            from coffea_casa import CoffeaCasaCluster
            cluster = CoffeaCasaCluster(memory="6 GiB")
            cluster.adapt(minimum=0, maximum=100)
            client = Client(cluster)
            print("✅ Dask client created and connected (CoffeaCasaCluster).")
        else:
            client = Client()
            print("✅ Local Dask client created.")

    try:
        # Build fileset
        fileset = build_fileset(args)
        print("Fileset keys:", list(fileset.keys()))

        # Build test fileset if needed
        if args.test:
            fileset_to_run = pick_test_fileset(fileset)
        else:
            fileset_to_run = fileset

        systematics, jet_systematics, syst_mode_tag = resolve_systematics(args)

        # Print config and confirm
        print_run_config(args, fileset_to_run, systematics, jet_systematics, syst_mode_tag)
        ans = input("Proceed with these settings? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted by user. Exiting cleanly.")
            return

        run = make_runner(
            use_dask=args.use_dask,
            client=client,
            workers=args.workers,
            chunksize=args.chunksize,
            maxchunks=maxchunks,
            retries=args.retries,
            treereduction=args.treereduction,
        )

        t0 = time.time()
        out = run(
            fileset_to_run,
            "Events",
            processor_instance=QJetMassProcessor(
                do_gen=not args.data,
                debug=args.debug,
                systematics=systematics,
                jet_systematics=jet_systematics,
                mode=args.mode,
            ),
        )
        t1 = time.time()
        print(f"Done Running, time taken {format_time(t1 - t0)}")

        if args.out_file:
            with open(args.out_file, "wb") as f:
                pickle.dump(out, f)
            print(f"✅ Output saved to {args.out_file}")

    finally:
        if client is not None:
            client.close()
            print("✅ Dask client closed.")


if __name__ == "__main__":
    main()
