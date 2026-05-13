"""Streamlit UI for running the Z+jet analysis (replaces run_analysis.ipynb).

Mirrors the widget layer of notebooks/run_analysis.ipynb, persists settings in
the same `.analysis_widget_config.json` so notebook and Streamlit users stay
interchangeable, and launches scripts/run_analysis_cli.py as an unbuffered
subprocess whose stdout is streamed into the page in real time via a
background reader thread + a 200 ms `st.fragment`.

Process-control plumbing (spawn, kill, PID file, stream state machine) lives
in `run_controller.py` so it can be unit-tested without booting Streamlit.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import zjet_corrections.notebook_utils as nbutils  # noqa: E402

# Local import so the controller is reloadable in dev.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_controller as rc  # noqa: E402

CONFIG_FILE = REPO_ROOT / ".analysis_widget_config.json"
CLI_SCRIPT = REPO_ROOT / "scripts" / "run_analysis_cli.py"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PID_FILE = OUTPUTS_DIR / ".streamlit_run.pid"
DEFAULTS = nbutils.ANALYSIS_CONFIG_DEFAULTS

# Files larger than this still appear in the browser but show no in-page
# download button (browsers + Streamlit start straining around hundreds of
# MB; copy from disk instead).
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024


# -------------------- Config I/O --------------------
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return nbutils.validate_analysis_config(json.loads(CONFIG_FILE.read_text()))
        except Exception as exc:
            st.warning(f"Could not read {CONFIG_FILE.name}: {exc}. Falling back to defaults.")
    return nbutils.validate_analysis_config({})


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def widget_values() -> dict:
    s = st.session_state
    return {
        "casa": s.casa,
        "test": s.test,
        "useDefault": s.useDefault,
        "executor_mode": s.executor_mode,
        "mode": s.mode,
        "era": s.era,
        "dataset": s.dataset,
        "chunksize": int(s.chunksize),
        "chunksize_test": int(s.chunksize_test),
        "group_mode": s.group_mode,
        "redirector": s.redirector,
        "prependstr": nbutils.resolve_redirector_prepend(s.redirector),
        "systematic_profile": s.systematic_profile,
    }


# -------------------- Launch / state --------------------
def launch_run(cfg: dict) -> None:
    save_config(cfg)
    cmd = [sys.executable, "-u", str(CLI_SCRIPT), "--config", str(CONFIG_FILE)]
    proc, pgid, master_fd = rc.spawn(cmd, cwd=REPO_ROOT)
    rc.write_pid_file(PID_FILE, proc.pid, pgid)
    state = rc.new_state()
    rc.attach_reader(state, proc, pgid, master_fd)
    st.session_state.run = state


def has_active_run() -> bool:
    state = st.session_state.get("run")
    if state is None:
        return False
    proc = state.get("proc")
    return proc is not None and proc.poll() is None


# -------------------- Live log fragment (auto-reruns every 200 ms) --------------------
@st.fragment(run_every="200ms")
def live_log_fragment() -> None:
    state = st.session_state.get("run")
    if state is None:
        st.info("No run in progress.")
        return

    rc.drain_queue(state)
    proc = state["proc"]
    returncode = proc.poll()

    elapsed = time.time() - (state["started_at"] or time.time())
    cols = st.columns([2, 1, 1])
    if returncode is None:
        cols[0].info(f"Running — pid {proc.pid}, pgid {state['pgid']}, elapsed {elapsed:.1f}s")
        if cols[1].button("Stop", key="stop_run", type="secondary"):
            rc.stop_run(proc, state["pgid"])
        if cols[2].button("Force kill", key="kill_run"):
            rc.kill_pgid(state["pgid"], hard=True)
    else:
        if not state["finished"]:
            # Drain any trailing bytes after process exit.
            rc.drain_queue(state)
            if state["current"]:
                # Flush a trailing line with no newline.
                state["committed"].append(state["current"])
                state["current"] = ""
            state["finished"] = True
            state["returncode"] = returncode
            rc.clear_pid_file(PID_FILE)
            rc.close_master_fd(state)
        if returncode == 0:
            cols[0].success(f"Finished cleanly (exit 0, elapsed {elapsed:.1f}s).")
        else:
            cols[0].error(f"Exited with code {returncode} (elapsed {elapsed:.1f}s).")
        if cols[1].button("Clear log", key="clear_log"):
            rc.close_master_fd(state)
            st.session_state.pop("run", None)
            st.rerun()

    # Live progress panel: one bar per label, only ever the latest frame.
    if state["progress"]:
        st.markdown("**Progress**")
        for label, info in state["progress"].items():
            st.progress(min(info["percent"] / 100.0, 1.0), text=info["raw"])

    # Log box: only non-progress lines + the current in-progress line.
    visible = state["committed"][-500:]
    if state["current"] and rc.parse_progress_line(state["current"]) is None:
        visible = visible + [state["current"]]
    st.code("\n".join(visible) or "(no output yet)")

    if state["finished"] and state["output_files"]:
        st.subheader("Output files")
        for path in state["output_files"]:
            st.code(path)


# -------------------- Output browser --------------------
def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("kB", "MB", "GB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def _fmt_mtime(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


@st.cache_data(show_spinner=False)
def _read_output_bytes(path_str: str, mtime: float, size: int) -> bytes:
    # Cache keyed on (path, mtime, size) so the file is only read once until
    # the file is rewritten by a new run.
    del mtime, size  # used only for cache invalidation
    return Path(path_str).read_bytes()


def _scan_outputs() -> list[tuple[Path, int, float]]:
    if not OUTPUTS_DIR.exists():
        return []
    items: list[tuple[Path, int, float]] = []
    for p in OUTPUTS_DIR.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        items.append((p, stat.st_size, stat.st_mtime))
    items.sort(key=lambda row: row[2], reverse=True)
    return items


def render_output_browser() -> None:
    st.subheader("Output files")
    header_cols = st.columns([4, 1])
    if header_cols[1].button("Refresh", key="refresh_outputs", use_container_width=True):
        st.rerun()

    files = _scan_outputs()
    if not files:
        st.info(f"No files in `{OUTPUTS_DIR.relative_to(REPO_ROOT)}/` yet.")
        return

    header_cols[0].caption(
        f"{len(files)} file(s) in `{OUTPUTS_DIR.relative_to(REPO_ROOT)}/`, newest first."
    )

    for path, size, mtime in files:
        rel = path.relative_to(OUTPUTS_DIR)
        row = st.columns([5, 1, 2, 2])
        row[0].markdown(f"`{rel}`")
        row[1].text(_fmt_size(size))
        row[2].text(_fmt_mtime(mtime))
        if size > MAX_DOWNLOAD_BYTES:
            row[3].caption("Too large to download in-app")
        else:
            row[3].download_button(
                "Download",
                data=_read_output_bytes(str(path), mtime, size),
                file_name=path.name,
                mime="application/octet-stream",
                key=f"dl_{path}",
                use_container_width=True,
            )


# -------------------- Orphan detection (boot-time) --------------------
def render_orphan_banner() -> None:
    if has_active_run():
        return
    entry = rc.read_pid_file(PID_FILE)
    if entry is None:
        return
    pid, pgid = entry
    if not rc.pid_is_alive(pid):
        rc.clear_pid_file(PID_FILE)
        return
    if not rc.pid_cmdline_contains(pid, "run_analysis_cli.py"):
        # Stale entry from a different process that recycled the PID.
        rc.clear_pid_file(PID_FILE)
        return
    with st.container(border=True):
        st.warning(
            f"Found a leftover analysis run from a previous Streamlit session: "
            f"pid={pid}, pgid={pgid}. It is still alive and may be writing to outputs/."
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Stop leftover (SIGTERM)", key="orphan_stop"):
            rc.kill_pgid(pgid, hard=False)
            time.sleep(1.0)
            if not rc.pid_is_alive(pid):
                rc.clear_pid_file(PID_FILE)
                st.rerun()
        if col_b.button("Force kill leftover (SIGKILL)", key="orphan_kill"):
            rc.kill_pgid(pgid, hard=True)
            time.sleep(0.5)
            if not rc.pid_is_alive(pid):
                rc.clear_pid_file(PID_FILE)
                st.rerun()


# -------------------- Page setup --------------------
st.set_page_config(page_title="Z+jet analysis runner", layout="wide")
st.title("Z+jet analysis runner")

if "config" not in st.session_state:
    st.session_state.config = load_config()
    for k, v in st.session_state.config.items():
        st.session_state.setdefault(k, v)

cfg = st.session_state.config

# -------------------- Sidebar form --------------------
with st.sidebar:
    st.header("Configuration")

    st.checkbox("casa", key="casa")
    st.checkbox("test", key="test")
    st.checkbox("useDefault", key="useDefault")

    # Widgets read their value from st.session_state[key], which is seeded
    # on first run via load_config(). Don't also pass index=/value= or
    # Streamlit warns about "default + Session State API" collisions.
    st.selectbox("executor_mode", nbutils.EXECUTOR_MODE_OPTIONS, key="executor_mode")
    st.selectbox("mode", nbutils.MODE_OPTIONS, key="mode")
    st.selectbox("era", nbutils.ERA_OPTIONS, key="era")
    st.selectbox("dataset", nbutils.DATASET_OPTIONS, key="dataset")
    st.selectbox("systematic_profile", nbutils.SYSTEMATIC_PROFILE_OPTIONS, key="systematic_profile")
    st.number_input("chunksize", min_value=1, step=1000, key="chunksize")
    st.number_input("chunksize_test", min_value=1, step=1000, key="chunksize_test")
    st.selectbox("group_mode", ["all_in_one", "per_group"], key="group_mode")
    st.selectbox("redirector", nbutils.REDIRECTOR_OPTIONS, key="redirector")

    col_a, col_b = st.columns(2)
    save_clicked = col_a.button("Save settings", use_container_width=True)
    reset_clicked = col_b.button("Reset to defaults", use_container_width=True)

if reset_clicked:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.config = nbutils.validate_analysis_config(DEFAULTS)
    save_config(st.session_state.config)
    st.rerun()

if save_clicked:
    st.session_state.config = widget_values()
    save_config(st.session_state.config)
    st.success(f"Saved to {CONFIG_FILE.name}")

# -------------------- Main pane --------------------
render_orphan_banner()

tab_run, tab_outputs = st.tabs(["Run", "Outputs"])

with tab_run:
    current_cfg = widget_values()
    st.subheader("Current settings")
    st.json(current_cfg, expanded=False)

    run_disabled = has_active_run()
    run_clicked = st.button(
        "Run analysis", type="primary",
        disabled=run_disabled,
        help="A run is already in progress." if run_disabled else None,
    )

    if run_clicked:
        launch_run(current_cfg)
        st.rerun()

    st.subheader("Live log")
    live_log_fragment()

with tab_outputs:
    render_output_browser()
