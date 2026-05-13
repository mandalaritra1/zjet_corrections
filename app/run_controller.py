"""Process-control helpers for the Streamlit analysis runner.

Kept Streamlit-free so the launch/stream/stop logic is unit-testable without
booting the page.
"""
from __future__ import annotations

import codecs
import errno
import fcntl
import os
import pty
import queue
import re
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path

# Strip ANSI CSI escape sequences (colors, cursor moves, "clear line", etc.)
# so progress bars from rich/tqdm display as plain text downstream.
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

PROGRESS_BAR_CHAR = "━"  # U+2501 BOX DRAWINGS HEAVY HORIZONTAL — rich/coffea
# Match the label as anything up to the first heavy bar OR the first NN%
# token. Works for both layouts:
#   coffea:        "Preprocessing 100% ━━━ 1/1 [...]"     (% before bar)
#   rich default:  "Processing ━━━ 50% 0:00:02"           (bar before %)
# Label characters explicitly exclude digits, `%`, and the bar glyph so the
# greedy walk stops at the first piece of the bar regardless of order.
PROGRESS_LABEL_RE = re.compile(r"^\s*([^━%\d]+?)\s+(?:\d+%|━)")
PROGRESS_PERCENT_RE = re.compile(r"(\d+)%")


def parse_progress_line(line: str) -> tuple[str, int] | None:
    """Return (label, percent) for a rich/coffea progress line, else None.

    Requires a bar character to avoid false positives on log lines like
    "5% disk free".
    """
    if PROGRESS_BAR_CHAR not in line:
        return None
    label_match = PROGRESS_LABEL_RE.match(line)
    pct_match = PROGRESS_PERCENT_RE.search(line)
    if not label_match or not pct_match:
        return None
    label = label_match.group(1).strip()
    if not label:
        return None
    try:
        return label, int(pct_match.group(1))
    except ValueError:
        return None


# -------------------- PID file --------------------
def write_pid_file(pid_file: Path, pid: int, pgid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}:{pgid}\n")


def read_pid_file(pid_file: Path) -> tuple[int, int] | None:
    if not pid_file.exists():
        return None
    try:
        pid_s, pgid_s = pid_file.read_text().strip().split(":")
        return int(pid_s), int(pgid_s)
    except Exception:
        return None


def clear_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_cmdline_contains(pid: int, needle: str) -> bool:
    """Linux-only best-effort check that /proc/<pid>/cmdline mentions a string."""
    p = Path("/proc") / str(pid) / "cmdline"
    if not p.exists():
        return pid_is_alive(pid)
    try:
        cmdline = p.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return False
    return needle in cmdline


# -------------------- Signalling --------------------
def kill_pgid(pgid: int, *, hard: bool = False) -> None:
    sig = signal.SIGKILL if hard else signal.SIGTERM
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def stop_run(proc: subprocess.Popen, pgid: int, *, timeout: float = 2.0) -> None:
    """SIGTERM the process group, then SIGKILL if it hasn't exited in `timeout`."""
    kill_pgid(pgid, hard=False)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_pgid(pgid, hard=True)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


# -------------------- Python interpreter resolution --------------------
def resolve_cli_python(override: str | None = None) -> tuple[str, str]:
    """Pick the Python interpreter to spawn the analysis CLI with.

    Returns (python_path, source) where `source` documents which mechanism
    selected the interpreter — useful for showing in the UI so the user
    knows whether the autodetect heuristic kicked in.

    Resolution order:
      1. Explicit `override` (e.g. from a UI text box).
      2. `ZJET_CLI_PYTHON` env var.
      3. `$VIRTUAL_ENV/bin/python` if a venv is active.
      4. `$CONDA_PREFIX/bin/python` if a conda env is active.
      5. `sys.executable` (the Python running this Streamlit process).

    The venv/conda-env checks matter on LPC: users frequently activate a
    site-specific env (e.g. `coffea2025`) for `lpcjobqueue` and then pick
    up `streamlit` from a different Python, leaving `sys.executable`
    pointing at the wrong interpreter.
    """
    if override:
        override = override.strip()
        if override:
            return override, "explicit override"

    env_override = os.environ.get("ZJET_CLI_PYTHON", "").strip()
    if env_override:
        return env_override, "ZJET_CLI_PYTHON env var"

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        cand = Path(venv) / "bin" / "python"
        if cand.exists():
            return str(cand), f"VIRTUAL_ENV ({venv})"

    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        cand = Path(conda) / "bin" / "python"
        if cand.exists():
            return str(cand), f"CONDA_PREFIX ({conda})"

    return sys.executable, "sys.executable (Streamlit's Python)"


# -------------------- Spawn --------------------
def spawn(
    cmd: list[str],
    *,
    cwd: Path,
    cols: int = 140,
    rows: int = 40,
) -> tuple[subprocess.Popen, int, int]:
    """Launch `cmd` attached to a pseudo-terminal in its own session.

    Returns (proc, pgid, master_fd). Using a PTY rather than a plain pipe is
    necessary because rich/tqdm (and therefore coffea) detect non-TTY stdout
    and either suppress progress-bar updates entirely or emit only the final
    state. Inside a PTY, the child's stdout.isatty() returns True and bars
    refresh at their normal cadence; we read those updates from `master_fd`.

    PYTHONUNBUFFERED / FORCE_COLOR / a fixed window size via TIOCSWINSZ help
    libraries that read env vars or query terminal geometry.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["FORCE_COLOR"] = "1"
    env["COLUMNS"] = str(cols)
    env["LINES"] = str(rows)
    env.setdefault("TERM", "xterm-256color")

    master_fd, slave_fd = pty.openpty()
    try:
        fcntl.ioctl(
            master_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, cols, 0, 0),
        )
    except OSError:
        pass  # Window size is a hint; not fatal if the ioctl is rejected.

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            bufsize=0,
            cwd=str(cwd),
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        os.close(master_fd)
        os.close(slave_fd)
        raise
    os.close(slave_fd)  # Parent doesn't need its end; child has it on 0/1/2.
    return proc, os.getpgid(proc.pid), master_fd


# -------------------- Reader thread + stream state machine --------------------
def reader_loop(fd: int, q: "queue.Queue[bytes | None]") -> None:
    """Pump raw output bytes from `fd` into `q`. Puts None on EOF.

    When `fd` is a PTY master, `os.read` raises OSError(EIO) once the slave
    side has been closed (typically because the child exited). That is the
    PTY way of signalling EOF, so it's handled identically to a 0-byte read.
    """
    try:
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                break
            if not chunk:
                break
            q.put(chunk)
    finally:
        q.put(None)


def new_state() -> dict:
    return {
        "queue": queue.Queue(),
        # Incremental UTF-8 decoder keeps partial multi-byte sequences across
        # chunk boundaries (rich emits lots of box-drawing characters).
        "decoder": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        "committed": [],
        "current": "",
        "pending_overwrite": False,
        "capture_outputs": False,
        "output_files": [],
        # label -> {"percent": int, "raw": str} — rich.Live multi-bar groups
        # commit each bar line on its own \n; we collapse them per-label so
        # the log isn't flooded with one entry per frame.
        "progress": {},
        "proc": None,
        "pgid": None,
        "master_fd": None,
        "reader": None,
        "started_at": None,
        "finished": False,
        "returncode": None,
    }


def close_master_fd(state: dict) -> None:
    fd = state.get("master_fd")
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    state["master_fd"] = None


def _commit_line(state: dict, line: str) -> None:
    if line == "---OUTPUT-FILES---":
        state["capture_outputs"] = True
        return
    if state["capture_outputs"]:
        state["output_files"].append(line)
        return
    parsed = parse_progress_line(line)
    if parsed is not None:
        label, pct = parsed
        state["progress"][label] = {"percent": pct, "raw": line.strip()}
        return
    state["committed"].append(line)


def _feed_text(state: dict, text: str) -> None:
    for ch in text:
        if ch == "\n":
            _commit_line(state, state["current"])
            state["current"] = ""
            state["pending_overwrite"] = False
        elif ch == "\r":
            # tqdm/rich emit "<full bar>\r" — keep the bar visible until
            # the next character begins a fresh overwrite from column 0.
            state["pending_overwrite"] = True
        elif ch == "\x08":
            # Backspace: trim last char of in-progress line.
            if state["current"]:
                state["current"] = state["current"][:-1]
        else:
            if state["pending_overwrite"]:
                state["current"] = ch
                state["pending_overwrite"] = False
            else:
                state["current"] += ch


def drain_queue(state: dict) -> bool:
    """Drain bytes queue through the \\r/\\n state machine. Return EOF flag."""
    eof = False
    q: queue.Queue = state["queue"]
    decoder = state["decoder"]
    while True:
        try:
            chunk = q.get_nowait()
        except queue.Empty:
            break
        if chunk is None:
            # Flush any trailing partial UTF-8 sequence as a replacement char.
            tail = decoder.decode(b"", final=True)
            if tail:
                _feed_text(state, ANSI_RE.sub("", tail))
            eof = True
            continue
        text = ANSI_RE.sub("", decoder.decode(chunk))
        _feed_text(state, text)
    return eof


def attach_reader(
    state: dict,
    proc: subprocess.Popen,
    pgid: int,
    master_fd: int,
) -> None:
    """Wire a fresh PTY-attached subprocess into `state` and start the reader."""
    state["proc"] = proc
    state["pgid"] = pgid
    state["master_fd"] = master_fd
    state["started_at"] = time.time()
    t = threading.Thread(
        target=reader_loop, args=(master_fd, state["queue"]),
        daemon=True,
    )
    t.start()
    state["reader"] = t
