# Z+jet analysis runner (Streamlit app)

A lightweight replacement for `notebooks/run_analysis.ipynb`. Renders the same
configuration form, launches `scripts/run_analysis_cli.py` as a subprocess,
and streams its stdout live to the page.

The Streamlit form reads/writes the same `.analysis_widget_config.json` at the
repo root that the notebook uses, so you can switch between notebook and app
without losing settings.

## Port convention

- **`8501`** (or any free local port): use for **local laptop runs**.
- **`8883`**: reserved for runs on a remote interactive node (LPC, casa) so
  that an `ssh -L 8883:localhost:8883` forward always points there. Do **not**
  use 8883 locally — you will collide with the forwarded session.

## Local

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Then open <http://localhost:8501>.

## LPC interactive node (port-forwarded)

On the node:

```bash
streamlit run app/streamlit_app.py --server.port 8883 --server.address 0.0.0.0
```

On your laptop:

```bash
ssh -L 8883:localhost:8883 <your-lpc-node>
```

Then open <http://localhost:8883>.

## Which Python runs the analysis subprocess?

The app spawns `scripts/run_analysis_cli.py` as a subprocess. By default it
picks the interpreter in this priority order:

1. The **"Override Python path"** text box (under *Environment* on the Run tab).
2. The `ZJET_CLI_PYTHON` environment variable.
3. `$VIRTUAL_ENV/bin/python` if a venv is active.
4. `$CONDA_PREFIX/bin/python` if a conda env is active.
5. `sys.executable` — whatever Python is running Streamlit.

The resolved path is shown both in the *Environment* expander and at the top
of the live log so you can verify it before/during a run.

### LPC example

`lpcjobqueue` (needed by `executor_mode=dask-lpc`) lives in a specific env at
LPC, but `streamlit` is often pip-installed in your user site, so `streamlit`
runs under a Python that doesn't have `lpcjobqueue`. The fix is to activate
the analysis env first so `VIRTUAL_ENV` is set — autodetect then picks it up:

```bash
source /path/to/coffea2025/bin/activate
streamlit run app/streamlit_app.py --server.port 8883 --server.address 0.0.0.0
```

Or set the env var explicitly:

```bash
export ZJET_CLI_PYTHON=/path/to/coffea2025/bin/python
streamlit run app/streamlit_app.py --server.port 8883 --server.address 0.0.0.0
```

Or paste the path into the *Override Python path* box in the UI.

## Headless CLI (no UI)

The same runner can be driven directly from the terminal:

```bash
python scripts/run_analysis_cli.py --config .analysis_widget_config.json
```

`.analysis_widget_config.json` is written either by the notebook (its existing
"Apply settings" button) or by the Streamlit app's "Save settings" button.
