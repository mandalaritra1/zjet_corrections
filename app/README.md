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

## Headless CLI (no UI)

The same runner can be driven directly from the terminal:

```bash
python scripts/run_analysis_cli.py --config .analysis_widget_config.json
```

`.analysis_widget_config.json` is written either by the notebook (its existing
"Apply settings" button) or by the Streamlit app's "Save settings" button.
