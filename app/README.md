# OptiLoop dashboard

From the repository root, install the UI dependency and launch the dashboard:

```bash
python -m pip install gradio
python app.py
```

In Google Colab, clone the repository, install Gradio in a cell with
`%pip install gradio`, then run `!python app.py`. The app uses `share=False`;
Gradio renders the local app in the notebook runtime. No GPU Python libraries
are needed to start the dashboard. For safe apply, install the full project
requirements and use a runtime with the benchmark model and GPU dependencies
configured.

The UI reads the newest `results/benchmark_*.json` and `results/agent_log.json`.
Until real files exist, it uses the empty schema-shaped files under
`results/sample/` and shows **SAMPLE DATA - NOT REAL RESULTS**. Empty sample
files intentionally show baselines as not run rather than suggesting a
completed experiment. Safe apply requires tuning calibration outputs and a
competent-default held-out output; it calls Agent B's `optiloop.safe_apply`
and Agent A's held-out runner, then records the result and any rollback.
