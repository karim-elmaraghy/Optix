# OptiLoop dashboard

From the repository root, install the UI dependency and launch the dashboard:

```bash
python -m pip install gradio
python app.py
```

In Google Colab, run these cells from a fresh runtime:

```python
!git clone https://github.com/karim-elmaraghy/Optix_.git
%cd /content/Optix_
%pip install gradio

import os
import subprocess

env = os.environ.copy()
env["GRADIO_SERVER_NAME"] = "0.0.0.0"
dashboard = subprocess.Popen(["python", "app.py"], env=env)

from google.colab import output
output.serve_kernel_port_as_iframe(7860)
```

The app intentionally launches with `share=False`; it does not support a
`share=True` option. Colab displays the local app through its kernel-port proxy.
No GPU Python libraries are needed to start the dashboard.

The UI reads the newest `results/benchmark_*.json` and `results/agent_log.json`.
Until real files exist, it uses the empty schema-shaped files under
`results/sample/` and shows **SAMPLE DATA - NOT REAL RESULTS**. Empty sample
files intentionally show baselines as not run rather than suggesting a
completed experiment. Safe apply requires tuning calibration outputs and a
competent-default held-out output, plus an active inference runtime callback.
Agent B's `optiloop.safe_apply` requires that callback; the dashboard currently
fails closed and does not claim that a profile was applied or rolled back.
