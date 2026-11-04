# Integrated OptiLoop demonstration

This branch combines PR #4 (9068273), PR #5 (4eee925), and PR #6
(710db1f) without changing their active branches. Additional application wiring
lives in `optiloop/application.py`, `optiloop/integration.py`, and `app.py`.
No benchmark JSON keys or rejection-log schemas were changed.

## Local CPU verification and dashboard

```sh
python -m pip install pytest gradio==6.31.0
python -m pytest tests -q
python app.py
```

Open http://127.0.0.1:7860. This starts the UI without loading GPU libraries.
Real optimization requires the full requirements, model license, T4 and keys.
The optional tiny-Llama batching tests skip if torch/transformers are missing.
Synthetic tests use temporary directories, dummy API transports and fabricated
test sentinels; they are not measurements. No live Meta or GPU test was run by
the integration author.

## Exact Google Colab T4 setup

Accept the Hugging Face license for `meta-llama/Llama-3.2-1B-Instruct` first.
Select **Runtime > Change runtime type > T4 GPU**. Run this Python cell:

```python
import os, sys, subprocess, getpass
from pathlib import Path

subprocess.run(['nvidia-smi'], check=True)
repo = Path('/content/Optix-integration')
if not repo.exists():
    subprocess.run(['git', 'clone', '--branch', 'integration-optiloop',
                    'https://github.com/karim-elmaraghy/Optix.git', str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
os.environ['HF_TOKEN'] = getpass.getpass('HF_TOKEN: ')
os.environ['MODEL_API_KEY'] = getpass.getpass('MODEL_API_KEY: ')
```

Run the CPU suite in the same runtime (this also exercises tiny-model tests
when the installed torch/transformers combination supports them):

```python
subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], check=True)
```

Check the real API without displaying credentials or raw exception text:

```python
from optiloop.agent import meta_completion
try:
    reply = meta_completion([{'role': 'user', 'content': 'Reply OK only.'}])
    assert reply.strip()
except Exception as exc:
    print('Meta smoke failed:', type(exc).__name__)
else:
    print('Meta smoke passed')
```

Launch the dashboard in the notebook (the password protects the shared URL):

```python
import app, gradio as gr
demo = app.create_app()
demo.queue().launch(share=True, auth=('demo', getpass.getpass('Dashboard password: ')),
                    theme=gr.themes.Soft(), css=app.CSS)
```

Standalone local command: `python app.py`. Colab shell alternative after
setting the environment: `GRADIO_SHARE=1 python app.py` (without notebook auth).
The integration author tested localhost startup only, not public sharing.

## Demonstration sequence

1. Start with saved Gate 1 files in `results/` if available; otherwise the
   dashboard explicitly displays sample/not-run state. Do not create result
   files from the supplied narrative numbers. Existing raw T4 batch-4 and
   batch-8 results failed T5 and are not approved winners.
2. Click **Request Meta optimization**. This performs real GPU work: both
   tuning baselines, a default repeat for T1 (>=99% matching output text),
   up to eight Meta-selected experiments, and matching batch-1 references
   for every batched experiment. Each benchmark uses three repeats.
   Missing keys stop before model loading; failed API proposals are visibly
   labeled `rule_fallback`, not presented as real Meta decisions.
3. Only successful, exact-config experiments passing calibrated quality and
   T5 (every document F1 >=0.95) are eligible. No candidate means no winner;
   do not promote the fastest raw measurement. T5 rejection is recorded as
   `T5_batching_failed`, not a weakened quality threshold.
4. Review proposal sources, quality F1/ROUGE-L and rejection reasons. Click
   **Confirm winner on held-out documents**. This reruns winner plus both
   baselines on the ten disjoint held-out documents, three repeats, with
   another matching batch-1 comparison if needed. T8 reports whether the
   speedup is within 10%; otherwise use the held-out speedup only.
5. **Check apply readiness** deliberately reports unavailable. Benchmark
   models are unloaded after each experiment. There is no persistent serving
   runtime to activate or restore. The existing CPU safe-apply tests prove
   callback/state rollback with a bad eight-token fixture; they do not prove
   GPU model rollback. The required runtime lifecycle is specified in
   `optiloop/safe_apply.py`; no automatic GPU activation is performed.
6. Preserve and commit genuine measured JSON before making hackathon claims.
   Keep console energy cross-check/compile diagnostics too. T9 full-grid
   agent-vs-random ablation remains separate and is not implemented by this
   dashboard flow. No agent superiority claim follows from this demo alone.

## Artifact paths and limitations

- Each optimization creates `results/run_<id>/` with unchanged-schema
  `benchmark_session.json`, `outputs_<profile>.json`, and `agent_log.json`
  (JSONL, including rejection records).
- `results/current_session.json` points the dashboard to the latest measured
  session; dependency-injected synthetic runs never update it. `session.json`
  is a separate status/provenance sidecar, not a benchmark schema extension.
- Held-out evidence is in the session's `heldout/confirmation.json` plus
  real output bundles. Held-out runs do not overwrite tuning artifacts.
- `gpu_verified:false` means no manual hardware certification is recorded;
  a saved numerical result is not a claim that the integration author ran it.
- The UI serializes optimization/held-out operations with one process lock.
  This is a single-process hackathon app, not a multi-server GPU service.
