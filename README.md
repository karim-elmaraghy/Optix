# Tunix

A developer-installed AI GPU inference optimization agent with a local
optimization workbench. No VS Code extension is required.

## Install and open the demo

From a VS Code terminal, activate your Python 3.10+ environment, then:

```sh
git clone --branch integration-optiloop https://github.com/karim-elmaraghy/Optix.git
cd Optix
python -m pip install -e .
tunix doctor
tunix init
tunix report --replay --port 7865
```

Open http://127.0.0.1:7865 and click **Open Workbench**. Doctor exits 2 on an
unsupported/missing GPU; saved-results replay still works without a GPU or key.
For another project, install this checkout using its absolute editable path,
run `tunix init` there, then `tunix report` for that project's results.

## Supported live workflow

**Tesla T4 + CUDA + meta-llama/Llama-3.2-1B-Instruct + CNN/DailyMail
summarization** is the only supported MVP. Arbitrary models, visitor-local
browser GPUs and persistent runtime activation are not implemented.

```sh
python -m pip install -e ".[gpu]"
tunix doctor
tunix benchmark
tunix optimize --budget 8
tunix export --output tunix-results.zip
tunix report
```

Accept the gated Llama license and securely set HF_TOKEN. Optimization also
requires MODEL_API_KEY. Neither belongs in tunix.json or exported settings.
The real backend measures baselines, attempts calibration, calls Meta for
candidate proposals only when calibration permits, and benchmarks/validates
candidates. Blocked optimization is not success. Benchmarking itself makes no
Meta call. Apply remains unavailable without a verified persistent runtime.

## Meta integration: advisory planner vs optimization

The standalone planner can explore the real V2 evidence even while
calibration-gated optimization is blocked. With MODEL_API_KEY securely set:

```sh
python -m optiloop.meta_planner --v2-benchmark results/benchmark_20261104T211417.json --allowed docs/meta-allowed-v2.json --output meta-proposal-new.json
```

Use a new output filename, outside results/, for each request. The command
checks archive hashes, submits measured evidence to the real Meta API and
validates the returned configuration against the remaining permitted space.
Successful output records source/model/HTTP provenance. It is **advisory only**:
not benchmarked, not quality-approved, not deployed, never an approved winner.
Missing credentials produce an explicit unverified record; no fallback is
substituted. The planner is CLI-only in this release. Live Meta access was not
verified in the release environment because MODEL_API_KEY is absent.

## Colab results -> Windows workbench

Run `tunix export`, download its ZIP, then use **Import benchmark results**.
Both Tunix ZIPs and the original V2 Colab ZIP are supported. Import preserves
original bytes in separate sessions and recomputes supported gates locally.
**Imported GPU results** is never presented as live Meta optimization.

See [COLAB_TRANSFER.md](COLAB_TRANSFER.md) for exact install, doctor, benchmark,
export/download and Windows upload commands. [DEVELOPER.md](DEVELOPER.md)
documents CLI paths, supported environments, exit codes and secure credentials.

## Verified evidence and release limitations

The committed real V2 T4 records show **5.63x raw throughput** and **81.8%
lower energy per document** for FP16 B8 vs competent FP16 B1. T5 batching
passes with minimum F1 1.000. These gains are **not quality-approved**.

- Default FP16 has 5/20 truncation flags; FP32 has 4/20. Strict calibration
  remains blocked; cap64 flags all 20 outputs. No threshold was relaxed.
- Supplied records used two repeats, below T3's three-repeat requirement.
- Held-out approval, a successful live Meta optimization and GPU rollback
  have not been demonstrated. No automatic deployment occurs.
- Original artifacts/hashes are in results/. Derived rejection reasons are
  in results/validation_report.json; details are in [V2_VALIDATION.md](V2_VALIDATION.md).
- Real ZIP upload -> session selection -> dashboard updates were exercised
  through the actual Gradio upload endpoint, including malicious rejection.
  Browser file-picker/visual interactions remain a manual review item.
- Synthetic tests are explicitly labeled, never displayed as measured runs.

Run the complete CPU suite with `python -m pip install -e ".[test]"` then
`python -m pytest tests -q`. Optional tiny-model batching tests skip if
GPU-framework dependencies are not installed. No PyPI publication was made.
Internal optiloop imports and original measured artifacts are preserved.
