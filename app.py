"""OptiLoop's results-first Gradio dashboard.

Run with ``python app.py`` after installing Gradio. This module deliberately
keeps GPU and benchmark dependencies out of its import path.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SAMPLE_DIR = RESULTS_DIR / "sample"

NAIVE_PROFILE = "naive_fp32_eager_b1"
DEFAULT_PROFILE = "default_fp16_sdpa_b1"
TITLE = "OptiLoop - Find the fastest correct profile for YOUR GPU"
CSS = """
.gradio-container { max-width: 1440px !important; }
.optix-card { border: 1px solid #29334a; border-radius: 14px; padding: 1rem; }
"""


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError(f"{path} must contain a JSON list of benchmark result objects")
    return data


def _load_agent_steps(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        steps = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        if isinstance(data, list):
            steps = data
        elif isinstance(data, dict):
            if "steps" in data or "agent_steps" in data:
                steps = data.get("steps", data.get("agent_steps", []))
            else:
                steps = [data]
        else:
            raise ValueError(f"{path} must contain a list or an object with a steps list")
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError(f"{path} steps must be a list of agent step objects")
    return [_normalize_agent_step(step) for step in steps]


def _normalize_agent_step(step: dict[str, Any]) -> dict[str, Any]:
    proposal = step.get("proposal")
    proposal = proposal if isinstance(proposal, dict) else {}
    result = step.get("result")
    result = result if isinstance(result, dict) else {}
    validation = result.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    accepted = step.get("accepted")
    decision = step.get("decision")
    if decision is None and isinstance(accepted, bool):
        decision = "PROMOTE" if accepted else "REJECT"
    reasons = step.get("reasons")
    reason = step.get("reason")
    if not reason and isinstance(reasons, list):
        reason = "; ".join(str(item) for item in reasons)
    return {
        **step,
        "config": step.get("config", proposal.get("config")),
        "hypothesis": step.get("hypothesis", proposal.get("hypothesis", proposal.get("reason"))),
        "source": step.get("source", proposal.get("source")),
        "decision": decision,
        "reason": reason or validation.get("error") or ", ".join(validation.get("reasons", [])),
        "benchmark": benchmark,
        "validation": validation,
        "accepted": accepted,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object")
        records.append(record)
    return records


def load_dashboard_data(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Load real results when present, otherwise the explicitly marked samples."""
    results_dir = Path(results_dir)
    sample_dir = results_dir / "sample"
    errors: list[str] = []
    benchmark_files = sorted(
        results_dir.glob("benchmark_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    benchmark_is_sample = not benchmark_files
    benchmark_path = benchmark_files[-1] if benchmark_files else sample_dir / "benchmark_sample.json"

    try:
        benchmarks = _load_benchmark(benchmark_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        benchmarks = []
        errors.append(f"Could not load benchmark results from {benchmark_path}: {error}")

    agent_path = results_dir / "agent_log.json"
    agent_is_sample = not agent_path.is_file()
    if agent_path.is_file():
        selected_agent_path = agent_path
    else:
        selected_agent_path = sample_dir / "agent_log_sample.json"
    try:
        agent_steps = _load_agent_steps(selected_agent_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        agent_steps = []
        errors.append(f"Could not load agent log from {selected_agent_path}: {error}")

    try:
        apply_events = _load_jsonl(results_dir / "apply_log.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        apply_events = []
        errors.append(f"Could not load safe-apply log: {error}")
    try:
        dashboard_apply_events = _load_jsonl(results_dir / "dashboard_apply_log.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        dashboard_apply_events = []
        errors.append(f"Could not load dashboard apply details: {error}")

    return {
        "benchmarks": benchmarks,
        "agent_steps": agent_steps,
        "apply_events": apply_events,
        "dashboard_apply_events": dashboard_apply_events,
        "benchmark_is_sample": benchmark_is_sample,
        "agent_is_sample": agent_is_sample,
        "has_real_benchmarks": bool(benchmark_files) and not benchmark_is_sample,
        "errors": errors,
        "results_dir": str(results_dir),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _format_metric(value: Any, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "Not recorded"
    return f"{number:.3f}{suffix}"


def _config_key(config: Any) -> str | None:
    if not isinstance(config, dict):
        return None
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _decision(step: dict[str, Any]) -> str:
    value = step.get("decision", step.get("status", ""))
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _source_label(step: dict[str, Any]) -> str:
    source = str(step.get("source", "")).strip().lower()
    if source in {"muse_api", "meta_api"}:
        return "Meta API"
    if source == "rule_fallback":
        return "Rule fallback (not Meta API)"
    return str(step.get("source") or "Source not recorded")


def _agent_decisions(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for step in steps:
        config = step.get("config", step.get("requested_config"))
        key = _config_key(config)
        if key:
            decisions[key] = step
    return decisions


def _profile_decision(
    result: dict[str, Any], decisions: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    config = result.get("requested_config", result.get("actual_config"))
    key = _config_key(config)
    return decisions.get(key) if key else None


def _status_and_reason(
    result: dict[str, Any], step: dict[str, Any] | None
) -> tuple[str, str]:
    if str(result.get("status", "")).lower() == "failed":
        return "Failed", str(result.get("error") or result.get("reason") or "Failure reason not recorded")
    decision = _decision(step or {})
    if decision in {"promote", "accepted", "accept", "pass", "passed"}:
        return "Accepted", str((step or {}).get("reason") or "Accepted; no additional reason recorded")
    if decision in {"reject", "rejected", "fail", "failed"}:
        return "Rejected", str((step or {}).get("reason") or "Rejection reason not recorded")
    reason = result.get("error") or result.get("reason")
    return "Decision not recorded", str(reason or "No acceptance decision or reason recorded")


def _find_chosen(
    benchmarks: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> dict[str, Any] | None:
    candidates = []
    for step in steps:
        decision = _decision(step)
        if step.get("chosen") is not True and step.get("selected") is not True and decision not in {
            "promote",
            "accepted",
            "accept",
        }:
            continue
        result = step.get("benchmark")
        result = result if isinstance(result, dict) else {}
        config = step.get("config", step.get("requested_config"))
        matched = next(
            (
                item
                for item in benchmarks
                if _config_key(config)
                and _config_key(config)
                in {
                    _config_key(item.get("requested_config")),
                    _config_key(item.get("actual_config")),
                }
            ),
            None,
        )
        selected = dict(matched or result)
        if selected:
            selected.setdefault("profile_name", step.get("profile_name"))
            selected.setdefault("requested_config", config)
            selected.setdefault("actual_config", config)
            selected["_agent_validation"] = step.get("validation", {})
            selected["_agent_decision"] = step.get("decision")
            selected["_agent_reason"] = step.get("reason")
            throughput = _number(selected.get("docs_per_sec"))
            candidates.append((step.get("chosen") is True or step.get("selected") is True, throughput, selected))
    explicit = [item for item in candidates if item[0]]
    if explicit:
        return explicit[-1][2]
    measured = [item for item in candidates if item[1] is not None]
    if measured:
        return max(measured, key=lambda item: item[1])[2]

    for result in reversed(benchmarks):
        if result.get("chosen") is True or result.get("selected") is True or result.get("winner") is True:
            return result
    return None


def _quality_values(result: dict[str, Any] | None) -> tuple[Any, Any]:
    if not result:
        return None, None
    validation = result.get("_agent_validation")
    if isinstance(validation, dict):
        rows = validation.get("docs")
        if isinstance(rows, list) and rows:
            f1 = [_number(row.get("token_f1")) for row in rows if isinstance(row, dict)]
            rouge = [_number(row.get("rouge_l")) for row in rows if isinstance(row, dict)]
            f1 = [value for value in f1 if value is not None]
            rouge = [value for value in rouge if value is not None]
            return (
                sum(f1) / len(f1) if f1 else None,
                sum(rouge) / len(rouge) if rouge else None,
            )
    return result.get("quality_f1"), result.get("rouge_l")


def _heldout_event(snapshot: dict[str, Any], chosen: dict[str, Any] | None) -> dict[str, Any]:
    if not chosen:
        return {}
    config = chosen.get("actual_config") or chosen.get("requested_config")
    for event in reversed(snapshot.get("dashboard_apply_events", [])):
        if _config_key(event.get("candidate_config")) == _config_key(config):
            return event
    for event in reversed(snapshot.get("apply_events", [])):
        if _config_key(event.get("active_config")) == _config_key(config):
            return event
    return {}


def _heldout(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    nested = result.get("heldout")
    data = nested if isinstance(nested, dict) else {}
    fields = {
        "status": data.get("status", result.get("heldout_status")),
        "decision": data.get("decision", result.get("heldout_decision")),
        "quality_f1": data.get("quality_f1", result.get("heldout_quality_f1")),
        "rouge_l": data.get("rouge_l", result.get("heldout_rouge_l")),
        "docs_per_sec": data.get("docs_per_sec", result.get("heldout_docs_per_sec")),
        "joules_per_doc": data.get("joules_per_doc", result.get("heldout_joules_per_doc")),
        "reason": data.get("reason", result.get("heldout_reason")),
    }
    if str(result.get("split", "")).lower() == "heldout":
        for field in ("status", "decision", "quality_f1", "rouge_l", "docs_per_sec", "joules_per_doc", "reason"):
            if fields[field] is None:
                fields[field] = result.get(field)
    return fields


def _baseline(benchmarks: list[dict[str, Any]], profile_name: str) -> dict[str, Any] | None:
    return next(
        (result for result in benchmarks if result.get("profile_name") == profile_name),
        None,
    )


def _speedup(candidate: Any, baseline: Any) -> float | None:
    first = _number(candidate)
    second = _number(baseline)
    if first is None or second is None or first <= 0 or second <= 0:
        return None
    return first / second


def _percent_saving(candidate: Any, baseline: Any) -> float | None:
    first = _number(candidate)
    second = _number(baseline)
    if first is None or second is None or second <= 0:
        return None
    return (second - first) / second * 100


def _display_value(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _gpu_info() -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        return f"GPU detection unavailable: {error}", "Benchmark GPU: Not recorded"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"nvidia-smi exited with code {completed.returncode}"
        return f"GPU detection unavailable: {detail}", "Benchmark GPU: Not recorded"
    detected = completed.stdout.strip().splitlines()
    return (
        f"Detected runtime GPU: {', '.join(detected) if detected else 'No GPU reported'} "
        "(this is the current runtime, not proof of a benchmark run)",
        "Benchmark GPU: Not recorded",
    )


def _result_metadata(benchmarks: list[dict[str, Any]]) -> tuple[str, str]:
    gpu_values = []
    model_values = []
    for result in benchmarks:
        for field in ("gpu_name", "gpu", "device_name"):
            value = result.get(field)
            if value and str(value) not in gpu_values:
                gpu_values.append(str(value))
        for field in ("model_name", "model", "model_id"):
            value = result.get(field)
            if value and str(value) not in model_values:
                model_values.append(str(value))
    return (
        f"Benchmark GPU: {', '.join(gpu_values) if gpu_values else 'Not recorded'}",
        f"Model: {', '.join(model_values) if model_values else 'Not recorded'}",
    )


def build_dashboard_view(data: dict[str, Any]) -> dict[str, Any]:
    steps = data["agent_steps"]
    benchmarks = list(data["benchmarks"])
    known_configs = {
        _config_key(result.get("requested_config") or result.get("actual_config"))
        for result in benchmarks
    }
    for step in steps:
        result = step.get("benchmark")
        if not isinstance(result, dict) or not result:
            continue
        key = _config_key(result.get("requested_config") or result.get("actual_config"))
        if key and key not in known_configs:
            benchmarks.append(result)
            known_configs.add(key)
    decisions = _agent_decisions(steps)
    naive = _baseline(benchmarks, NAIVE_PROFILE)
    competent = _baseline(benchmarks, DEFAULT_PROFILE)
    chosen = _find_chosen(benchmarks, steps)

    if data["benchmark_is_sample"]:
        banner = (
            "**SAMPLE DATA - NOT REAL RESULTS**  \n"
            "No real benchmark result set is available. Profiles, measured metrics, "
            "and hold-out confirmation remain in a not-run state."
        )
    elif not data["benchmarks"]:
        banner = "**No valid real benchmark results are available.** See any loading errors below."
    else:
        banner = "**Real benchmark results loaded.** Verify every displayed field against the saved JSON."
        if data["agent_is_sample"]:
            banner += " Agent log: not run or unavailable."
    if data["errors"]:
        banner += "\n\n**Data loading errors:** " + " | ".join(data["errors"])

    runtime_gpu, _ = _gpu_info()
    benchmark_gpu, model = _result_metadata(benchmarks)
    gpu_and_model = f"### Hardware and model\n\n{runtime_gpu}\n\n{benchmark_gpu}\n\n{model}"

    profile_rows = []
    for result in benchmarks:
        step = _profile_decision(result, decisions)
        status, reason = _status_and_reason(result, step)
        quality_source = dict(result)
        if step:
            quality_source["_agent_validation"] = step.get("validation")
        quality_f1, rouge_l = _quality_values(quality_source)
        profile_rows.append(
            [
                _display_value(result.get("profile_name")),
                status,
                _format_metric(result.get("docs_per_sec")),
                _format_metric(result.get("latency_p50_sec"), " s"),
                _format_metric(result.get("latency_p95_sec"), " s"),
                _format_metric(result.get("joules_per_doc")),
                _format_metric(quality_f1),
                _format_metric(rouge_l),
                reason,
            ]
        )
    if not profile_rows:
        profile_rows = [
            [NAIVE_PROFILE, "Not run", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "No benchmark result exists"],
            [DEFAULT_PROFILE, "Not run", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "Not recorded", "No benchmark result exists"],
        ]

    baseline_rows = []
    for label, profile in (("Naive baseline (fp32/eager/batch 1)", naive), ("Competent default (fp16/SDPA/batch 1)", competent)):
        baseline_rows.append(
            [
                label,
                _display_value(profile.get("status")) if profile else "Not run",
                _format_metric(profile.get("docs_per_sec")) if profile else "Not recorded",
                _format_metric(profile.get("latency_p50_sec"), " s") if profile else "Not recorded",
                _format_metric(profile.get("latency_p95_sec"), " s") if profile else "Not recorded",
                _format_metric(profile.get("joules_per_doc"), " J/doc") if profile else "Not recorded",
            ]
        )

    chosen_name = _display_value(chosen.get("profile_name")) if chosen else "No configuration selected"
    chosen_config = chosen.get("actual_config") or chosen.get("requested_config") if chosen else None
    heldout = _heldout(chosen)
    heldout_event = _heldout_event(data, chosen)
    event_validation = heldout_event.get("apply_result", {}).get("validation", {})
    if not isinstance(event_validation, dict):
        event_validation = {}
    event_quality = _quality_values({"_agent_validation": event_validation})
    heldout_benchmark = heldout_event.get("benchmark")
    if not isinstance(heldout_benchmark, dict):
        heldout_benchmark = {}
    heldout["quality_f1"] = heldout.get("quality_f1") or event_quality[0]
    heldout["rouge_l"] = heldout.get("rouge_l") or event_quality[1]
    heldout["docs_per_sec"] = heldout.get("docs_per_sec") or heldout_benchmark.get("docs_per_sec")
    heldout["joules_per_doc"] = heldout.get("joules_per_doc") or heldout_benchmark.get("joules_per_doc")
    if not heldout.get("decision") and event_validation:
        heldout["decision"] = "accepted" if event_validation.get("accepted") is True else "rejected"
        heldout["reason"] = "; ".join(event_validation.get("reasons", []))
    heldout_decision = str(heldout.get("decision") or heldout.get("status") or "").lower()
    if any(token in heldout_decision for token in ("accept", "promote", "pass")):
        heldout_summary = "Held-out confirmation: accepted in recorded results."
    elif any(token in heldout_decision for token in ("reject", "fail")):
        heldout_summary = "Held-out confirmation: rejected in recorded results."
    else:
        heldout_summary = "Held-out confirmation: not recorded; do not treat as passed."
    winner_lines = [
        f"**Chosen profile:** {chosen_name}",
        f"**Configuration:** `{json.dumps(chosen_config, sort_keys=True) if chosen_config else 'Not recorded'}`",
        heldout_summary,
        f"**Held-out quality F1:** {_format_metric(heldout.get('quality_f1'))}",
        f"**Held-out ROUGE-L:** {_format_metric(heldout.get('rouge_l'))}",
        f"**Held-out throughput:** {_format_metric(heldout.get('docs_per_sec'), ' docs/s')}",
        f"**Held-out energy:** {_format_metric(heldout.get('joules_per_doc'), ' J/doc')}",
        f"**Held-out reason:** {_display_value(heldout.get('reason') or 'No reason recorded')}",
    ]

    comparisons: list[list[str]] = []
    winner_docs = chosen.get("docs_per_sec") if chosen else None
    winner_energy = chosen.get("joules_per_doc") if chosen else None
    for label, baseline in (("vs naive baseline", naive), ("vs competent default", competent)):
        speedup = _speedup(winner_docs, baseline.get("docs_per_sec") if baseline else None)
        saving = _percent_saving(winner_energy, baseline.get("joules_per_doc") if baseline else None)
        comparisons.append(
            [
                label,
                f"{speedup:.3f}x" if speedup is not None else "Not recorded",
                f"{saving:.2f}%" if saving is not None else "Not recorded",
            ]
        )
    default_speedup = _speedup(winner_docs, competent.get("docs_per_sec") if competent else None)
    if chosen and default_speedup is not None:
        gate = "GATE 1 PASSED" if default_speedup >= 1.2 else "GATE 1 NOT PASSED"
    else:
        gate = "GATE 1 NOT RUN - winner and competent-default measurements are required"
    winner_lines.append(f"**{gate}**")

    chart_rows = []
    plotted: set[str] = set()
    for role, result in (("Naive", naive), ("Competent default", competent), ("Chosen", chosen)):
        if not result:
            continue
        value = _number(result.get("docs_per_sec"))
        name = str(result.get("profile_name") or role)
        if value is not None and name not in plotted:
            chart_rows.append({"Profile": name, "docs/s": value, "Comparison": role})
            plotted.add(name)

    rejection_rows = []
    for result in benchmarks:
        step = _profile_decision(result, decisions)
        status, reason = _status_and_reason(result, step)
        if status in {"Rejected", "Failed"}:
            rejection_rows.append([result.get("profile_name", "Profile not recorded"), status, reason])
    if not rejection_rows:
        rejection_rows = [["No rejected or failed configurations recorded", "Not run", "No rejection/failure reason available"]]

    agent_rows = [
        [
            _display_value(step.get("hypothesis")),
            _source_label(step),
            _display_value(step.get("decision", step.get("status"))),
            _display_value(step.get("reason")),
        ]
        for step in steps
    ] or [["No agent steps recorded", "Source not recorded", "Not run", "No agent log entries available"]]

    return {
        "banner": banner,
        "gpu_and_model": gpu_and_model,
        "profile_rows": profile_rows,
        "baseline_rows": baseline_rows,
        "comparison_rows": comparisons,
        "chart_rows": chart_rows,
        "winner_markdown": "\n\n".join(winner_lines),
        "rejection_rows": rejection_rows,
        "agent_rows": agent_rows,
        "snapshot": {
            **data,
            "benchmarks": benchmarks,
            "chosen": chosen,
            "naive": naive,
            "competent": competent,
            "winner": chosen,
        },
    }


def refresh_dashboard() -> tuple[Any, ...]:
    view = build_dashboard_view(load_dashboard_data())
    return (
        view["banner"],
        view["gpu_and_model"],
        view["profile_rows"],
        view["baseline_rows"],
        view["comparison_rows"],
        view["chart_rows"],
        view["winner_markdown"],
        view["rejection_rows"],
        view["agent_rows"],
        view["snapshot"],
    )


def _output_bundle(path: Path, split: str) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("docs"), list):
        raise ValueError(f"{path} must contain an output bundle with docs")
    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a config")
    docs = [
        doc for doc in data["docs"]
        if isinstance(doc, dict) and doc.get("split") == split
    ]
    if not docs:
        raise ValueError(f"{path} has no {split} output documents")
    return {"config": config, "docs": docs}


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def safe_apply(snapshot: dict[str, Any] | None) -> str:
    """Run Agent B's held-out validation and rollback path with Agent A's runner."""
    if not snapshot or not snapshot.get("has_real_benchmarks"):
        return "Not applied: real benchmark results are not available."
    chosen = snapshot.get("chosen")
    if not isinstance(chosen, dict):
        return "Not applied: no accepted, measured configuration is recorded."
    config = chosen.get("actual_config") or chosen.get("requested_config")
    if not isinstance(config, dict):
        return "Not applied: the chosen configuration is not recorded."
    profile_name = chosen.get("profile_name")
    if not profile_name:
        return "Not applied: the chosen profile name is not recorded."

    results_dir = Path(snapshot.get("results_dir", RESULTS_DIR))
    try:
        default_tuning = _output_bundle(results_dir / f"outputs_{DEFAULT_PROFILE}.json", "tuning")
        naive_tuning = _output_bundle(results_dir / f"outputs_{NAIVE_PROFILE}.json", "tuning")
        default_heldout = _output_bundle(results_dir / f"outputs_{DEFAULT_PROFILE}.json", "heldout")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return f"Not applied: required tuning calibration and competent-default held-out artifacts are unavailable: {error}"

    try:
        from optiloop.benchmark import run_config
        from optiloop.profiles import DEFAULT_PROFILE as DEFAULT_PROFILE_NAME
        from optiloop.profiles import PROFILES, validate_config
        from optiloop.safe_apply import safe_apply as run_safe_apply
        from optiloop.validator import calibrate
        from optiloop.workload import load_workload
    except ImportError as error:
        return f"Not applied: benchmark or Agent B integration is unavailable: {error}"

    if DEFAULT_PROFILE_NAME != DEFAULT_PROFILE or DEFAULT_PROFILE not in PROFILES:
        return "Not applied: shared competent-default profile does not match the dashboard contract."
    try:
        validate_config(config)
        default_config = dict(PROFILES[DEFAULT_PROFILE])
        calibration = calibrate(default_tuning, naive_tuning)
    except (KeyError, TypeError, ValueError) as error:
        return f"Not applied: invalid profile or tuning calibration artifacts: {error}"

    active_config: dict[str, Any] = {}
    heldout_result: dict[str, Any] = {}

    def apply_config(next_config: dict[str, Any]) -> None:
        validate_config(next_config)
        active_config.clear()
        active_config.update(next_config)

    def evaluate_heldout(next_config: dict[str, Any]) -> dict[str, Any]:
        _tuning_docs, heldout_docs = load_workload()
        result = run_config(
            next_config,
            heldout_docs,
            profile_name=profile_name,
            split="heldout",
            repeats=3,
            results_dir=str(results_dir),
        )
        heldout_result.update(result)
        return {"benchmark": result["benchmark"], "outputs": result["outputs"]}

    try:
        event = run_safe_apply(
            config,
            default_config,
            default_heldout,
            calibration,
            apply_config,
            evaluate_heldout,
            state_path=str(results_dir / "active_config.json"),
            log_path=str(results_dir / "apply_log.jsonl"),
        )
    except RuntimeError as error:
        return f"Safe apply could not confirm rollback: {error}"

    record = {
        "candidate_config": config,
        "apply_result": event,
        "benchmark": heldout_result.get("benchmark"),
    }
    try:
        _append_jsonl(results_dir / "dashboard_apply_log.jsonl", record)
    except OSError as error:
        return f"Safe apply returned {event.get('action')}; dashboard confirmation could not be saved: {error}"

    validation = event.get("validation", {})
    measured = heldout_result.get("benchmark", {})
    quality_rows = validation.get("docs", []) if isinstance(validation, dict) else []
    f1, rouge = _quality_values({"_agent_validation": {"docs": quality_rows}})
    return json.dumps(
        {
            "action": event.get("action"),
            "active_config": event.get("active_config"),
            "heldout_accepted": validation.get("accepted"),
            "heldout_reasons": validation.get("reasons"),
            "mean_heldout_token_f1": f1,
            "mean_heldout_rouge_l": rouge,
            "heldout_docs_per_sec": measured.get("docs_per_sec"),
            "heldout_joules_per_doc": measured.get("joules_per_doc"),
        },
        sort_keys=True,
    )


def create_app() -> Any:
    import gradio as gr

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n\nA results-first view of correctness, speed, energy, and agent decisions.")
        banner = gr.Markdown()
        snapshot_state = gr.State()
        with gr.Row():
            gpu = gr.Markdown(elem_classes=["optix-card"])
            gr.Markdown(
                "### Trust the result\n"
                "Only values present in the loaded JSON are shown as measurements. "
                "Missing fields remain **Not recorded**.",
                elem_classes=["optix-card"],
            )
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Tested configurations")
                profiles = gr.Dataframe(
                    headers=[
                        "Profile",
                        "Status",
                        "docs/s",
                        "p50 latency",
                        "p95 latency",
                        "J/doc",
                        "Quality F1 vs default",
                        "ROUGE-L vs gold",
                        "Reason",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Column(scale=2):
                gr.Markdown("## Baselines")
                baselines = gr.Dataframe(
                    headers=["Baseline", "Run status", "docs/s", "p50 latency", "p95 latency", "Energy"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown("## Comparison")
                comparisons = gr.Dataframe(
                    headers=["Comparison", "Throughput speedup", "Energy saved"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                )
        gr.Markdown("## Throughput comparison")
        chart = gr.BarPlot(
            x="Profile",
            y="docs/s",
            color="Comparison",
            color_map={"Naive": "#ef4444", "Competent default": "#94a3b8", "Chosen": "#22c55e"},
            colors_in_legend=["Naive", "Competent default", "Chosen"],
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Chosen configuration and held-out confirmation")
                winner = gr.Markdown()
            with gr.Column():
                gr.Markdown("## Rejection log")
                rejections = gr.Dataframe(
                    headers=["Profile", "Status", "Recorded reason"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
        gr.Markdown("## Agent proposals")
        gr.Markdown("Meta API proposals and rule-based fallback are labeled separately; fallback is not the Meta agent.")
        agent_log = gr.Dataframe(
            headers=["Hypothesis", "Source", "Decision", "Reason"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            refresh = gr.Button("Refresh results", variant="secondary")
            apply_button = gr.Button("Safely apply chosen profile", variant="primary")
        gr.Markdown(
            "Safe apply re-runs held-out validation through Agent B and rolls back on rejection. "
            "The dashboard has no separate model-serving endpoint."
        )
        apply_result = gr.Textbox(
            label="Safe apply and rollback",
            value="No profile has been applied.",
            interactive=False,
            lines=3,
        )

        outputs = [
            banner,
            gpu,
            profiles,
            baselines,
            comparisons,
            chart,
            winner,
            rejections,
            agent_log,
            snapshot_state,
        ]
        demo.load(refresh_dashboard, outputs=outputs)
        refresh.click(refresh_dashboard, outputs=outputs)
        apply_button.click(safe_apply, inputs=snapshot_state, outputs=apply_result)
    return demo


if __name__ == "__main__":
    import gradio as gr

    app = create_app()
    app.launch(
        share=False,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate"),
        css=CSS,
    )
