import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import app


class DashboardTests(unittest.TestCase):
    def test_empty_results_are_explicitly_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            sample = results / "sample"
            sample.mkdir()
            (sample / "benchmark_sample.json").write_text("[]", encoding="utf-8")
            (sample / "agent_log_sample.json").write_text('{"steps":[]}', encoding="utf-8")

            data = app.load_dashboard_data(results)
            view = app.build_dashboard_view(data)

        self.assertIn("SAMPLE DATA - NOT REAL RESULTS", view["banner"])
        self.assertIn("No real benchmark result set", view["banner"])
        self.assertEqual([row[1] for row in view["profile_rows"]], ["Not run", "Not run"])
        self.assertEqual(view["chart_rows"], [])
        self.assertFalse(data["has_real_benchmarks"])

    def test_real_json_values_and_reasons_are_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            benchmark = [
                {
                    "profile_name": "naive_fp32_eager_b1",
                    "status": "success",
                    "requested_config": {"dtype": "float32"},
                    "docs_per_sec": 1.25,
                    "joules_per_doc": 4.5,
                    "latency_p50_sec": 0.8,
                    "latency_p95_sec": 1.1,
                },
                {
                    "profile_name": "default_fp16_sdpa_b1",
                    "status": "success",
                    "requested_config": {"dtype": "float16"},
                    "docs_per_sec": 2.5,
                    "joules_per_doc": 2.0,
                    "latency_p50_sec": 0.4,
                    "latency_p95_sec": 0.6,
                },
                {
                    "profile_name": "candidate",
                    "status": "failed",
                    "error": "measured backend failure",
                    "requested_config": {"dtype": "float16", "batch": 4},
                },
                {
                    "profile_name": "rejected_candidate",
                    "status": "success",
                    "requested_config": {"dtype": "float16", "batch": 8},
                    "docs_per_sec": 2.4,
                },
            ]
            (results / "benchmark_real.json").write_text(json.dumps(benchmark), encoding="utf-8")
            (results / "agent_log.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "config": {"dtype": "float16", "batch": 4},
                                "hypothesis": "Try a larger batch",
                                "source": "rule_fallback",
                                "decision": "REJECT",
                                "reason": "Candidate failed during execution",
                            },
                            {
                                "config": {"dtype": "float16", "batch": 8},
                                "hypothesis": "Try a different batch",
                                "source": "muse_api",
                                "decision": "REJECT",
                                "reason": "measured validator rejection",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            data = app.load_dashboard_data(results)
            view = app.build_dashboard_view(data)

        self.assertEqual(view["profile_rows"][2][1], "Failed")
        self.assertEqual(view["profile_rows"][2][8], "measured backend failure")
        self.assertIn("Rule fallback", view["agent_rows"][0][1])
        self.assertIn("measured backend failure", view["rejection_rows"][0][2])
        self.assertIn("measured validator rejection", view["rejection_rows"][1][2])
        self.assertEqual(view["comparison_rows"][0][1], "Not recorded")
        self.assertFalse(data["benchmark_is_sample"])

    def test_agent_sources_are_distinguished(self):
        self.assertEqual(app._source_label({"source": "muse_api"}), "Meta API")
        self.assertEqual(app._source_label({"source": "meta_api"}), "Meta API")
        self.assertEqual(
            app._source_label({"source": "rule_fallback"}),
            "Rule fallback (not Meta API)",
        )

    def test_agent_b_jsonl_promotions_use_measured_results(self):
        config = {
            "dtype": "float16",
            "attn": "sdpa",
            "batch": 4,
            "compile": False,
            "max_new_tokens": 128,
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            (results / "benchmark_real.json").write_text(
                json.dumps(
                    [
                        {
                            "profile_name": "default_fp16_sdpa_b1",
                            "status": "success",
                            "requested_config": {"dtype": "float16", "batch": 1},
                            "docs_per_sec": 2.0,
                        },
                        {
                            "profile_name": "candidate_b4",
                            "status": "success",
                            "requested_config": config,
                            "actual_config": config,
                            "docs_per_sec": 3.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            step = {
                "proposal": {
                    "config": config,
                    "reason": "Try batching",
                    "source": "meta_api",
                },
                "result": {
                    "benchmark": {
                        "profile_name": "candidate_b4",
                        "status": "success",
                        "requested_config": config,
                        "actual_config": config,
                        "docs_per_sec": 3.0,
                    },
                    "validation": {
                        "accepted": True,
                        "docs": [{"token_f1": 0.9, "rouge_l": 0.8, "reasons": []}],
                    },
                },
                "accepted": True,
                "reasons": [],
            }
            (results / "agent_log.json").write_text(
                json.dumps(step) + "\n",
                encoding="utf-8",
            )
            (results / "dashboard_apply_log.jsonl").write_text(
                json.dumps(
                    {
                        "candidate_config": config,
                        "apply_result": {
                            "action": "promoted",
                            "active_config": config,
                            "validation": {
                                "accepted": True,
                                "split": "heldout",
                                "docs": [
                                    {"token_f1": 0.95, "rouge_l": 0.85, "reasons": []}
                                ],
                                "reasons": [],
                            },
                        },
                        "benchmark": {"docs_per_sec": 2.8, "joules_per_doc": 1.1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            data = app.load_dashboard_data(results)
            view = app.build_dashboard_view(data)

        self.assertIn("Meta API", view["agent_rows"][0][1])
        self.assertEqual(view["winner_markdown"].splitlines()[0], "**Chosen profile:** candidate_b4")
        candidate_row = next(row for row in view["profile_rows"] if row[0] == "candidate_b4")
        self.assertEqual(candidate_row[1], "Accepted")
        self.assertEqual(candidate_row[6], "0.900")
        self.assertEqual(candidate_row[7], "0.800")
        self.assertIn("Held-out confirmation: accepted", view["winner_markdown"])
        self.assertIn("0.950", view["winner_markdown"])
        self.assertIn("0.850", view["winner_markdown"])
        self.assertIn("2.800 docs/s", view["winner_markdown"])
        self.assertIn("1.100 J/doc", view["winner_markdown"])

    def test_safe_apply_fails_closed_without_real_results(self):
        self.assertIn("Not applied", app.safe_apply(None))

    def test_module_import_does_not_require_gpu_or_ui_packages(self):
        check = (
            "import sys, app; "
            "assert not {'torch', 'transformers', 'datasets', 'gradio'} & sys.modules.keys()"
        )
        completed = subprocess.run(
            [sys.executable, "-c", check],
            cwd=Path(app.__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
