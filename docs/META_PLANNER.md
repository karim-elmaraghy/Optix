# Meta Experiment Planner (advisory)

Protocol checked against Meta's official documentation on November 4, 2026:
https://dev.meta.ai/docs/cookbook/quickstart-chat-completions and
https://dev.meta.ai/docs/protocols/chat-completions.
The documented model is `muse-spark-1.3`, using Bearer authentication at
`https://api.meta.ai/v1/chat/completions`, a model/messages request, and
`choices[0].message.content`. Responses must finish with `stop` and contain
nonempty text. The transport uses `max_completion_tokens` and default temperature.

Set MODEL_API_KEY securely in the process environment. Never put it in arguments
or evidence files. META_MODEL optionally overrides the requested model.

```powershell
.venv/Scripts/python.exe -m optiloop.meta_planner --ping --output meta-ping.json
.venv/Scripts/python.exe -m optiloop.meta_planner --history actual-v2-tuning-history.json --allowed permitted-configurations.json --output meta-plan.json
```

History must be a JSON list of actual benchmark rows (or optimizer history records
with result.benchmark). Allowed configurations must be a JSON list using the
existing agent contract: dtype, attn, batch, compile, max_new_tokens. Supply tuning
evidence only. The caller must establish the measurements' authenticity; this
command does not certify evidence as authentic V2 GPU results. No example fixture
is shipped as benchmark evidence.

The planner submits the supplied history and allowed configurations, excludes
previously requested configurations, and requires an exact permitted proposal and
nonempty hypothesis. Output includes submitted messages, evidence file hashes,
raw returned proposal text, validated proposal, requested/returned model IDs,
response ID, HTTP status and UTC timestamp. It never overwrites an existing file.
Protect evidence files from sensitive content before submission.

Successful live calls carry source=meta_api. Failures have source=null and no
proposal; there is no rule fallback in this demonstration. The existing optimizer
retains its explicit rule_fallback behavior. A ping proves connectivity only.
An advisory proposal proves neither benchmark execution nor quality approval nor
deployment. No optimizer, validator, or safe-apply gate is changed.

This checkout has no Tunix CLI/workbench or authentic V2 benchmark artifacts.
Workbench integration and a demonstration using that history remain pending their
availability. The sanitized local verification record reports missing credentials
as unverified, never as a successful API call.
