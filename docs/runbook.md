# Runbook

Stub at M1; written properly at M10 once there is an operational surface. It
will cover: what to do when the grounding verifier starts refusing everything,
how to read a Langfuse trace for a bad memo, how to clear and rebuild the
index, cost-spike response, and an explicit "what NOT to do" section.

## What exists today

### Start the local stack
```powershell
docker compose up -d
```

### Run the test suite
```powershell
$env:PYTHONPATH = "src"
python -m pytest
```

### Fast, deterministic run
```powershell
$env:PC_CI_MODE = "1"
```
