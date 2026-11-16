"""Remove only the four mistakenly imported teacher-training runs."""
import hashlib
import json
from pathlib import Path

import wandb

PROJECT = "chenxheee-sjtu-hpc-center/bridge_mopd"
TARGETS = {
    "bf017936bc01c8c1": "H800/2026-11-13/1.7B_IF教师_同步GRPO",
    "bfba0a4e06f84a0d": "H800/2026-11-14/1.7B_IF教师_异步GRPO",
    "bfca7f8509592508": "H800/2026-11-13/1.7B_law_SFT教师",
    "bff336cc3d026eec": "H800/2026-11-13/1.7B_med_SFT教师",
}
api = wandb.Api()
for run in list(api.runs(PROJECT)):
    if run.id not in TARGETS:
        continue
    if run.name != TARGETS[run.id]:
        raise RuntimeError("Run display name changed; refusing to delete " + run.id)
    run.delete(delete_artifacts=False)
    print("REMOVED", run.id, run.name, flush=True)
api.flush()
remaining = {r.id for r in api.runs(PROJECT)}
assert not remaining.intersection(TARGETS), "Some teacher imports are still present"
records = json.loads(Path("data/index.json").read_text())["runs"]
expected = {"bf" + hashlib.sha256(r["name"].encode()).hexdigest()[:14] for r in records}
assert expected.issubset(remaining), "A retained student import is missing"
print("VERIFIED: teacher imports absent; retained student curves", len(expected), flush=True)
