"""Import saved scalar histories into the official W&B service."""
import hashlib
import json
from pathlib import Path

import wandb


def main():
    index = json.loads(Path("data/index.json").read_text())
    uploaded = []
    for record in index["runs"]:
        excluded = ["工程测试", "bridge_smoke", "旧版卷积bug", "badconfig", "补充评测与分析"]
        if any(word in record["name"] for word in excluded):
            continue
        recovered_curve = "4B_TA_MOPD_历史记录" in record["name"]
        if record["kind"] != "训练" and not recovered_curve:
            continue
        metrics = json.loads(Path("data", record["id"] + ".json").read_text())["metrics"]
        metrics = {tag: values for tag, values in metrics.items() if len(values) >= 2}
        rows = {}
        for tag, points in metrics.items():
            for step, value in points:
                rows.setdefault(int(step), {})[tag] = value
        if len(rows) < 2:
            continue
        run_id = "bf" + hashlib.sha256(record["name"].encode()).hexdigest()[:14]
        tags = ["backfill", record["machine"], record["kind"]]
        if recovered_curve:
            tags.append("recovered-evaluation-curve")
        run = wandb.init(
            project="bridge_mopd", id=run_id, name=record["name"],
            resume="allow", mode="online", group=record["machine"],
            tags=tags, job_type="historical-metrics-import", save_code=False,
            config={"snapshot_time": index["updated"], "source": "saved TensorBoard scalar histories",
                    "machine": record["machine"], "record_type": record["kind"],
                    "note": "Use train/step for the original training step. Import timestamps are not training timestamps."},
            settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_meta=True,
                                    x_disable_stats=True, console="off", init_timeout=90),
        )
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")
        next_step = run.step
        for step, values in sorted(rows.items()):
            if step >= next_step:
                run.log({**values, "train/step": step}, step=step)
        url = run.url
        run.finish()
        uploaded.append({"name": record["name"], "run_id": run_id, "url": url, "last_step": max(rows)})
        Path("uploaded_runs.json").write_text(json.dumps(uploaded, ensure_ascii=False, indent=2))
        print("UPLOADED", record["name"], url, flush=True)
    print("UPLOAD COMPLETE:", len(uploaded), "records", flush=True)


if __name__ == "__main__":
    main()
