"""Read-only: dump the W&B report spec (blocks, panels, metrics, run sets) to report_dump.json."""
import json
import os
import wandb

api = wandb.Api(timeout=60)
out = {}
q = """query($id: ID!) { view(id: $id) { id displayName description updatedAt spec } }"""
r = api.client.execute(wandb.apis.public.gql(q), variable_values={"id": "VmlldzoxODMxNzAwNA=="})
v = r["view"]
out["title"], out["description"], out["updatedAt"] = v["displayName"], v["description"], v["updatedAt"]
spec = json.loads(v["spec"]) if isinstance(v["spec"], str) else v["spec"]
out["spec"] = spec
try:
    import wandb_workspaces.reports.v2 as wr
    rep = wr.Report.from_url("https://wandb.ai/chenxheee-sjtu-hpc-center/bridge_mopd/reports/x--VmlldzoxODMxNzAwNA==")
    blocks = []
    for b in rep.blocks:
        d = {"type": type(b).__name__}
        if hasattr(b, "text"):
            d["text"] = str(b.text)[:2000]
        if hasattr(b, "panels"):
            d["panels"] = [{"type": type(p).__name__, "title": getattr(p, "title", None),
                            "x": getattr(p, "x", None), "y": getattr(p, "y", None)} for p in b.panels]
            d["runsets"] = [{"name": getattr(rs, "name", None), "filters": str(getattr(rs, "filters", ""))[:300]}
                            for rs in getattr(b, "runsets", [])]
        blocks.append(d)
    out["blocks"] = blocks
except Exception as e:
    out["blocks_error"] = repr(e)
json.dump(out, open("report_dump.json", "w"), ensure_ascii=False, indent=1, default=str)
print("blocks:", len(out.get("blocks", [])), "spec bytes:", len(json.dumps(spec)))
