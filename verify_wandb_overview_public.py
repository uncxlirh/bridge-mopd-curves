"""Read the native saved report and its plotted values without credentials."""
import json
import urllib.request

VARIABLES = {"entity": "chenxheee-sjtu-hpc-center", "project": "bridge_mopd"}


def query(document, variables):
    request = urllib.request.Request("https://api.wandb.ai/graphql",
        headers={"Content-Type": "application/json", "User-Agent": "wandb-overview-public-read"},
        data=json.dumps({"query": document, "variables": variables}).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert not result.get("errors"), result.get("errors")
    return result["data"]


data = query("""query($entity:String!, $project:String!) {
 project(name:$project, entityName:$entity) {
  allViews(viewType:"runs", first:50) { edges { node { displayName spec } } }
 }
}""", VARIABLES)
report = next(x["node"] for x in data["project"]["allViews"]["edges"]
              if x["node"]["displayName"] == "渐进式教师实验总览")
spec = json.loads(report["spec"])
panels = []
def walk(value):
    if isinstance(value, dict):
        if value.get("viewType") == "Run History Line Plot":
            panels.append(value["config"])
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
walk(spec)
assert panels and all(p["xAxis"] in {"plot/step", "train/step"} for p in panels)
evaluations = [p for p in panels if any(y.startswith("evaluation/") for y in p["metrics"])]
assert evaluations and all(p.get("smoothingWeight", 0) == 0 for p in evaluations)
required = {"evaluation/"+k+"_pct" for k in ["math", "code", "medqa", "casehold", "finqa", "instruction_following"]}
assert required <= {m for p in panels for m in p["metrics"]}
assert {"bridge/abs_k1/"+d for d in ["math","code","med","law","fin","if"]} <= {m for p in panels for m in p["metrics"]}
assert "六领域正式评测结果" in json.dumps(spec, ensure_ascii=False)
history_query = """query($entity:String!, $project:String!, $run:String!, $specs:[JSONString!]!) {
 project(name:$project, entityName:$entity) { run(name:$run) { sampledHistory(specs:$specs) } }
}"""
for run, metric in [("bf73d3a9cbb33325", "training/math_accuracy_pct"),
                    ("bff34503e96342d3", "evaluation/macro6_pct")]:
    result = query(history_query, {**VARIABLES, "run": run,
        "specs": [json.dumps({"keys": ["plot/step", metric], "samples": 3})]})
    points = result["project"]["run"]["sampledHistory"][0]
    if isinstance(points, str):
        points = json.loads(points)
    assert points and all(0 <= p[metric] <= 100 for p in points)
    if metric.startswith("evaluation/"):
        assert any(p["plot/step"] == 100 and abs(p[metric]-41.47)<.01 for p in points)
    print("ANONYMOUS REAL CURVE", metric, points, flush=True)
print("VERIFIED NATIVE REPORT", len(panels), "line charts; original training steps; unsmoothed evaluations", flush=True)
