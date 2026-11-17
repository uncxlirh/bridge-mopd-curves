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
panels, bars = [], []
def walk(value):
    if isinstance(value, dict):
        if value.get("viewType") == "Bar Chart":
            bars.append(value["config"])
        if value.get("viewType") == "Run History Line Plot":
            panels.append(value["config"])
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
walk(spec)
assert not panels, "The report should not contain diagnostic plot walls"
assert len(bars)==2
assert all(not m.startswith("summary_metrics.") for p in bars for m in p["metrics"])
assert {m for p in bars for m in p["metrics"]} == {"checkpoint/40/gain/evaluation/macro6_pct", "checkpoint/100/gain/evaluation/macro6_pct"}
serialized=json.dumps(spec,ensure_ascii=False)
assert all(t in serialized for t in ["领域变化","完整分项成绩","**47.04**","**61.90**","**57.89**","**67.60**"])
required = {"evaluation/"+k+"_pct" for k in ["math", "code", "medqa", "casehold", "finqa", "instruction_following"]}
summary_query = """query($entity:String!, $project:String!, $run:String!) {
 project(name:$project, entityName:$entity) { run(name:$run) { summaryMetrics } }
}"""
values=query(summary_query,{**VARIABLES,"run":"bf3adff1d85d08f6"})["project"]["run"]["summaryMetrics"]
if isinstance(values,str): values=json.loads(values)
for key,expected in [("checkpoint/100/gain/evaluation/casehold_pct",1.311),
                     ("checkpoint/100/gain/evaluation/instruction_following_pct",-1.239372),
                     ("checkpoint/100/matched_gain/evaluation/casehold_pct",2.371093)]:
    assert abs(values[key]-expected)<.0001, (key,values.get(key))
print("ANONYMOUS CHECKPOINT BARS",len(bars),"panels; actual gains include positive and negative values",flush=True)
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
for metric in sorted(required):
    result = query(history_query, {**VARIABLES, "run": "bf73d3a9cbb33325",
        "specs": [json.dumps({"keys": ["plot/step", metric], "samples": 3})]})
    points = result["project"]["run"]["sampledHistory"][0]
    if isinstance(points, str):
        points = json.loads(points)
    assert points and all(0 <= p[metric] <= 100 for p in points), metric
    print("ANONYMOUS DOMAIN EVALUATION", metric, points, flush=True)
for domain in ["math", "code", "med", "law", "fin", "if"]:
    metric = "bridge/abs_k1/"+domain
    result = query(history_query, {**VARIABLES, "run": "bf3adff1d85d08f6",
        "specs": [json.dumps({"keys": ["train/step", metric], "samples": 3})]})
    points = result["project"]["run"]["sampledHistory"][0]
    if isinstance(points, str):
        points = json.loads(points)
    assert points and all(p[metric] >= 0 for p in points), metric
    print("ANONYMOUS DOMAIN TRAINING", metric, points, flush=True)
for run, metric, expected in [
    ("bf3adff1d85d08f6", "evaluation/macro6_pct", 47.040252),
    ("bf3adff1d85d08f6", "recovery/mean_pct", 61.896325),
    ("bf3adff1d85d08f6", "recovery/worst_pct", 18.453104),
    ("bff34503e96342d3", "recovery/worst_pct", -190.295344),
]:
    result = query(history_query, {**VARIABLES, "run": run,
        "specs": [json.dumps({"keys": ["plot/step", metric], "samples": 100})]})
    points = result["project"]["run"]["sampledHistory"][0]
    if isinstance(points, str):
        points = json.loads(points)
    assert any(p["plot/step"]==100 and abs(p[metric]-expected)<.0001 for p in points), (metric,points)
    print("ANONYMOUS FINAL AND RECOVERY", run, metric, points, flush=True)
print("VERIFIED NATIVE REPORT", len(panels), "line charts; original training steps; unsmoothed evaluations", flush=True)
