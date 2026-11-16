"""Make only the requested W&B project publicly readable and verify curves."""
import base64
import json
import os
import urllib.request
import urllib.error

ENTITY = "chenxheee-sjtu-hpc-center"
PROJECT = "bridge_mopd"


def query(document, variables, authenticated=True):
    headers = {"Content-Type": "application/json", "User-Agent": "wandb-project-visibility"}
    if authenticated:
        credential = base64.b64encode(("api:" + os.environ["WANDB_API_KEY"]).encode()).decode()
        headers["Authorization"] = "Basic " + credential
    request = urllib.request.Request("https://api.wandb.ai/graphql", headers=headers,
                                     data=json.dumps({"query": document, "variables": variables}).encode())
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"]))
    return result["data"]


variables = {"entity": ENTITY, "project": PROJECT}
read = """query($entity:String!, $project:String!) {
  project(name:$project, entityName:$entity) { name access }
}"""
before = query(read, variables)["project"]
assert before, "Project not found"
print("BEFORE", before, flush=True)
if before["access"] != "PUBLIC":
    mutation = """mutation($entity:String!, $project:String!) {
      upsertModel(input:{entityName:$entity, name:$project, access:"PUBLIC"}) {
        model { name access }
      }
    }"""
    result = query(mutation, variables)
    print("UPDATED", result["upsertModel"]["model"], flush=True)
after = query(read, variables, authenticated=False)["project"]
assert after and after["access"] == "PUBLIC", after
print("ANONYMOUS PROJECT ACCESS", after, flush=True)
history = """query($entity:String!, $project:String!, $run:String!, $specs:[JSONString!]!) {
  project(name:$project, entityName:$entity) {
    run(name:$run) { sampledHistory(specs:$specs) }
  }
}"""
v = {**variables, "run": "bfcfa67bdec7754c",
     "specs": [json.dumps({"keys": ["_step", "train/step", "actor/distillation/abs_loss"], "samples": 3})]}
points = query(history, v, authenticated=False)["project"]["run"]["sampledHistory"][0]
if isinstance(points, str):
    points = json.loads(points)
assert points, "Anonymous history is empty"
print("ANONYMOUS CURVE POINTS", len(points), flush=True)
print("VERIFIED PUBLIC READ ACCESS", f"https://wandb.ai/{ENTITY}/{PROJECT}", flush=True)
