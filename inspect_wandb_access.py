"""Read actual W&B schema and visibility-related frontend constants."""
import base64
import json
import os
import re
import urllib.request
import urllib.error

authorization = "Basic " + base64.b64encode(("api:" + os.environ["WANDB_API_KEY"]).encode()).decode()
document = """query { __schema { types { name kind enumValues { name description } } }
  __type(name:"UpsertModelInput") { inputFields { name description type { name kind } } }
}"""
req = urllib.request.Request("https://api.wandb.ai/graphql", headers={"Content-Type":"application/json", "Authorization":authorization},
                             data=json.dumps({"query":document}).encode())
try:
    response = urllib.request.urlopen(req, timeout=30)
except urllib.error.HTTPError as error:
    response = error
with response:
    result = json.load(response)
if result.get("errors"):
    print("INTROSPECTION ERRORS", result["errors"])
else:
    for field in result["data"]["__type"]["inputFields"]:
        if any(s in field["name"].lower() for s in ["access", "publish", "benchmark"]):
            print("INPUT FIELD", json.dumps(field))
    for typ in result["data"]["__schema"]["types"]:
        if typ["kind"] == "ENUM" and any(s in typ["name"].lower() for s in ["access", "project", "visibility"]):
            print("ENUM", json.dumps(typ))
url = "https://wandb.ai/chenxheee-sjtu-hpc-center/bridge_mopd/overview"
try:
    with urllib.request.urlopen(url, timeout=20) as response:
        html = response.read().decode()
    scripts = re.findall(r'<script[^>]+src=["\x27]([^"\x27]+)', html)
    print("FRONTEND SCRIPTS", json.dumps(scripts))
except Exception as error:
    print("FRONTEND READ", type(error).__name__)
