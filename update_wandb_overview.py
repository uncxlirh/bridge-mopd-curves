"""Configure native W&B workspaces/reports and clearly defined metric aliases."""
import hashlib
import argparse
import json
import math
from pathlib import Path

import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces.expr import Metric

ENTITY = "chenxheee-sjtu-hpc-center"
PROJECT = "bridge_mopd"
BASE = f"https://wandb.ai/{ENTITY}/{PROJECT}"
TITLE = "渐进式教师实验总览"


def public_url(url):
    # SDK 0.4.13 returns Forge URLs without its required /wandb route.
    # The legacy public host redirects to the correct frontend route.
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    if parts.netloc == "forge.coreweave.com":
        path = parts.path.removeprefix("/wandb")
        return urlunsplit(("https", "wandb.ai", path, parts.query, parts.fragment))
    return url


def run_id(record):
    return "bf" + hashlib.sha256(record["name"].encode()).hexdigest()[:14]


def panel(title, key, x="plot/step", percent=False, smooth=False, position=0):
    return wr.LinePlot(
        title=title, x=x, y=[key], title_x="训练步数", title_y="%" if percent else None,
        range_y=(0, 100) if percent else (None, None),
        smoothing_type="exponential" if smooth else "none", smoothing_factor=.6 if smooth else 0,
        smoothing_show_original=True, ignore_outliers=False, aggregate=False,
        max_runs_to_show=8, legend_position="south", font_size="medium",
        layout=wr.Layout(x=(position % 2) * 12, y=(position // 2) * 8, w=12, h=8),
    )


def settings(records, report=False):
    cls = wr.RunSettings if report else ws.RunSettings
    return {run_id(r): cls(color=r["overview"]["color"], disabled=False) for r in records}


def runset(records, title):
    return wr.Runset(entity=ENTITY, project=PROJECT, name=title,
                     filters=[Metric("ID").isin([run_id(r) for r in records])],
                     run_settings=settings(records, report=True),
                     visible_columns=["run:displayName", "config:recorded_step.value", "config:budget_steps.value",
                                      "summary:evaluation/checkpoint_step", "summary:evaluation/macro6_pct",
                                      "summary:recovery/mean_pct", "summary:recovery/worst_pct"],
                     lock_columns=True)


def grid(records, title, panels):
    if len(panels) == 1:
        panels[0].layout.w = 24
    return wr.PanelGrid(runsets=[runset(records, title)], panels=panels)


def checkpoint_snapshots(analysis):
    baseline = next(r for r in analysis["records"] if r["key"] == "q17_vanilla100")
    dn_baseline = next(r for r in analysis["records"] if r["key"] == "q17_vanilla_dn100")
    result = {}
    for record in analysis["records"]:
        values = {}
        for step, row in record["checkpoints"].items():
            for metric, value in row.items():
                if metric.startswith(("evaluation/", "recovery/", "evaluation_detail/")):
                    values[f"checkpoint/{step}/{metric}"] = value
            ref = baseline["checkpoints"][step]
            matched = dn_baseline["checkpoints"][step] if record["key"] in {"q17_best_dn100", "q17_vanilla_dn100"} else ref
            for metric in ["evaluation/"+d+"_pct" for d in ["math","code","medqa","casehold","finqa","instruction_following","macro6"]] + ["recovery/mean_pct","recovery/worst_pct"]:
                values[f"checkpoint/{step}/gain/{metric}"] = round(row[metric]-ref[metric],6)
                values[f"checkpoint/{step}/matched_gain/{metric}"] = round(row[metric]-matched[metric],6)
        result[record["data_id"]] = values
    return result


def refresh_aliases(api, records):
    analysis = json.loads(Path("data/analysis.json").read_text())
    snapshots = checkpoint_snapshots(analysis)
    labels = {r["data_id"]:r["label"] for r in analysis["records"]}
    for record in records:
        path = f"{ENTITY}/{PROJECT}/{run_id(record)}"
        public = api.run(path)
        public.name = labels.get(record["id"], record.get("display_name", record["name"]))
        public.config.update(record.get("overview", {}))
        public.config["original_run_label"] = record["name"]
        public.config["primary_training_metric"] = "bridge/abs_k1/<domain>"
        public.config["supplementary_evaluation_metrics"] = ["recovery/mean_pct", "recovery/worst_pct"]
        public.config["primary_evaluation_metric"] = "evaluation/macro6_pct" if record.get("overview",{}).get("scale")=="1.7B" else "evaluation/macro3_pct"
        public.update()
        content = json.loads(Path("data", record["id"] + ".json").read_text())
        derived = content.get("derived", {})
        rows = {}
        for key, points in derived.items():
            for step, value in points:
                rows.setdefault(str(step), {})[key] = value
        # History is authoritative: summary can be reshaped by another importer.
        previous, raw_previous = {}, {}
        for item in public.scan_history(page_size=1000):
            for axis, target, prefixes in [("plot/step", previous, ("training/", "evaluation/", "diagnostics/", "schedule/", "performance/", "recovery/", "comparison/", "evaluation_detail/")),
                                            ("train/step", raw_previous, ("eval/",))]:
                value = item.get(axis)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                cells = {k:v for k,v in item.items() if k.startswith(prefixes)
                         and isinstance(v, (int,float)) and math.isfinite(v)}
                target.setdefault(str(int(value)), {}).update(cells)
        changes = {}
        for step, values in rows.items():
            changed = {k: v for k, v in values.items() if previous.get(step, {}).get(k) != v}
            if changed:
                changes[int(step)] = changed
        # Earlier imports incorrectly dropped single-checkpoint evaluation tags.
        # Read existing raw evaluation cells once, then append only missing or
        # changed cells; do not replay the original training history.
        raw_rows = {}
        for key, points in content["metrics"].items():
            if key.startswith("eval/"):
                for step, value in points:
                    raw_rows.setdefault(str(step), {})[key] = value
        raw_changes = {int(step): {k:v for k,v in values.items() if raw_previous.get(step, {}).get(k) != v}
                       for step,values in raw_rows.items()}
        raw_changes = {s:v for s,v in raw_changes.items() if v}
        if changes or raw_changes:
            run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(record), resume="must", mode="online",
                             settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_meta=True,
                                                     x_disable_stats=True, console="off"))
            run.define_metric("plot/step")
            for prefix in ["training/*", "evaluation/*", "diagnostics/*", "schedule/*", "performance/*", "recovery/*", "comparison/*", "evaluation_detail/*"]:
                run.define_metric(prefix, step_metric="plot/step", step_sync=False)
            for step, values in sorted(raw_changes.items()):
                run.log({"train/step": step, **values})
            for step, values in sorted(changes.items()):
                run.log({"plot/step": step, **values})
            run.summary["train/step"] = int(run.summary.get("raw_import_last_source_step", record["lastStep"]))
            run.finish()
        if record["id"] in snapshots:
            public = api.run(path)
            public.summary.update(snapshots[record["id"]])
        print("OVERVIEW METRICS", record.get("display_name"), "changed steps", len(changes), flush=True)


def construct(index, refs=None):
    analysis = json.loads(Path("data/analysis.json").read_text())
    notes = analysis["notes"]
    records = [r for r in index["runs"] if r.get("overview", {}).get("scale") == "1.7B"
               and r["overview"].get("method") in {"plain", "progressive"}]
    assert len(records) == 6, "Missing formal 1.7B experiment metadata"
    current = [r for r in records if r["overview"]["normalized"]]
    established = [r for r in records if not r["overview"]["normalized"] and "注入" not in r["name"]]
    injected = [r for r in records if "注入" in r["name"]]
    primary = established + current
    domains = [("数学", "math"), ("代码", "code"), ("医学", "medqa"),
               ("法律", "casehold"), ("金融", "finqa"), ("指令遵循", "instruction_following")]
    train_domains = [("数学","math"),("代码","code"),("医学","med"),("法律","law"),("金融","fin"),("指令遵循","if")]
    def snapshot_bar(title, key, position=0, step=100, bounds=(None,None), unit="百分点"):
        return wr.BarPlot(title=title,metrics=[f"checkpoint/{step}/{key}"],
                          orientation="h",range_x=bounds,title_x=unit,
                          aggregate=False,max_runs_to_show=8,max_bars_to_show=8,font_size="medium",
                          layout=wr.Layout(x=(position%2)*12,y=(position//2)*8,w=12,h=8))
    overall = [snapshot_bar("100 步 · 六域均分增益", "gain/evaluation/macro6_pct"),
               snapshot_bar("100 步 · 平均恢复率增益", "gain/recovery/mean_pct",1),
               snapshot_bar("100 步 · 最差领域恢复率增益", "gain/recovery/worst_pct",2),
               snapshot_bar("40 步 · 六域均分增益", "gain/evaluation/macro6_pct",3,step=40)]
    def domain_gains(step):
        baseline = next(r for r in analysis["records"] if r["key"]=="q17_vanilla100")["checkpoints"][str(step)]
        rows = [r["checkpoints"][str(step)] for r in analysis["records"] if r["key"]!="q17_adaptnc_inj100"]
        maximum = max(abs(row["evaluation/"+key+"_pct"]-baseline["evaluation/"+key+"_pct"]) for row in rows for title,key in domains)
        limit = math.ceil(maximum*1.15*2)/2
        return [snapshot_bar(f"{title} · {step} 步相对普通 MOPD", "gain/evaluation/"+key+"_pct",i,step=step,bounds=(-limit,limit)) for i,(title,key) in enumerate(domains)]
    breakdown = domain_gains(100)
    early_breakdown = domain_gains(40)
    recoveries = [snapshot_bar(title+" · 100 步恢复率", "recovery/"+key+"_pct",i,unit="恢复率 %")
                  for i,(title,key) in enumerate(domains)]
    domain_training = [panel(title+"训练 · 与当前教师的差异（不是正确率）", "bridge/abs_k1/"+key,
                             x="train/step", smooth=True, position=i) for i,(title,key) in enumerate(train_domains)]
    diagnostics = [(snapshot_bar(title,key,position=i,unit="%") if key.startswith("evaluation/") else panel(title,key,x=x,position=i)) for i,(title,key,x) in enumerate([
        ("数学训练题正确率 · 仅数学训练题", "training/math_accuracy_pct", "plot/step"),
        ("医疗 · 本步实际使用的教师 λ", "schedule/used_med_lambda", "plot/step"),
        ("法律 · 本步实际使用的教师 λ", "schedule/used_law_lambda", "plot/step"),
        ("IFBench · prompt-loose %", "evaluation/ifbench_pct", "plot/step"),
        ("IFEval · prompt-strict %", "evaluation/ifeval_pct", "plot/step"),
        ("梯度范数 · 裁剪前", "diagnostics/gradient_norm", "plot/step"),
        ("蒸馏 loss · 目标不同不能当能力排名", "diagnostics/distillation_abs_loss", "plot/step"),
        ("训练平均回答长度 · token", "diagnostics/response_tokens", "plot/step"),
        ("训练回答写满上限 · %", "diagnostics/truncated_pct", "plot/step"),
        ("每步耗时 · 秒", "performance/step_seconds", "plot/step"),
    ])]
    dn = [panel(title+" · DN 实际乘子", "dn/w/"+key, x="train/step", position=i)
          for i,(title,key) in enumerate(train_domains)]
    styles = [snapshot_bar(title+" · 100 步评测平均回答长度", "evaluation_detail/"+key+"_response_tokens",position=i,unit="token")
              for i,(title,key) in enumerate([("医学","medqa"),("法律","casehold"),("金融","finqa"),("IFBench","ifbench"),("IFEval","ifeval")])]
    styles.append(snapshot_bar("IFEval · 100 步评测截断率", "evaluation_detail/ifeval_truncated_pct",position=5,unit="%"))
    guide = notes["conclusion"]
    workspace = ws.Workspace(entity=ENTITY,project=PROJECT,name="1.7B 渐进式教师主对照",
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="plot/step",smoothing_type="none",smoothing_weight=0,max_runs=8,ignore_outliers=False),
        runset_settings=ws.RunsetSettings(filters=[Metric("ID").isin([run_id(r) for r in primary])],run_settings=settings(primary),
            pinned_columns=["run:displayName","config:recorded_step.value","summary:evaluation/checkpoint_step", "summary:evaluation/macro6_pct", "summary:recovery/mean_pct", "summary:recovery/worst_pct"]),
        sections=[
            ws.Section(name="独立检查点 · 相对普通 MOPD 的增益",is_open=True,panels=overall),
            ws.Section(name="六领域正式评测 · 100 步增减（百分点）",is_open=True,panels=breakdown),
            ws.Section(name="六领域正式评测 · 40 步增减（百分点）",is_open=False,panels=early_breakdown),
            ws.Section(name="六领域恢复率 · 100 步独立比较",is_open=False,panels=recoveries),
            ws.Section(name="六领域训练 · 与当前教师的差异",is_open=False,panels=domain_training),
            ws.Section(name="教师推进与训练诊断",is_open=False,panels=diagnostics),
            ws.Section(name="DN 按领域乘子",is_open=False,panels=dn),
        ])
    final = ["| 100 步实验 | 数学 | 代码 | 医学 | 法律 | 金融 | IFBench | IFEval | 六域均分 | 平均恢复率 | 最差恢复率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    early = ["| 40 步实验 | 六域均分 | 比同检查点普通组高 | 平均恢复率 | 最差恢复率 |",
             "|---|---:|---:|---:|---:|"]
    progress = ["| 实验 | 训练进度 | 最新完整评测检查点 |","|---|---:|---:|"]
    for r in analysis["records"]:
        v=r["checkpoints"]["100"]
        values=[v["evaluation/"+k+"_pct"] for k in ["math","code","medqa","casehold","finqa","ifbench","ifeval"]]
        values += [v["evaluation/macro6_pct"],v["recovery/mean_pct"],v["recovery/worst_pct"]]
        final.append("| "+r["label"]+" | "+" | ".join(f"{n:.2f}" for n in values)+" |")
        progress.append(f"| {r['label']} | {r['last_step']}/100 | {max(map(int,r['checkpoints']))} |")
        if r["key"]!="q17_adaptnc_inj100":
            v=r["checkpoints"]["40"]
            early.append(f"| {r['label']} | {v['evaluation/macro6_pct']:.2f} | {v['delta/evaluation/macro6_pct']:+.2f} | {v['recovery/mean_pct']:.2f} | {v['recovery/worst_pct']:.2f} |")
    references=["| 固定参考端点 | "+" | ".join(t for t,k in domains)+" |","|---|"+"---:|"*6]
    for kind,title in [("base","原始学生"),("teacher","各域教师组合")]:
        references.append("| "+title+" | "+" | ".join(f"{analysis['reference'][kind][k]:.4f}" for t,k in domains)+" |")
    summary = ["| 第 100 步主对照 | 六域原始均分 | 平均恢复率 | 最差恢复率 |",
               "|---|---:|---:|---:|"]
    raw = ["| 100 步实验 | 数学 | 代码 | 医学 | 法律 | 金融 | IFBench | IFEval |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in analysis["records"]:
        v=r["checkpoints"]["100"]
        if r["key"] in {"q17_vanilla100","q17_vanilla_dn100","q17_adaptnc100","q17_best_dn100"}:
            summary.append(f"| {r['label']} | {v['evaluation/macro6_pct']:.2f} | {v['recovery/mean_pct']:.2f} | {v['recovery/worst_pct']:.2f} |")
        raw.append("| "+r["label"]+" | "+" | ".join(f"{v['evaluation/'+k+'_pct']:.2f}" for k in ["math","code","medqa","casehold","finqa","ifbench","ifeval"])+" |")
    compact = ["| 实验 | 40 步原始均分 | 100 步原始均分 | 100 步平均恢复率 | 100 步最差恢复率 |",
               "|---|---:|---:|---:|---:|"]
    for r in analysis["records"]:
        v=r["checkpoints"]["100"]; v40=r["checkpoints"]["40"]
        compact.append(f"| {r['label']} | {v40['evaluation/macro6_pct']:.2f} | {v['evaluation/macro6_pct']:.2f} | {v['recovery/mean_pct']:.2f} | {v['recovery/worst_pct']:.2f} |")
    blocks=[
        wr.P("2026 年 11 月 17 日更新。六组实验已完成 100 步训练和评测。"),
        wr.H1("为什么看这些指标"),wr.MarkdownBlock(notes["metric_reason"]),
        wr.H1("结果"),wr.MarkdownBlock("\n".join(compact)),
        wr.H2("正向结果"),wr.P(notes["positive_short"]),
        wr.H2("负向结果"),wr.P(notes["negative_short"]),
        wr.H2("基本持平"),wr.P(notes["flat_short"]),
        wr.H2("六领域正式评测结果"),wr.P(notes["bar_guide"]),
        grid(primary,"100 步 · 相对普通 MOPD 的各域增减",breakdown),
        wr.H2("原始成绩表",collapsed_blocks=[wr.MarkdownBlock("\n".join(raw))]),
        wr.H1("渐进教师怎么算"),wr.MarkdownBlock(notes["algorithm_short"]),wr.P(notes["algorithm_actual"]),
        wr.H2("DN 归一化"),wr.P(notes["dn_short"]),
        wr.H2("定标计算细节",collapsed_blocks=[wr.MarkdownBlock(notes["algorithm_calibration"])]),
        wr.H2("恢复率和最差领域",collapsed_blocks=[grid(primary,"独立检查点 · 整体增益",overall),grid(primary,"100 步 · 六域恢复率",recoveries)]),
        wr.H2("40 步各领域结果",collapsed_blocks=[wr.MarkdownBlock("\n".join(early)),grid(primary,"40 步 · 相对普通 MOPD 的各域增减",early_breakdown)]),
        wr.H2("同 DN 配方对照",collapsed_blocks=[grid(current,"普通＋DN 对渐进＋DN",[snapshot_bar("100 步 · 同 DN 配方均分增益","matched_gain/evaluation/macro6_pct"),snapshot_bar("100 步 · 同 DN 配方最差恢复率增益","matched_gain/recovery/worst_pct",1)])]),
        wr.H2("训练诊断",collapsed_blocks=[wr.P(notes["critic"]),grid(primary,"实际教师与训练指标",diagnostics),grid(primary,"六领域训练过程",domain_training),grid(current,"DN 逐领域乘子",dn)]),
        wr.H2("回答长度和截断率",collapsed_blocks=[grid(records,"正式评测回答长度与截断率",styles)]),
        wr.H2("评测口径",collapsed_blocks=[wr.P("数学/代码：DN-MOPD 8K、关闭思考；其余五项：IM-MOPD avg@3、开启思考。IFBench 为 prompt-loose，IFEval 为 prompt-strict。"),wr.P(notes["definitions"]),wr.MarkdownBlock("\n".join(references)),wr.P(notes["reference_note"])]),
    ]
    archives=[r for r in index["runs"] if r.get("overview",{}).get("scale")=="2B"]
    blocks.append(wr.H2("2B 历史实验 · 不与 1.7B 混比",collapsed_blocks=[wr.P("历史组起点、机器与领域集合不同，三域均分不能与六域均分直接比较。"),grid(archives,"2B 历史对照",[panel("历史三领域均分","evaluation/macro3_pct",percent=True)])]))
    report=wr.Report(entity=ENTITY,project=PROJECT,title=TITLE,description="指标、结果、渐进教师算法",width="fluid",blocks=blocks)
    return workspace,report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--views-only",action="store_true")
    args=parser.parse_args()
    index=json.loads(Path("data/index.json").read_text())
    if not args.views_only:
        api=wandb.Api()
        refresh_aliases(api,index["runs"])
    workspace,report=construct(index)
    file=Path("dashboard_refs.json")
    refs=json.loads(file.read_text()) if file.exists() else {}
    if refs.get("workspace"):
        old=ws.Workspace.from_url(refs["workspace"])
        old.name,old.sections,old.settings,old.runset_settings=workspace.name,workspace.sections,workspace.settings,workspace.runset_settings
        workspace=old
    workspace.save()
    refs["workspace"]=public_url(workspace.url)
    file.write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    print("DASHBOARD WORKSPACE",refs["workspace"],flush=True)
    if refs.get("report"):
        old=wr.Report.from_url(refs["report"]);old.title,old.description,old.blocks,old.width=report.title,report.description,report.blocks,report.width
        report=old
    report.save()
    refs["report"]=public_url(report.url)
    file.write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    print("DASHBOARD REPORT",refs["report"],flush=True)
    # Reload native saved objects to verify that filters and chart definitions survived saving.
    loaded=ws.Workspace.from_url(workspace.url)
    assert len(loaded.sections)==7
    loaded_report=wr.Report.from_url(report.url)
    assert loaded_report.title==TITLE
    print("VERIFIED NATIVE W&B OVERVIEW",flush=True)


if __name__=="__main__":
    main()
