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
    return wr.PanelGrid(runsets=[runset(records, title)], panels=panels, hide_run_sets=True)


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
    labels = {r["data_id"]:"1.7B "+analysis["presentation"]["labels"][r["key"]] for r in analysis["records"]}
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
    a=json.loads(Path("data/analysis.json").read_text())
    n,ui=a["notes"],a["presentation"]
    records=[r for r in index["runs"] if r.get("overview",{}).get("scale")=="1.7B"
             and r["overview"].get("method") in {"plain","progressive"}]
    assert len(records)==6
    primary=[r for r in records if "注入" not in r["name"]]
    stages=[wr.BarPlot(title=f"{step} 步",metrics=[f"checkpoint/{step}/gain/evaluation/macro6_pct"],
                       orientation="h",range_x=(-1,2),title_x="相对普通 MOPD（百分点）",
                       aggregate=False,max_runs_to_show=5,max_bars_to_show=5,font_size="medium",
                       layout=wr.Layout(x=i*12,y=0,w=12,h=6)) for i,step in enumerate([40,100])]
    teachers=[panel(title,f"schedule/used_{domain}_lambda",position=i) for i,(title,domain) in enumerate([("医疗教师 λ","med"),("法律教师 λ","law")])]
    for p in teachers:p.layout.h=6
    core_training=[panel(title,key,position=i) for i,(title,key) in enumerate([
        ("学生蒸馏 loss", "diagnostics/distillation_abs_loss"),
        ("学生梯度范数", "diagnostics/gradient_norm"),
        ("训练回答长度", "diagnostics/response_tokens"),
        ("训练截断比例 %", "diagnostics/truncated_pct"),
    ])]
    domain_signals=[panel(title+" · 学生与当前教师差异", "bridge/abs_k1/"+d,x="train/step",position=i)
                    for i,(title,d) in enumerate([("数学","math"),("代码","code"),("医学","med"),("法律","law"),("金融","fin"),("IF","if")])]
    dn_panels=[panel(title+" · DN 乘子", "dn/w/"+d,x="train/step",position=i)
               for i,(title,d) in enumerate([("数学","math"),("代码","code"),("医学","med"),("法律","law"),("金融","fin"),("IF","if")])]
    for p in core_training+domain_signals+dn_panels:p.layout.h=6
    workspace=ws.Workspace(entity=ENTITY,project=PROJECT,name="1.7B 渐进式教师主对照",auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="plot/step",smoothing_type="none",smoothing_weight=0,max_runs=5,ignore_outliers=False),
        runset_settings=ws.RunsetSettings(filters=[Metric("ID").isin([run_id(r) for r in primary])],run_settings=settings(primary),
            pinned_columns=["run:displayName","summary:evaluation/macro6_pct","summary:recovery/mean_pct","summary:recovery/worst_pct"]),
        sections=[ws.Section(name="学生训练 · 总览",is_open=True,panels=core_training),
                  ws.Section(name="阶段对比 · 正式评测",is_open=True,panels=stages),
                  ws.Section(name="各领域蒸馏信号",is_open=False,panels=domain_signals),
                  ws.Section(name="教师进度",is_open=False,panels=teachers),
                  ws.Section(name="DN 各领域乘子",is_open=False,panels=dn_panels)])
    refs_table=["| 参考 | 数学 | 代码 | 医学 | 法律 | 金融 | IF |","|---|---:|---:|---:|---:|---:|---:|"]
    domains=["math","code","medqa","casehold","finqa","instruction_following"]
    for key,title in [("base","起点"),("teacher","教师")]:
        refs_table.append("| "+title+" | "+" | ".join(f"{a['reference'][key][d]:.2f}" for d in domains)+" |")
    blocks=[
        wr.P("Bridge-MOPD · Qwen3-1.7B · 2026 年 11 月 17 日"),
        wr.MarkdownBlock(ui["status"]),
        wr.MarkdownBlock(n["metric_reason"]),
        wr.H1("总览"),wr.P("学生训练：loss 看与当前教师的差异，梯度范数看更新幅度，回答长度和截断率看生成行为。"),
        grid(primary,"学生训练 · 关键指标",core_training),
        wr.P("loss 为回答 token 上的平均绝对 log 概率差；梯度范数为裁剪前的值；截断率是回答写满长度上限的比例。"),
        wr.H1("评测"),wr.MarkdownBlock(ui["summary"]),
        wr.P("粗体为每列最高值。渐进表示不强切，DN 表示反馈按领域归一化。恢复率单位为 %。"),
        wr.MarkdownBlock(ui["positive"]+"\n\n"+ui["negative"]+"\n\n"+ui["flat"]),
        wr.H2("阶段对比"),grid(primary,"40 步与 100 步 · 正式评测",stages),
        wr.H2("领域变化"),wr.P("100 步的普通 MOPD 与渐进＋DN。粗体为两组中较高的成绩。"),wr.MarkdownBlock(ui["pair"]),
        wr.H2("完整分项成绩",collapsed_blocks=[wr.MarkdownBlock(ui["raw"]),wr.P("粗体为每个 benchmark 的最高分。")]),
        wr.H1("设置与算法"),
        wr.H2("学生训练设置"),wr.MarkdownBlock(n["student_training"]),
        wr.H2("六组实验的区别"),wr.MarkdownBlock(n["student_groups"]),
        wr.H2("批次构成"),wr.P("示例：渐进＋DN 第 100 步，96 题，每题 4 个回答。"),wr.MarkdownBlock(ui.get("batch_composition","")),
        wr.H2("方法与升档"),wr.P("中间教师由固定起点和领域专家逐参数插值，学生按领域逐步向完整专家学习："),
        wr.MarkdownBlock("```text\nT_d(λ) = (1−λ)θ₀ + λθ_d\nλ = 0.125、0.25、0.5、0.75、1\n```"),wr.MarkdownBlock(ui["algorithm"]),
        wr.P("这轮最终数学、代码、金融和 IF 使用完整教师；医疗、法律停在 λ=0.25。旧版第 81 步统一强切，新版取消了这个固定期限。"),
        wr.H2("损失、升档与归一化"),wr.P(n["student_loss"]),wr.MarkdownBlock(n["student_control"]),
        wr.H2("运行与恢复"),wr.P(n["student_resume"]),
        wr.H2("算法定标细节",collapsed_blocks=[wr.MarkdownBlock(n["algorithm_calibration"])]),
        wr.H2("指标与评测口径",collapsed_blocks=[wr.P(n["definitions"]),wr.P("数学/代码：DN-MOPD 8K、关闭思考；其余五项：IM-MOPD avg@3、开启思考。IFBench 为 prompt-loose，IFEval 为 prompt-strict。"),wr.MarkdownBlock("\n".join(refs_table)),wr.P(n["reference_note"])]),
        wr.MarkdownBlock("[完整训练指标：按领域与 DN 分组查看](https://wandb.ai/chenxheee-sjtu-hpc-center/bridge_mopd?nw=xmjip17v9ke) · 展示分区参考 [MiMo RL](https://mimo.xiaomi.com/rl/)。")
    ]
    # Use the normal reading width rather than stretching tables and text across the screen.
    report=wr.Report(entity=ENTITY,project=PROJECT,title=TITLE,description="总览 · 训练指标 · 正式评测 · 学生设置与算法",width="readable",blocks=blocks)
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
    assert len(loaded.sections)==5
    loaded_report=wr.Report.from_url(report.url)
    assert loaded_report.title==TITLE
    print("VERIFIED NATIVE W&B OVERVIEW",flush=True)


if __name__=="__main__":
    main()
