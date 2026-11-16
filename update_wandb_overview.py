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
                                      "summary:evaluation/checkpoint_step", "summary:evaluation/macro6_pct"],
                     lock_columns=True)


def grid(records, title, panels):
    if len(panels) == 1:
        panels[0].layout.w = 24
    return wr.PanelGrid(runsets=[runset(records, title)], panels=panels)


def refresh_aliases(api, records):
    for record in records:
        path = f"{ENTITY}/{PROJECT}/{run_id(record)}"
        public = api.run(path)
        public.name = record.get("display_name", record["name"])
        public.config.update(record.get("overview", {}))
        public.config["original_run_label"] = record["name"]
        public.config["primary_training_metric"] = "training/math_accuracy_pct"
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
            for axis, target, prefixes in [("plot/step", previous, ("training/", "evaluation/", "diagnostics/", "schedule/", "performance/")),
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
            for prefix in ["training/*", "evaluation/*", "diagnostics/*", "schedule/*", "performance/*"]:
                run.define_metric(prefix, step_metric="plot/step", step_sync=False)
            for step, values in sorted(raw_changes.items()):
                run.log({"train/step": step, **values})
            for step, values in sorted(changes.items()):
                run.log({"plot/step": step, **values})
            run.summary["train/step"] = int(run.summary.get("raw_import_last_source_step", record["lastStep"]))
            run.finish()
        print("OVERVIEW METRICS", record.get("display_name"), "changed steps", len(changes), flush=True)


def construct(index, refs=None):
    records = [r for r in index["runs"] if r.get("overview", {}).get("scale") == "1.7B"
               and r["overview"].get("method") in {"plain", "progressive"}]
    if not records:
        raise ValueError("Missing prepared overview metadata; run export_wandb_curves.py before publishing")
    current = [r for r in records if r["overview"]["normalized"]]
    established = [r for r in records if not r["overview"]["normalized"] and "注入" not in r["name"]]
    injected = [r for r in records if "注入" in r["name"]]
    primary = established + current
    performance = [panel("六领域评测均分 · 看整体效果", "evaluation/macro6_pct", percent=True)]
    performance[0].layout.w = 24
    for item in performance:
        item.layout.y += 5
    breakdown = [panel(title, key, percent=True, position=i) for i, (title, key) in enumerate([
        ("数学评测 · AIME25/26 平均", "evaluation/math_pct"),
        ("代码评测 · LCB v5/v6 平均", "evaluation/code_pct"),
        ("医疗评测 · MedQA", "evaluation/medqa_pct"),
        ("法律评测 · CaseHOLD", "evaluation/casehold_pct"),
        ("金融评测 · FinQA", "evaluation/finqa_pct"),
        ("指令遵循评测 · IFBench/IFEval 平均", "evaluation/instruction_following_pct"),
    ])]
    domain_training = [panel(f"{title}训练 · 学生与当前教师的差异", f"bridge/abs_k1/{key}",
                             x="train/step", smooth=True, position=i)
                       for i,(title,key) in enumerate([("数学","math"),("代码","code"),("医学","med"),
                                                       ("法律","law"),("金融","fin"),("指令遵循","if")])]
    diagnostics = [panel(title, key, position=i) for i, (title, key) in enumerate([
        ("数学训练题正确率 · 只代表训练数学表现", "training/math_accuracy_pct"),
        ("指令遵循细项 · IFBench", "evaluation/ifbench_pct"),
        ("指令遵循细项 · IFEval", "evaluation/ifeval_pct"),
        ("学生训练所用医疗教师强度 λ", "schedule/med_lambda"),
        ("学生训练所用法律教师强度 λ", "schedule/law_lambda"),
        ("梯度范数 · 裁剪前", "diagnostics/gradient_norm"),
        ("蒸馏绝对 loss · 越低不等于能力越强", "diagnostics/distillation_abs_loss"),
        ("平均回答长度 · token", "diagnostics/response_tokens"),
        ("回答写满上限比例 · %", "diagnostics/truncated_pct"),
        ("数学题占当批题目的比例 · %", "training/math_prompt_share_pct"),
        ("每步耗时 · 秒", "performance/step_seconds"),
    ])]
    note = ("### 先看六领域正式评测，再看整体均分\n\n"
            "**数学、代码、医学、法律、金融、指令遵循**分别展示，不能用数学训练分数代表所有领域。六领域均分用于看整体提升，分项用于看收益和退化发生在哪里。\n\n"
            "指令遵循分数是 IFBench 与 IFEval 的平均；两项原始分数保留在诊断区。比较同一起点、同一配方、同一检查点的实验。\n\n"
            "评测只显示实际完成的检查点，不平滑、不补中间分数。空白表示尚无已完成评测，不是零分。\n\n"
            "训练区的数学正确率只反映数学训练题；其他领域没有同口径的在线正确率，不能从 critic 推算。不同教师目标下的 loss 也不能直接当能力分数。\n\n"
            "W&B 的 Finished 表示补传结束，实际训练进度看表中的已记录步数。")
    workspace = ws.Workspace(entity=ENTITY, project=PROJECT, name="1.7B 渐进式教师主对照",
        auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="plot/step", smoothing_type="none", smoothing_weight=0,
                                      max_runs=8, ignore_outliers=False),
        runset_settings=ws.RunsetSettings(filters=[Metric("ID").isin([run_id(r) for r in primary])],
            run_settings=settings(primary), pinned_columns=["run:displayName", "config:recorded_step.value",
                                                            "summary:evaluation/checkpoint_step", "summary:evaluation/macro6_pct"]),
        sections=[
            ws.Section(name="先看这里：六领域整体评测", is_open=True,
                       panels=[wr.MarkdownPanel(markdown=note, layout=wr.Layout(x=0,y=0,w=24,h=5)), *performance]),
            ws.Section(name="六领域正式评测 · 单位 %", is_open=True, panels=breakdown),
            ws.Section(name="六领域训练 · 学生与当前教师差异（不是正确率）", is_open=True, panels=domain_training),
            ws.Section(name="教师推进与训练诊断", is_open=False, panels=diagnostics),
        ])
    table = ["| 实验 | 已记录训练进度 | 最新完整评测检查点 | 六领域评测均分 |",
             "|---|---:|---:|---:|"]
    for record in records:
        content=json.loads(Path("data", record["id"]+".json").read_text())
        points=content.get("derived",{}).get("evaluation/macro6_pct",[])
        checkpoint,score=(str(points[-1][0]),f"{points[-1][1]:.2f}") if points else ("未齐","—")
        table.append(f"| {record.get('display_name')} | {record['lastStep']}/{record['overview']['budget_steps']} | {checkpoint} | {score} |")
    domain_columns = [("数学", "math"), ("代码", "code"), ("医学", "medqa"),
                      ("法律", "casehold"), ("金融", "finqa"), ("指令遵循", "instruction_following")]
    domain_table = ["| 实验 | 检查点 | " + " | ".join(t for t,k in domain_columns) + " | 六域均分 |",
                    "|---|---:|" + "---:|" * 7]
    for record in records:
        d = json.loads(Path("data", record["id"]+".json").read_text()).get("derived", {})
        checkpoints = sorted({int(step) for key,points in d.items() if key.startswith("evaluation/")
                              and key.endswith("_pct") for step,value in points})
        checkpoint = checkpoints[-1] if checkpoints else None
        def score(key):
            values = dict(d.get(key, []))
            return f"{values[checkpoint]:.2f}" if checkpoint in values else "—"
        values = [score(f"evaluation/{key}_pct") for title,key in domain_columns]
        domain_table.append("| " + " | ".join([record["display_name"], str(checkpoint) if checkpoint else "未评测",
                                               *values, score("evaluation/macro6_pct")]) + " |")
    blocks = [wr.P("更新于 "+index["updated"]+"。仅展示学生 MOPD 实验及其对照。"),
              wr.MarkdownBlock(note), wr.H1("六领域正式评测结果"), wr.MarkdownBlock("\n".join(domain_table)),
              wr.P("表中每行只取同一个检查点；— 表示该项未完成。训练进度与评测进度分开记录。"),
              wr.MarkdownBlock("\n".join(table)),
              wr.P("六领域均分 = mean(数学、代码、医疗、法律、金融、(IFBench + IFEval)/2)。不同检查点和不同配方分开比较。"),
              wr.P("数学与代码使用 DN-MOPD 8K、不思考的评测；其余使用 IM-MOPD avg@3、开启思考。IFBench 是宽松 prompt 级准确率，IFEval 是严格 prompt 级准确率。"),
              grid(established,"不归一化的已有对照",[panel("六领域评测均分 · %","evaluation/macro6_pct",percent=True)]),
              wr.H1("最新归一化对照"),
              wr.P("这里只比较普通 MOPD＋归一化与渐进式教师＋归一化。训练步数按最近同步的日志显示，最终评测未齐时不提前判输赢。"),
              grid(current,"同归一化配方的主对照",[panel("六领域评测均分 · 等待完整评测","evaluation/macro6_pct",percent=True)]),
              wr.H2("正式评测分项"),grid(primary,"1.7B 学生实验",breakdown),
              wr.H2("六领域训练过程"),
              wr.P("逐领域显示 bridge/abs_k1：学生与当前教师在生成 token 上的 log 概率差绝对值的均值。它衡量对当前教师的贴近程度，不是正确率，也不能单凭它判断能力。教师切换或推进时目标会变，因此需结合教师强度与正式评测看。显示原始值及 EMA 0.6。"),
              grid(primary,"六领域学生训练诊断",domain_training),
              wr.H2("教师推进和训练稳定性",collapsed_blocks=[grid(primary,"1.7B 学生实验",diagnostics)]),
    ]
    if injected:
        blocks.append(wr.H2("医疗法律注入起点 · 单独查看",collapsed_blocks=[
            wr.P("这一组改变了学生起点，不与原始学生起点的组混作同一条件。"),
            grid(injected,"注入起点",[panel("数学训练题正确率 · %","training/math_accuracy_pct",percent=True),
                 panel("六领域评测均分 · %","evaluation/macro6_pct",percent=True,position=1)])]))
    archives=[r for r in index["runs"] if r.get("overview",{}).get("scale")=="2B"]
    blocks.append(wr.H2("2B 历史实验 · 不与 1.7B 混比",collapsed_blocks=[
        wr.P("历史组包含不同起点和不同机器。三领域均分与六领域均分不是同一个指标。"),
        grid(archives,"2B 历史对照",[panel("三领域评测均分 · %","evaluation/macro3_pct",percent=True),
            panel("数学训练题正确率 · %","training/math_accuracy_pct",percent=True,smooth=True,position=1)])]))
    blocks.append(wr.P("展示组织参考 MiMo RL 页：训练表现与 benchmark 分区、定义清楚、同实验固定颜色。指标计算仍使用本项目真实口径。"))
    report = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE,
                      description="六领域正式评测与整体均分；训练表现作为诊断", width="fluid", blocks=blocks)
    return workspace, report


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
    refs["workspace"]=workspace.url
    file.write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    print("DASHBOARD WORKSPACE",workspace.url,flush=True)
    if refs.get("report"):
        old=wr.Report.from_url(refs["report"]);old.title,old.description,old.blocks,old.width=report.title,report.description,report.blocks,report.width
        report=old
    report.save()
    refs["report"]=report.url
    file.write_text(json.dumps(refs,ensure_ascii=False,indent=2))
    print("DASHBOARD REPORT",report.url,flush=True)
    # Reload native saved objects to verify that filters and chart definitions survived saving.
    loaded=ws.Workspace.from_url(workspace.url)
    assert len(loaded.sections)==4
    loaded_report=wr.Report.from_url(report.url)
    assert loaded_report.title==TITLE
    print("VERIFIED NATIVE W&B OVERVIEW",flush=True)


if __name__=="__main__":
    main()
