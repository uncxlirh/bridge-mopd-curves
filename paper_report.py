"""New W&B report (2026-11-23): what we did, why, and the two positive results (steps 40 / 100).
Numbers: bridge-mopd reports/q17_six_20261118/PAIRED_CI.md (3 seeds pooled, paired 95% intervals) and SIGNAL.md.
Creates four small summary runs (one per method) so the report can show native W&B bar charts."""
import hashlib, json
from pathlib import Path
import wandb
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces.expr import Metric

ENTITY, PROJECT = "chenxheee-sjtu-hpc-center", "bridge_mopd"
TITLE = "渐进式教师：小模型学得更多、更均衡（Qwen3-1.7B，六个领域）"
METHODS = [  # name, color, seeds, numbers
    ("普通 MOPD", "#9e9e9e", 3, {"s40_mean": 31.60, "s40_worst": -23.18, "s40_avg6": 44.09,
                                "s100_mean": 54.54, "s100_worst": 6.72, "s100_avg6": 46.45, "share_medlaw": 76.1}),
    ("只调小信号（对照）", "#f2a65a", 3, {"s40_mean": 47.11, "s40_worst": -15.41, "s40_avg6": 45.58,
                                     "s100_mean": 56.77, "s100_worst": -11.39, "s100_avg6": 46.59, "share_medlaw": 26.6}),
    ("只换近老师（对照）", "#7fb3d5", 1, {"s40_mean": 29.94, "s40_worst": 15.54, "s40_avg6": 43.62,
                                     "s100_mean": 52.82, "s100_worst": 11.56, "s100_avg6": 46.47, "share_medlaw": 77.8}),
    ("渐进式教师（我们）", "#c0392b", 3, {"s40_mean": 50.59, "s40_worst": 9.08, "s40_avg6": 45.93,
                                     "s100_mean": 63.45, "s100_worst": 15.75, "s100_avg6": 47.21, "share_medlaw": 25.4}),
]
rid = lambda n: "sum" + hashlib.sha256(("paper-2026-11-23-v2/" + n).encode()).hexdigest()[:12]  # noqa: E731


def public_url(url):
    from urllib.parse import urlsplit, urlunsplit
    p = urlsplit(url)
    return urlunsplit(("https", "wandb.ai", p.path.removeprefix("/wandb"), p.query, p.fragment)) if p.netloc == "forge.coreweave.com" else url


def upload():
    for name, _, seeds, v in METHODS:
        run = wandb.init(entity=ENTITY, project=PROJECT, id=rid(name), name=name, resume="allow", mode="online",
                         group="汇报 2026-11-23", job_type="report-summary", tags=["report-summary"],
                         config={"seeds": seeds, "source": "PAIRED_CI.md / SIGNAL.md (bridge-mopd main)"},
                         settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True, console="off"))
        if not run.summary.get("history_done"):  # learning curve: step 0 (the model itself), 40, 100
            run.define_metric("plot/step"); run.define_metric("*", step_metric="plot/step")
            for st, row in [(0, SUMMARY_STEPS[0]), (40, {"mean": v["s40_mean"], "worst": v["s40_worst"], "avg6": v["s40_avg6"]}),
                            (100, {"mean": v["s100_mean"], "worst": v["s100_worst"], "avg6": v["s100_avg6"]})]:
                run.log({"plot/step": st, "recovery/mean_pct": row["mean"], "recovery/worst_pct": row["worst"], "score/avg6": row["avg6"]})
        for k, x in v.items():
            run.summary[k] = x
        run.summary["history_done"] = True
        run.finish()


D6 = ["math", "code", "med", "law", "fin", "if"]
CURVES = [  # name, color, source: W&B snapshot (data/<id>.json) or B200 export (paper/*.json)
    ("每步意见 · 普通 MOPD", "#9e9e9e", "data/73d3a9cbb33325f3.json"),
    ("每步意见 · 只调小信号", "#f2a65a", "paper/sigmatch_s43.json"),
    ("每步意见 · 渐进式教师", "#c0392b", "data/838ab96e2e8f1917.json"),
]
SUMMARY_STEPS = {0: {"mean": 0.0, "worst": 0.0, "avg6": 40.98}}  # step 0 = Qwen3-1.7B itself


def steps_of(src):
    j = json.loads(Path(src).read_text())
    if "steps" in j:  # B200 export: per step, per domain abs_k1 / resp_len / signal weight
        return {int(s): {d: v[d]["abs_k1"] * v[d]["resp_len"] * v[d]["w"] for d in D6} for s, v in j["steps"].items()}, {}
    m = j["metrics"]
    k = {d: dict(map(tuple, m["bridge/abs_k1/" + d])) for d in D6}
    L = {d: dict(map(tuple, m["bridge/resp_len/" + d])) for d in D6}
    lam = {d: dict(map(tuple, m.get("bridge/lam/" + d, []))) for d in ("med", "law")}
    common = set.intersection(*[set(k[d]) & set(L[d]) for d in D6])
    return {int(s): {d: k[d][s] * L[d][s] for d in D6} for s in common}, lam


def start(name, rid_, cfg):
    return wandb.init(entity=ENTITY, project=PROJECT, id=rid_, name=name, resume="allow", mode="online",
                      group="汇报 2026-11-23", job_type="report-summary", tags=["report-summary"], config=cfg,
                      settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True, console="off"))


def upload_curves():
    for name, _, src in CURVES:
        run = start(name, rid(name), {"source": src})
        if run.summary.get("history_done"):
            run.finish(); continue
        run.define_metric("plot/step"); run.define_metric("*", step_metric="plot/step")
        sig, lam = steps_of(src)
        for s in sorted(sig):
            w = sig[s]; tot = sum(w.values())
            row = {"plot/step": s, "share/medlaw_pct": 100 * (w["med"] + w["law"]) / tot}
            row.update({f"share/{d}_pct": 100 * w[d] / tot for d in D6})
            for d in ("med", "law"):
                if lam.get(d) and s in lam[d]:
                    row[f"teacher/{d}_lambda"] = lam[d][s]
            run.log(row)
        run.summary["history_done"] = True
        run.finish()


def bar(title, metric, rng, pos, xlabel="%"):
    return wr.BarPlot(title=title, metrics=[metric], orientation="h", range_x=rng, title_x=xlabel, aggregate=False,
                      max_runs_to_show=4, max_bars_to_show=4, font_size="medium",
                      layout=wr.Layout(x=(pos % 2) * 12, y=(pos // 2) * 7, w=12, h=7))


def line(title, keys, pos, rng=(None, None), ylabel="%", smooth=False, w=12):
    return wr.LinePlot(title=title, x="plot/step", y=keys, title_x="训练步数", title_y=ylabel, range_y=rng,
                       smoothing_type="exponential" if smooth else "none", smoothing_factor=.6 if smooth else 0,
                       smoothing_show_original=smooth, aggregate=False, max_runs_to_show=4, legend_position="south",
                       font_size="medium", layout=wr.Layout(x=(pos % 2) * 12, y=(pos // 2) * 8, w=w, h=8))


def grid(names, panels):
    rs = wr.Runset(entity=ENTITY, project=PROJECT, name="汇报",
                   filters=[Metric("ID").isin([rid(n) for n in names])],
                   run_settings={rid(n): wr.RunSettings(color=c, disabled=False)
                                 for n, c in [(m[0], m[1]) for m in METHODS] + [(c[0], c[1]) for c in CURVES] if n in names})
    return wr.PanelGrid(runsets=[rs], panels=panels, hide_run_sets=True)


def build():
    allm = [m[0] for m in METHODS]
    main2 = ["普通 MOPD", "渐进式教师（我们）"]
    curv = [c[0] for c in CURVES]
    blocks = [
        wr.P("Bridge-MOPD · Qwen3-1.7B · 六个领域 · 训练 100 步 · 2026 年 11 月 23 日"),
        wr.H1("一分钟看懂"),
        wr.MarkdownBlock(
            "**要做的事**：把六位专科老师（数学、代码、医疗、法律、金融、听指令）的本事，教给同一个小模型。"
            "普通 MOPD 的做法是：小模型自己做题，每道题由对应科目的老师批改，小模型照着批改去改。\n\n"
            "**遇到的问题**：医疗、法律两位老师和小模型差得太远，批改意见又多又重，占了全部意见的四分之三。"
            "小模型被这两科拉着走，其他科目顾不上，法律自己反而越学越差。\n\n"
            "**我们的办法**：医疗、法律先换成一位“中间老师”：水平介于真老师和小模型之间，由两者按比例混合得到。"
            "小模型跟上了，再一步步换成更接近真老师的版本。其他四科照常用真老师。\n\n"
            "**结果**：小模型学到的老师本事，40 步时从 32% 提高到 51%，100 步时从 55% 提高到 64%；法律不再变差。"),
        wr.MarkdownBlock("> **学到多少** = 小模型比开始时多涨的分 ÷ 老师比小模型开始时高出的分。100% 就是学到了老师的水平，"
                         "负数表示比开始时还差。六科各算一个，再看平均和最差的那一科。"),
        wr.H1("结果一：学得更多"),
        wr.P("六科平均学到多少（三个随机种子的平均）。"),
        grid(main2, [line("随训练步数：平均学到多少", ["recovery/mean_pct"], 0, (0, 70)),
                     line("随训练步数：最差一科", ["recovery/worst_pct"], 1, (-30, 30))]),
        grid(main2, [bar("训练 40 步", "s40_mean", (0, 70), 0), bar("训练 100 步", "s100_mean", (0, 70), 1)]),
        wr.MarkdownBlock(
            "| | 40 步 | 100 步 |\n|---|---:|---:|\n"
            "| 普通 MOPD | 31.6% | 54.5% |\n| **渐进式教师** | **50.6%** | **63.5%** |\n"
            "| 多学到 | **+19.0**（95% 区间 +14.4 ～ +23.7） | **+8.9**（+4.7 ～ +13.1） |\n\n"
            "六科平均分也更高：40 步 45.93 对 44.09，100 步 47.21 对 46.45。"),
        wr.H1("结果二：最差的那一科也保住了"),
        wr.P("普通 MOPD 里最差的是法律：训练 40 步后比开始时还差。渐进式教师下它也在进步。"),
        grid(main2, [bar("训练 40 步 · 最差一科", "s40_worst", (-30, 30), 0), bar("训练 100 步 · 最差一科", "s100_worst", (-30, 30), 1)]),
        wr.MarkdownBlock(
            "| | 40 步 | 100 步 |\n|---|---:|---:|\n"
            "| 普通 MOPD | −23.2% | 6.7% |\n| **渐进式教师** | **9.1%** | **15.8%** |\n"
            "| 差 | **+32.3**（+9.9 ～ +37.1） | **+9.0**（+3.4 ～ +29.8） |"),
        wr.H1("为什么有效：意见分得更均匀"),
        wr.P("前 40 步里，医疗、法律的批改意见占全部意见的比例："),
        grid(curv, [line("每一步：医疗 + 法律占的意见", ["share/medlaw_pct"], 0, (0, 100), smooth=True, w=24)]),
        wr.P("普通 MOPD 一直在 75% 上下；渐进式教师从第 2 步起降到约 25%，之后随着医疗、法律的老师升档慢慢回升。"
             "只调小信号的对照被强行压到同样的比例。"),
        grid([CURVES[2][0]], [line("渐进式教师：医疗、法律的老师档位（1 = 真老师）", ["teacher/med_lambda", "teacher/law_lambda"], 0, (0, 1), ylabel="档位", w=24)]),
        grid(allm, [bar("前 40 步平均：医疗 + 法律占的意见", "share_medlaw", (0, 100), 0)]),
        wr.MarkdownBlock(
            "中间老师同时做了两件事。我们把它们拆开各做一遍：\n\n"
            "| 做法 | 医疗+法律的意见 | 40 步平均学到 | 40 步最差一科 |\n|---|---:|---:|---:|\n"
            "| 普通 MOPD | 76% | 31.6% | −23.2% |\n"
            "| 只把医疗、法律的意见调小 | 27% | 47.1% | −15.4% |\n"
            "| 只换成近一些的老师，意见比例不变 | 78% | 29.9% | 15.5% |\n"
            "| **渐进式教师：两件都做** | **25%** | **50.6%** | **9.1%** |\n\n"
            "- **意见调小**：其他科目分到了注意力，平均学得更多。\n"
            "- **老师更近**：小模型学得动，法律保住了。\n"
            "- 只做一件都不够，渐进式教师两件都做到。"),
        grid(allm, [bar("40 步 · 平均学到", "s40_mean", (0, 70), 0), bar("40 步 · 最差一科", "s40_worst", (-30, 30), 1)]),
        wr.H1("每一科"),
        wr.MarkdownBlock(
            "渐进式教师减普通 MOPD（分）：\n\n"
            "| | 数学 | 代码 | 医疗 | 法律 | 金融 | 听指令 |\n|---|---:|---:|---:|---:|---:|---:|\n"
            "| 40 步 | **+2.66** | **+2.21** | +0.58 | **+3.32** | **+0.83** | **+1.44** |\n"
            "| 100 步 | **+0.73** | +0.23 | **+1.40** | **+0.93** | +0.48 | +0.80 |\n\n粗体：95% 区间在 0 以上。"),
        wr.H1("实验设置"),
        wr.MarkdownBlock(
            "- 小模型：Qwen3-1.7B；六位老师：数学、代码、医疗、法律、金融、听指令。\n"
            "- 每步 96 道题，每题 4 个回答，训练 100 步；三个随机种子。\n"
            "- 测试：数学 AIME，代码 LiveCodeBench，医疗 MedQA，法律 CaseHOLD，金融 FinQA，听指令 IFEval + IFBench。\n"
            "- 95% 区间：三个种子合并后按题目重新抽样 1 万次。\n\n"
            "[训练曲线与完整分项](https://wandb.ai/chenxheee-sjtu-hpc-center/bridge_mopd/reports/渐进式教师实验总览--VmlldzoxODMxNzAwNA==)"),
    ]
    return wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, description="做了什么 · 为什么 · 结果", width="readable", blocks=blocks)


def forge_url(url):  # the form wr.Report.from_url parses: Forge host, no /wandb prefix, plain slug before the id
    rid_ = url.rsplit("--", 1)[-1]
    return f"https://forge.coreweave.com/{ENTITY}/{PROJECT}/reports/r--{rid_}"


def main():
    upload()
    upload_curves()
    report = build()
    found = []
    try:
        found = [r.url for r in wandb.Api().reports(path=f"{ENTITY}/{PROJECT}")
                 if TITLE in (getattr(r, "display_name", None), getattr(r, "name", None), getattr(r, "title", None))]
    except Exception as e:
        print("::notice title=report lookup failed::" + repr(e)[:300], flush=True)
    loaded = []
    for u in found:
        try:
            loaded.append(wr.Report.from_url(forge_url(u)))
        except Exception as e:
            print("::notice title=report load failed::" + u + " " + repr(e)[:200], flush=True)
    if found and not loaded:  # never create another copy when the existing ones cannot be opened
        raise SystemExit("existing reports found but none could be loaded")
    if loaded:
        keep, extra = loaded[0], loaded[1:]
        keep.title, keep.description, keep.blocks, keep.width = report.title, report.description, report.blocks, report.width
        report = keep
        for old in extra:  # duplicates from earlier runs: mark, the user can delete them in the UI
            old.title = "（重复，可删除）" + TITLE
            old.save()
            print("::notice title=duplicate marked::" + public_url(old.url), flush=True)
    report.save()
    url = public_url(report.url)
    Path("paper_report_ref.json").write_text(json.dumps({"report": url}, ensure_ascii=False, indent=2))
    print("::notice title=paper report::" + url, flush=True)
    assert wr.Report.from_url(report.url).title == TITLE
    print("VERIFIED", url, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("::error title=paper report failed::" + traceback.format_exc()[-900:].replace("\n", " | "), flush=True)
        raise
