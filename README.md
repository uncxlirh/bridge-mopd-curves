# Bridge-MOPD 官方 W&B 补传

[打开六领域实验总览](https://forge.coreweave.com/chenxheee-sjtu-hpc-center/bridge_mopd/reports/渐进式教师实验总览--VmlldzoxODMxNzAwNA==)

先看数学、代码、医学、法律、金融、指令遵循的正式评测，再看六领域均分。指令遵循取 IFBench 和 IFEval 平均，两个细项另行保留。表格按同一检查点列分数；缺失项显示 —，不按零分处理。曲线横轴是原始训练步数，评测不平滑、不补点。

`critic/score/mean` 不是六领域准确率。这套奖励记录中其他训练领域是零分占位，因此它只支持重建数学训练题正确率。该曲线放在训练诊断区，不能用它判断其他领域或最终评测输赢。不同教师目标下的 loss 也只用于诊断。

只上传正式学生实验的训练曲线及已完成检查点评测。教师训练、冒烟测试、工程测试、校准、旧 bug 实验及独立静态测试不上传；正式学生实验只有一个检查点的评测仍保留。目前 15 条学生实验记录。W&B Finished 表示本次补传结束，实际进度看总览表。

使用官方 W&B Reports / Workspaces，没有自制展示界面。GitHub Actions 使用仓库加密 Secret `WANDB_BACKFILL_API_KEY`；密钥不写进代码和日志。更新 data 后自动补传，也可手动运行 `upload-wandb.yml`；仅调整视图可运行 `publish-wandb-views.yml`。`verify-wandb-overview.yml` 不带密钥，核对匿名访问、六领域图和实际分数。
