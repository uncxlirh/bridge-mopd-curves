# Bridge-MOPD 官方 W&B 补传

这里只保留正式实验的训练曲线与阶段评测曲线，使用官方 W&B 客户端导入。

不包含自制展示界面。冒烟测试、工程测试、校准记录、旧版 bug 实验和单点静态结果不补传。

GitHub Actions 使用仓库加密 Secret `WANDB_BACKFILL_API_KEY`；密钥不写入代码。更新 data 后自动补传，也可以手动运行工作流。
