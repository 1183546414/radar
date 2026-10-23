"""雷达抗干扰离线学习 — 主流程入口。

在项目根目录执行:
    python main.py

运行日志（每次执行前清空）: outputs/logs/main.log
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from radar.data.build_dataset import dataset_main
    from radar.train.state_vae_contrastive import train_vae_main
    from radar.train.train_q_latent import train_q_latent_main
    from radar.train.train_q_retrieval import train_q_retrieval_main
    from radar.analyze.clustering import clustering_main
    from radar.infer.holdout_eval import holdout_eval_main

    print("=== 1/6 数据处理 ===")
    dataset_main()

    print("=== 2/6 训练状态 VAE ===")
    train_vae_main()

    print("=== 2b/6 生成 VAE 重建可视化 (5×4 对比 + 绝对误差) ===")
    from radar.viz.reconstructed_samples import plot_reconstructed_samples_main

    plot_reconstructed_samples_main()

    print("=== 2c/6 潜向量 t-SNE 可视化 (干扰类型) ===")
    from radar.viz.latent import plot_latent_viz_main

    plot_latent_viz_main()

    print("=== 2d/6 潜空间无监督聚类 (KMeans) ===")
    clustering_main()

    print("=== 3/6 训练 Q 网络 Q(z,A)->R ===")
    train_q_latent_main()

    print("=== 4/6 训练检索条件 Q Q(z,z_ctx,A)->R ===")
    train_q_retrieval_main()

    print("=== 5/6 留出场景推理评估 ===")
    holdout_eval_main(regenerate_holdout=True)

    print("=== 5b/6 生成训练效果分析图 ===")
    from radar.viz.training_summary import plot_training_summary_main

    plot_training_summary_main()

    print("=== 5c/6 导出留出集对比表 ===")
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_baseline_table.py")],
        check=True,
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )

    print("=== 5d/6 生成留出集主报图 (ΔSINR / 胜率 / regret 分位数) ===")
    from radar.viz.holdout_presentation import plot_holdout_presentation_main

    plot_holdout_presentation_main(refresh=False)


if __name__ == "__main__":
    from radar.paths import LOG_MAIN
    from radar.run_log import run_with_log

    with run_with_log(LOG_MAIN, script_name="main.py"):
        print(f"日志文件: {LOG_MAIN.resolve()}\n")
        main()
