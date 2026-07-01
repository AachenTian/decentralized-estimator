import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. 配置信息
# =========================
FILE_NAME = "multi_run_metrics_final.csv"
X_METRIC = "total_comm_count"
Y_METRIC = "episode_reward_dyna"

# 插值范围 (0 到 1.2M)
COMMON_X = np.linspace(0, 350000, 30)

# 设置纵轴范围 (按你给出的图，-50 到 -15 比较合适)
Y_LIMIT = (-45, -15)

# 平滑窗口大小 (数值越大，曲线越平滑，阴影越窄)
SMOOTH_WINDOW = 1


def main():
    if not os.path.exists(FILE_NAME):
        print(f"错误：找不到文件 {FILE_NAME}")
        return

    # 读取数据
    df = pd.read_csv(FILE_NAME)

    # 获取所有的 run_id
    unique_run_ids = df['run_id'].unique()

    # 分组：前6个是 AORPO，后6个是 Ours
    groups = {
        "AORPO": unique_run_ids[:6],
        "Ours": unique_run_ids[6:12]
    }

    # 横向压缩图片
    plt.figure(figsize=(5, 6))

    for group_name, run_ids in groups.items():
        y_list = []

        for rid in run_ids:
            # 提取单个 Run 的数据并排序
            run_data = df[df['run_id'] == rid].sort_values(X_METRIC)
            x_raw = run_data[X_METRIC].values
            y_raw = run_data[Y_METRIC].values

            if len(x_raw) < 2:
                continue

            # 插值，left=y_raw[0] 消除左侧缝隙
            y_interp = np.interp(COMMON_X, x_raw, y_raw, left=y_raw[0], right=y_raw[-1])

            # --- 平滑处理：在聚合前对每条原始线进行滑动平均 ---
            y_smoothed = pd.Series(y_interp).rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean().to_numpy()
            y_list.append(y_smoothed)

        if not y_list:
            continue

        # 聚合计算
        ys = np.stack(y_list, axis=0)
        mean = np.nanmean(ys, axis=0)
        std = np.nanstd(ys, axis=0)

        # --- 核心修改：使用 SEM (标准误差) 代替 STD ---
        # SEM = STD / sqrt(n)
        sem = std / np.sqrt(len(y_list))

        # 使用 SEM 绘制阴影，会让图片看起来更“聚拢”
        shade = sem

        # 绘制主线
        line, = plt.plot(COMMON_X, mean, linewidth=2.5, label=group_name)

        # 绘制阴影 (设置更低的 alpha 值增加透明度)
        plt.fill_between(COMMON_X, mean - shade, mean + shade, alpha=0.15)

        # 绘制最后 5 个点的平均虚线
        final_avg = np.nanmean(mean[-15:])
        plt.axhline(y=final_avg, color=line.get_color(), linestyle='--', linewidth=1.2, alpha=0.6)

    # --- 图表美化 ---
    plt.ylim(Y_LIMIT)
    plt.xlim(0, COMMON_X.max())
    plt.gca().margins(x=0)

    # X轴格式化
    from matplotlib.ticker import FuncFormatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}M' if x != 0 else '0'))

    plt.title("Cooperative communication", fontsize=16)
    plt.xlabel("Total Communication Count", fontsize=14)
    plt.ylabel("Episode Rewards", fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.2, linestyle=':')
    plt.tight_layout()

    # 保存
    save_name = "smoothed_sem_plot.png"
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    print(f"✅ 绘图完成！更窄平滑版本的图片已保存为 {save_name}")
    plt.show()


if __name__ == "__main__":
    main()