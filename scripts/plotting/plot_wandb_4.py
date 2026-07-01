import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COMM_KEY = "total_comm_count"
REWARD_KEYS = {
    "MPE Simple Spread": "episode_reward_dyna",
    "MPE Simple FACMAC": "epi_reward_agent-0"
}

FILES = {
    "MPE Simple Spread": "mpe_simple_spread.csv",
    "MPE Simple FACMAC": "mpe_simple_tag.csv"
}

def interpolate_run(df, common_comm, reward_key):

    x = df[COMM_KEY].to_numpy()

    y = df[reward_key].to_numpy()

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_interp = np.interp(common_comm, x, y, left=y[0], right=y[-1])
    return y_interp

def compress_comm(df, comm_key, reward_key):
    # 按 step 排序
    df = df.sort_values("_step")

    # 对每个 comm，只保留最后一个（即最大 step）
    df = df.groupby(comm_key, as_index=False).first()

    return df

def aggregate(df, run_ids, common_comm, reward_key):

    curves = []

    for rid in run_ids:
        run_df = df[df["run_id"] == rid]
        run_df = compress_comm(run_df, COMM_KEY, reward_key)
        curves.append(interpolate_run(run_df, common_comm, reward_key))

    curves = np.stack(curves)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    return mean, std


fig, axes = plt.subplots(1,2, figsize=(10,4))

for i,(env, path) in enumerate(FILES.items()):

    reward_key = REWARD_KEYS[env]

    df = pd.read_csv(path)

    runs = df["run_id"].unique()

    # 前半 AORPO
    aorpo_runs = runs[:len(runs)//2]
    # aorpo_runs = runs[:2]

    # 后半 ours
    ours_runs = runs[len(runs)//2:]
    # ours_runs = runs[-3:]

    max_comm = df[COMM_KEY].max()

    COMMON_COMM = np.linspace(0, max_comm, 40)

    mean_aorpo, std_aorpo = aggregate(df, aorpo_runs, COMMON_COMM, reward_key)
    mean_ours, std_ours = aggregate(df, ours_runs, COMMON_COMM, reward_key)

    ax = axes[i]

    ax.plot(COMMON_COMM / max_comm, mean_ours, linewidth=2.5, label="Uncertainty-Aware")
    ax.fill_between(COMMON_COMM / max_comm, mean_ours-std_ours, mean_ours+std_ours, alpha=0.2)

    ax.plot(COMMON_COMM / max_comm, mean_aorpo, linewidth=2.5, label="AORPO")
    ax.fill_between(COMMON_COMM / max_comm, mean_aorpo-std_aorpo, mean_aorpo+std_aorpo, alpha=0.2)

    ax.set_title(env)
    ax.set_xlabel("Communication Ratio")

    if i == 0:
        ax.set_ylabel("Episode Reward")

    ax.grid(alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False
)

plt.tight_layout(rect=(0,0.08,1,1))

plt.savefig("reward_vs_comm.pdf", dpi=300)
plt.show()
