import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# =========================
# 基本配置
# =========================

ENTITY = "yachen-tian-rwth-aachen-university"

PROJECTS = {
    "MPE Simple Spread": "AORPO-dynamics model",
    "MPE Simple Facmac": "AORPO-simple_tag"
}

METRICS = {
    "MPE Simple Spread": "episode_reward_env",
    "MPE Simple Facmac": "epi_reward_agent-0"
}
STEP_KEY = "_step"

OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

COMMON_STEPS = np.arange(0, 34000, 1400)

RUN_GROUPS = {
    "MPE Simple Spread":{
        "Uncertainty-Aware": [
            "routing reset_seed = 30 k = 6 policy lr = 0.01",
            "routing reset_seed = 40 k = 6",
            "routing reset_seed = 45 k=6",
            "env_reset=50 uncertainty k=6",
            "routing reset_seed = 55 k = 6 policy lr = 0.01",
            "routing reset_seed = 65 k = 6 policy lr = 0.01",
        ],
        "AORPO": [
            "no routing aorpo rest_seed = 30 k = 6",
            "no routing aorpo rest_seed = 40",
            "no routing reset_seed = 45",
            "no routing aorpo reset_seed = 50 k = 6",
            "no routing reset_seed = 55",
            "no routing reset_seed = 65",
        ],
    },
    "MPE Simple Facmac":{
        "Uncertainty-Aware": [
            "env_reset=40 uncertainty alpha50 q1e-3 quantile_opp: 0.95",
            "env_reset=30 uncertainty alpha50 q1e-3 quantile_opp: 0.95",
            "env_reset=45 uncertainty alpha50 q1e-3 quantile_opp: 0.95"
        ],
        "AORPO": [
            "env_reset=40 aorpo alpha50",
            "env_reset=30 aorpo alpha50",
            "env_reset=45 aorpo alpha50"
        ]
    }
}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================
# 工具函数
# =========================

def fetch_history(run, metric):
    hist = list(run.scan_history(keys=[STEP_KEY, metric]))
    if len(hist) == 0:
        return None

    df = pd.DataFrame(hist)

    if metric not in df.columns:
        return None

    df = df[[STEP_KEY, metric]].dropna()
    df = df.sort_values(STEP_KEY)
    df = df.drop_duplicates(subset=[STEP_KEY])

    return df


def interpolate(df, metric):

    x = df[STEP_KEY].to_numpy()
    y = df[metric].to_numpy()

    if len(x) < 2:
        return None

    y_interp = np.interp(COMMON_STEPS, x, y, left=y[0], right=y[-1])
    return y_interp


def aggregate(y_list):

    ys = np.stack(y_list)

    mean = np.nanmean(ys, axis=0)
    std = np.nanstd(ys, axis=0)

    return mean, std


# =========================
# 主逻辑
# =========================

def main():

    api = wandb.Api()

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    for i, (env_name, project) in enumerate(PROJECTS.items()):

        metric = METRICS[env_name]

        print(f"\nProcessing environment: {env_name}")

        runs = api.runs(f"{ENTITY}/{project}")

        run_dict = {run.name: run for run in runs}

        ax = axes[i]

        for group_name, run_names in RUN_GROUPS[env_name].items():

            y_list = []

            for name in run_names:

                if name not in run_dict:
                    continue

                run = run_dict[name]

                df = fetch_history(run, metric)
                if df is None:
                    continue

                y = interpolate(df, metric)
                if y is None:
                    continue

                y_list.append(y)

            if len(y_list) == 0:
                continue

            mean, std = aggregate(y_list)

            ax.plot(
                COMMON_STEPS,
                mean,
                linewidth=2.5,
                label=group_name
            )

            ax.fill_between(
                COMMON_STEPS,
                mean - std,
                mean + std,
                alpha=0.2
            )

        ax.set_title(env_name)
        ax.set_xlabel("Environment Interactions")

        if i == 0:
            ax.set_ylabel("Episode Reward")

        ax.grid(alpha=0.3)

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f'{int(x/1000)}K' if x != 0 else "0")
        )

    # axes[0].legend(frameon=False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))

    plt.savefig(os.path.join(OUT_DIR, "reward_vs_steps.pdf"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "reward_vs_steps.png"), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()