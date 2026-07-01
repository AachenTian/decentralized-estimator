import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

ENTITY = "yachen-tian-rwth-aachen-university"
PROJECT = "AORPO-dynamics model"

STEP_KEY = "_step"
METRIC = "episode_reward_dyna"

COMMON_STEPS = np.arange(0, 34000, 1400)

RUN_GROUPS = {
    "Uncertainty-Aware": [
        "routing reset_seed = 30 k = 6 policy lr = 0.01",
        "routing reset_seed = 40 k = 6",
        "routing reset_seed = 45 k=6",
        # "env_reset=50 uncertainty k=6",
        # "routing reset_seed = 55 k = 6 policy lr = 0.01",
        # "routing reset_seed = 65 k = 6 policy lr = 0.01",
    ],

    "Comm-Only": [
        "env_reset=30 aorpo_rollout k",
        "env_reset=40 aorpo_rollout k",
        "env_reset=45 aorpo_rollout k",
    ],

    "Termination-Only": [
        "env_reset=30 aorpo_comm",
        "env_reset=40 aorpo_comm",
        "env_reset=45 aorpo_comm",
    ],

    "AORPO": [
        "no routing aorpo rest_seed = 30 k = 6",
        "no routing aorpo rest_seed = 40",
        "no routing reset_seed = 45",
        # "no routing aorpo reset_seed = 50 k = 6",
        # "no routing reset_seed = 55",
        # "no routing reset_seed = 65",
    ],
}

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

run_dict = {run.name: run for run in runs}

def fetch_history(run):

    hist = list(run.scan_history(keys=[STEP_KEY, METRIC]))

    if len(hist) == 0:
        return None

    df = pd.DataFrame(hist)

    if METRIC not in df.columns:
        return None

    df = df[[STEP_KEY, METRIC]].dropna()
    df = df.sort_values(STEP_KEY)

    return df


def interpolate(df):

    x = df[STEP_KEY].to_numpy()
    y = df[METRIC].to_numpy()

    return np.interp(COMMON_STEPS, x, y, left=y[0], right=y[-1])


def aggregate(curves):

    curves = np.stack(curves)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    return mean, std


plt.figure(figsize=(5,4))

for group, names in RUN_GROUPS.items():

    curves = []

    for name in names:

        if name not in run_dict:
            continue

        run = run_dict[name]

        df = fetch_history(run)

        if df is None:
            continue

        curves.append(interpolate(df))

    if len(curves) == 0:
        continue

    mean, std = aggregate(curves)

    colors = {
        "Uncertainty-Aware": "tab:blue",
        "Comm-Only": "tab:orange",
        "Termination-Only": "tab:green",
        "AORPO": "gray",
    }

    color = colors[group]
    plt.plot(COMMON_STEPS, mean, linewidth=2.8, label=group, color=color)
    plt.fill_between(COMMON_STEPS, mean-std, mean+std, alpha=0.12, color=color)

plt.xlabel("Environment Interactions")
plt.ylabel("Episode Reward")

plt.xlim(0, 33000)

from matplotlib.ticker import FuncFormatter
plt.gca().xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{int(x/1000)}K' if x != 0 else '0')
)

plt.grid(alpha=0.3)

handles, labels = plt.gca().get_legend_handles_labels()

fig = plt.gcf()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False
)


plt.tight_layout(rect=(0,0.12,1,1))

plt.savefig("ablation_spread.pdf", dpi=300)
plt.show()