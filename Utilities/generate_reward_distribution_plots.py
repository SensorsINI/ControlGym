import os
import numpy as np
import glob

import matplotlib.pyplot as plt
plt.style.use(["science"])

# List hyperparameter names and values
fig_title = r"RPGD Ablation on Double Inverted Pendulum"
hp_name = r"\# parallel rollouts"
hp_values = ["1", "8", "32", "128", "256"]

# List GUILD AI hashes of runs to plot
GUILD_AI_HASHES = [
    "d1cc6b21",
    "dbf31c93",
    "4c8c1726",
    "d72d8392",
    "b5138d98",
]
PATHS_TO_EXPERIMENTS = [os.path.join(".env", ".guild", "runs", h) for h in GUILD_AI_HASHES]

# Or specify paths manually, like so:
# PATHS_TO_EXPERIMENTS = [
#     ".env/.guild/runs/4c8c1726",
# ]

num_experiments = len(PATHS_TO_EXPERIMENTS)
all_rewards_data = []

if __name__ == "__main__":
    for path_to_experiment in PATHS_TO_EXPERIMENTS:
        output_scalars_files = glob.glob(f"{path_to_experiment}*{os.sep}Output{os.sep}**{os.sep}*output_scalars*.csv", recursive=True)
        assert len(output_scalars_files) == 1
        
        output_scalars_of_experiment = np.loadtxt(
            output_scalars_files[0],
            dtype=np.float32,
            delimiter=",",
            skiprows=1,
        )
        
        rewards_of_experiment = output_scalars_of_experiment[:, 0]
        all_rewards_data.append(rewards_of_experiment)
        
    all_rewards_data = np.stack(all_rewards_data, axis=0)  # One row for each experiment
    num_trials = all_rewards_data.shape[1]
    
    fig, ax = plt.subplots(
        figsize=(4, 3),
        dpi=300.0,
    )
    
    for i in range(num_experiments):
        scatter_y = np.repeat(i + 1, num_trials) + np.clip(0.07 * np.random.standard_normal(num_trials), -0.35, 0.35)
        r = all_rewards_data[i, :]

        ax.scatter(r, scatter_y, marker=".", c="b", s=4)
        ax.plot([np.mean(r), np.mean(r)], [i + 1 - 0.35, i + 1 + 0.35], c="r", linewidth=0.75)
        
    ax.set_xlabel("realized mean reward per episode\nmore positive is better")
    ax.set_ylabel(hp_name)
    ax.set_yticks(np.arange(1, num_experiments + 1), labels=hp_values)
    ax.set_title(fig_title, fontdict={"fontsize": 9}, pad=3.0)
    # ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=1.03)

    savedir = os.path.join("Output", "cost_scatter_plots")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    fig.savefig(os.path.join(savedir, fig_title + "_" + "_".join(GUILD_AI_HASHES) + ".pdf"), bbox_inches="tight", pad_inches=0.03)