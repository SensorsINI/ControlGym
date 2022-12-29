import os
import numpy as np
import glob
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(["science"])

"""
Save a scatter plot which compares the effect of a hyperparameter on the distribution of realized control rewards.
This script assumes that the GUILD AI library was used to generate the results.
You specify a list of hashes to compare.
Each hash references a GUILD AI experiment with e.g. 100 randomized episodes.
The randomized episodes have different initialization and seeds, so that you can be sure the effect of a hyperparameter you get is not a random outlier.
Typically, all the runs in your list used the same hyperparameters, except one that was varied.
You then have to manually define the name of the parameter and the values you gave it. (This is not extracted automatically).
The resulting plot has one row for each GUILD hash, so that e.g. 100 points in the row show the distribution of control rewards obtained during the 100 random episodes.
On the horizontal axis, you see the scale of the rewards.
"""

### List hyperparameter names and values
fig_title = r"ME-RPGD Ablation on 3D Point Mass"
hp_name = r"sampling distribution"
hp_values = ["normal", "uniform"]

### List GUILD AI hashes of runs to plot
GUILD_AI_HASHES = [
    "ae0bca4d",
    "58115d1a",
    # "09d4e1ba",
    # ...
]

### Make sure that ".env" below is the path to your python virtual environment
PATHS_TO_EXPERIMENTS = [os.path.join(".env", ".guild", "runs", h) for h in GUILD_AI_HASHES]

### Or specify paths manually, like so:
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
    
    all_x = []
    all_y = []
    
    for i in range(num_experiments):
        scatter_y = np.repeat(i + 1, num_trials) + np.clip(0.07 * np.random.standard_normal(num_trials), -0.35, 0.35)
        r = all_rewards_data[i, :]
        
        all_x.extend(list(r))
        all_y.extend(list(scatter_y))

        ax.scatter(r, scatter_y, marker=".", c="b", s=4)
        ax.plot([np.mean(r), np.mean(r)], [i + 1 - 0.35, i + 1 + 0.35], c="r", linewidth=0.75)
    
    # Save data as .dat file
    df = pd.DataFrame({
        "xcol": all_x,
        "ycol": all_y,
        "metacol": all_x,
    })
    savedir = os.path.join("Output", "cost_scatter_plots")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(os.path.join(savedir, fig_title + "_" + "_".join(GUILD_AI_HASHES) + ".dat"), sep="\t", index=False)
    
    # Save plot file 
    ax.set_xlabel("realized mean reward per episode\nmore positive is better")
    ax.set_ylabel(hp_name)
    ax.set_yticks(np.arange(1, num_experiments + 1), labels=hp_values)
    ax.set_title(fig_title, fontdict={"fontsize": 9}, pad=3.0)
    # ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=1.03)

   
    fig.savefig(os.path.join(savedir, fig_title + "_" + "_".join(GUILD_AI_HASHES) + ".pdf"), bbox_inches="tight", pad_inches=0.03)