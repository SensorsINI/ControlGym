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
fig_title = r"RPGD G"
hp_name = r"Gradient steps"
# hp_values = [0, 2, 5, 10, 50, 200]
# hp_values = ["0", "2", "5", "10", "50", "200"]
# hp_values = ["0", "0r", "2", "2r", "5", "5r", "10", "10r", "15", "15r", "20", "20r", "30", "30r", "50", "50r", "200", "200r"]
hp_values = ["0", "0r", "1", "1r", "2", "2r", "5", "5r", "10", "10r", "15", "15r", "20", "20r", "30", "30r", "50", "50r", "100", "100r", "200", "200r", "500", "500r"]

GUILD_AI_HASHES = [
    "3c03aca2", "d43ec534", "807a6fb8",
    "e2a72f17", "031b33d5", "cdb857e4",
    "05c73ad6", "42600468", "96747e11",
    "c36c87cc", "c9087368", "912ce432",
    "d6f48b22", "b87f4228", "2d6f428f",
    "99acafe8", "f8bc8282", "9b972c9a",
    "282fc807", "caee26e5", "8fa26f36",
    "91a864c2", "18dda127", "11dbf244",
    "b6adbda2", "e6bd239d", "b91835b2",
    "b3135ad8", "9a9ef245", "3211bf2c",
    "6642649a", "69772ab3", "894a4832",
    "59fca9f8", "2df6a51b", "74779d58",
    "26f74f3c", "a6056e2c", "92cfca04",
    "e3204673", "3bd92b0d", "953b8006",
    "0ed503af", "1a80749e", "5b345f6e",
    "027b7535", "f9bf22fc", "acf90273",
    "4531ee78", "0b150f89", "592dad45",
    "fc271905", "cd2f94f6", "cedf7aed",
    "321e2097", "77e60238", "de127cc2",
    "4f1b41f8", "f13e3f37", "f79513dc",
    "395d1560", "e3948911", "288f1566",
    "533e2163", "cd339615", "1479536d",
    "ea44b7ae", "ec4233ed", "ca527b34",
    "56ecab37", "5df6aeb0", "c4f0325a",

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

    counter_single_type = 0
    for i in range(num_experiments):
        if (i//3)%2 == 0:
            colour_mean = "red"
        else:
            colour_mean = "green"

        r = all_rewards_data[i, :]
        # r = r[r > 1]
        all_x.extend(list(r))

        scatter_y = np.repeat(i//3 + 1, len(r)) + np.clip(0.07 * np.random.standard_normal(len(r)), -0.35, 0.35)
        all_y.extend(list(scatter_y))
        if i % 3 == 0:
            ax.scatter(r, scatter_y, marker=".", c="b", s=4)
        ax.plot([np.mean(r), np.mean(r)], [i//3 + 1 - 0.35, i//3 + 1 + 0.35], c=colour_mean, linewidth=0.75)
    
    # Save data as .dat file
    df = pd.DataFrame({
        "xcol": all_x,
        "ycol": all_y,
        "metacol": all_x,
    })
    savedir = os.path.join("Output", "cost_scatter_plots")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # df.to_csv(os.path.join(savedir, fig_title + "_" + "_".join(GUILD_AI_HASHES) + ".dat"), sep="\t", index=False)
    df.to_csv(os.path.join(savedir, fig_title + "_" + "all" + ".dat"), sep="\t", index=False)
    
    # Save plot file
    # ax.set_xlim(left=0.95)
    ax.set_xlabel("realized mean reward per episode\nmore positive is better")
    ax.set_ylabel(hp_name)
    ax.set_yticks(np.arange(1, num_experiments//3 + 1), labels=hp_values)
    ax.set_title(fig_title, fontdict={"fontsize": 9}, pad=3.0)
    # ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=1.03)

   
    # fig.savefig(os.path.join(savedir, fig_title + "_" + "_".join(GUILD_AI_HASHES) + ".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(savedir, fig_title + "_" + "all" + ".pdf"), bbox_inches="tight",
                pad_inches=0.03)