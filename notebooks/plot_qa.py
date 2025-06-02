import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

# Set up plot configuration to use Carlito font, if available
plt.rcParams.update({"font.family": "carlito"})

# Font size 20 color #595959
COLOR = "#595959"
plt.rcParams.update(
    {
        "font.size": 20,
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
    }
)

# Updated data with precomputed CIs
data = {
    "2 Agents": {"mean": 43.1, "ci": 2.5},
    "3 Agents": {"mean": 42.5, "ci": 2.5},
    "4 Agents": {"mean": 41.1, "ci": 2.5},
    "5 Agents": {"mean": 43.0, "ci": 3.4},
    #"6 Agents": {"mean": 29.92, "ci": 2.7},
}

# Convert percentages to proportions
for k in data:
    data[k]["mean"] /= 100
    data[k]["ci"] /= 100

# Create boxplot data with lower and upper CI
boxplot_data = [
    (v["mean"] - v["ci"], v["mean"], v["mean"] + v["ci"]) for v in data.values()
]

# Plot with swapped axes and standard colors
fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(
    boxplot_data, vert=True, patch_artist=True, labels=data.keys(), widths=0.2
)

# Setting standard color for boxes
for patch in bp["boxes"]:
    patch.set_facecolor("#0f96d4")  # Using a single color for simplicity
for median in bp["medians"]:
    median.set_color("black")  # Changing median line to black for better visibility

loc = plticker.MultipleLocator(
    base=0.01
)  # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)
# Adding grid for better readability
ax.yaxis.grid(True, linestyle="--", which="both", color="grey", alpha=0.6)

# Adding baseline data
baseline_mean = 39.5 / 100
baseline_ci = 0 / 100
ax.hlines(
    baseline_mean,
    xmin=0.5,
    xmax=4.5,
    colors="gray",
    linestyles="dashed",
    label="Baseline (Meta)",
)
ax.fill_between(
    [0.5, 4.5],
    baseline_mean - baseline_ci,
    baseline_mean + baseline_ci,
    color="gray",
    alpha=0.1,
)

# Convert y-axis to percentages
ax.set_yticklabels(["{:.0%}".format(x) for x in ax.get_yticks()])

# Labels and legend
ax.set_ylabel("Accuracy")
ax.legend()

plt.savefig("Figue1-final.png", dpi=300)
