import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def process_eval_file(file_path):
    data = load_data(file_path)
    df = pd.DataFrame(data)
    return df


def process_stats_file(file_path):
    data = load_data(file_path)
    # Extract only the average scores
    df = pd.DataFrame(
        {k: v["average_score"] for k, v in data.items() if "average_score" in v},
        index=[0],
    )
    return df


def aggregate_data(files):
    eval_data = []
    stats_data = []

    for file in tqdm(files):
        option, *dataset, repeat_info = file.split("_")
        dataset = "_".join(dataset)
        repeat = repeat_info.split("-")[0]
        file_type = repeat_info.split("-")[1].split(".")[0]

        if file_type == "eval":
            df = process_eval_file(file)
            df["option"] = option
            df["dataset"] = dataset
            df["repeat"] = repeat
            eval_data.append(df)
        elif file_type == "stats":
            df = process_stats_file(file)
            df["option"] = option
            df["dataset"] = dataset
            df["repeat"] = repeat
            stats_data.append(df)

    eval_df = pd.concat(eval_data, ignore_index=True)
    stats_df = pd.concat(stats_data, ignore_index=True)

    return eval_df, stats_df


def plot_turns_with_std(df):
    grouped = (
        df.groupby(["option", "dataset"])["turns"].agg(["mean", "std"]).reset_index()
    )

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(grouped))

    plt.bar(
        index,
        grouped["mean"],
        bar_width,
        yerr=grouped["std"],
        capsize=5,
        label="Mean Turns with Std Dev",
    )

    plt.xlabel("Experiment Condition")
    plt.ylabel("Number of Turns")
    plt.title("Mean Turns with Standard Deviation by Experiment Condition")
    plt.xticks(
        index,
        [f"{row['option']}_{row['dataset']}" for _, row in grouped.iterrows()],
        rotation=45,
        ha="right",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("turns_with_std_dev.png")
    plt.close()


def plot_clock_seconds_with_std(df):
    grouped = (
        df.groupby(["option", "dataset"])["clockSeconds"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(grouped))

    plt.bar(
        index,
        grouped["mean"],
        bar_width,
        yerr=grouped["std"],
        capsize=5,
        label="Mean Clock Seconds with Std Dev",
    )

    plt.xlabel("Experiment Condition")
    plt.ylabel("Clock Seconds")
    plt.title("Mean Clock Seconds with Standard Deviation by Experiment Condition")
    plt.xticks(
        index,
        [f"{row['option']}_{row['dataset']}" for _, row in grouped.iterrows()],
        rotation=45,
        ha="right",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("clock_seconds_with_std_dev.png")
    plt.close()


def plot_decision_success_with_std(df):
    if "decisionSuccess" not in df.columns:
        print(
            "Warning: 'decisionSuccess' column not found. Skipping decision success plot."
        )
        return

    df["decision_success_numeric"] = df["decisionSuccess"].map({True: 1, False: 0})
    grouped = (
        df.groupby(["option", "dataset"])["decision_success_numeric"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(grouped))

    plt.bar(
        index,
        grouped["mean"],
        bar_width,
        yerr=grouped["std"],
        capsize=5,
        label="Mean Decision Success Rate with Std Dev",
    )

    plt.xlabel("Experiment Condition")
    plt.ylabel("Decision Success Rate")
    plt.title(
        "Mean Decision Success Rate with Standard Deviation by Experiment Condition"
    )
    plt.xticks(
        index,
        [f"{row['option']}_{row['dataset']}" for _, row in grouped.iterrows()],
        rotation=45,
        ha="right",
    )
    plt.legend()
    plt.ylim(0, 1)  # Set y-axis limits for percentage
    plt.tight_layout()
    plt.savefig("decision_success_with_std_dev.png")
    plt.close()


def plot_score_distributions_with_std(df):
    print("Shape of stats_df:", df.shape)
    print("Columns in stats_df:", df.columns)
    print("First few rows of stats_df:")
    print(df.head())

    # Check if 'option' and 'dataset' columns exist
    if "option" not in df.columns or "dataset" not in df.columns:
        print(
            "Warning: 'option' or 'dataset' columns not found in stats data. Unable to create score distribution plots."
        )
        return

    # Melt the dataframe, excluding 'option', 'dataset', and 'repeat' columns
    id_vars = ["option", "dataset", "repeat"]
    value_vars = [col for col in df.columns if col not in id_vars]
    melted_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Score Type",
        value_name="Score",
    )

    # Group by 'option', 'dataset', and 'Score Type', then calculate mean and std
    grouped = (
        melted_df.groupby(["option", "dataset", "Score Type"])["Score"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Create a separate plot for each Score Type
    for score_type in grouped["Score Type"].unique():
        plt.figure(figsize=(10, 6))

        # Filter data for the current score type
        score_data = grouped[grouped["Score Type"] == score_type]

        # Create bar plot
        x = range(len(score_data))
        plt.bar(
            x,
            score_data["mean"],
            yerr=score_data["std"],
            capsize=5,
            color=plt.cm.Set3(range(len(score_data))),
        )  # Use a color cycle

        plt.xlabel("Experiment Condition")
        plt.ylabel("Average Score")
        plt.title(f"Mean {score_type} Score with Standard Deviation")
        plt.xticks(
            x,
            [f"{row['option']}_{row['dataset']}" for _, row in score_data.iterrows()],
            rotation=45,
            ha="right",
        )

        # Add value labels on top of each bar
        for i, v in enumerate(score_data["mean"]):
            plt.text(
                i, v + score_data["std"].iloc[i], f"{v:.2f}", ha="center", va="bottom"
            )

        plt.tight_layout()
        plt.savefig(f'{score_type.replace(" ", "_").lower()}_score.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM discussion data and create plots."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing JSON files"
    )
    args = parser.parse_args()

    files = [f for f in os.listdir(args.input_folder) if f.endswith((".json"))]
    eval_df, stats_df = aggregate_data(files)

    print("Shape of eval_df:", eval_df.shape)
    print("Columns in eval_df:", eval_df.columns)
    print("First few rows of eval_df:")
    print(eval_df.head())

    available_columns = eval_df.columns

    if "turns" in available_columns:
        plot_turns_with_std(eval_df)
    else:
        print("Warning: 'turns' column not found. Skipping turns plot.")

    if "clockSeconds" in available_columns:
        plot_clock_seconds_with_std(eval_df)
    else:
        print("Warning: 'clockSeconds' column not found. Skipping clock seconds plot.")

    plot_decision_success_with_std(eval_df)

    if not stats_df.empty:
        plot_score_distributions_with_std(stats_df)
    else:
        print("Warning: No stats data available. Skipping score distributions plot.")


if __name__ == "__main__":
    main()
