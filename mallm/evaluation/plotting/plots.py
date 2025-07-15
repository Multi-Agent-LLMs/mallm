import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Set the style for beautiful plots
plt.style.use('seaborn-v0_8-pastel')
sns.set_palette("pastel")

# Define a beautiful color palette
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']


def get_colors(n_colors: int) -> Union[list[str], np.ndarray[Any, Any]]:
    """Generate enough colors for n_colors by cycling or using colormap"""
    if n_colors <= len(COLORS):
        return COLORS[:n_colors]
    # Use a colormap for more colors
    return plt.cm.Set3(np.linspace(0, 1, n_colors))  # type: ignore


def get_consistent_color_mapping(options: list[str]) -> dict[str, Any]:
    """Create consistent color mapping based on option names"""
    # Sort options to ensure consistent assignment
    unique_options = sorted(set(options))

    # Generate enough colors
    if len(unique_options) <= len(COLORS):
        colors = COLORS[:len(unique_options)]
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_options)))

    # Create mapping
    return dict(zip(unique_options, colors))


def process_eval_file(file_path: str) -> pd.DataFrame:
    data = json.loads(Path(file_path).read_text())
    return pd.DataFrame(data)


def format_metric(text: str) -> str:
    text = text.replace("_", " ")
    return text.capitalize().replace("Correct", "Accuracy").replace("Correct", "Accuracy").replace("Rougel", "ROUGE-L").replace("Rouge1", "ROUGE-1").replace("Rouge2", "ROUGE-2").replace("Bleu", "BLEU").replace("Distinct1", "Distinct-1").replace("Distinct2", "Distinct-2")


def process_stats_file(file_path: str) -> pd.DataFrame:
    data = json.loads(Path(file_path).read_text())
    # Extract only the average scores
    return pd.DataFrame(
        {k: v["averageScore"] for k, v in data.items() if "averageScore" in v},
        index=[0],
    )


def aggregate_data(
    files: list[str], input_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_data = []
    stats_data = []

    for file in tqdm(files):
        try:
            *option, dataset, repeat_info = file.split("_")
            option = "_".join(option)
            repeat = repeat_info.split("-")[0]
            file_type = repeat_info.split("-")[1].split(".")[0]
        except IndexError:
            continue

        if file_type == "eval":
            df = process_eval_file(f"{input_path}/{file}")
            df["option"] = option
            df["dataset"] = dataset
            df["repeat"] = repeat
            eval_data.append(df)
        elif file_type == "stats":
            df = process_stats_file(f"{input_path}/{file}")
            df["option"] = option
            df["dataset"] = dataset
            df["repeat"] = repeat
            stats_data.append(df)

    eval_df = pd.concat(eval_data, ignore_index=True)
    stats_df = pd.concat(stats_data, ignore_index=True)

    return eval_df, stats_df


def plot_turns_with_std(df: pd.DataFrame, input_path: str, global_color_mapping: Optional[dict[str, Any]] = None) -> None:
    """Create a beautiful violin plot for turns distribution"""
    # Filter out rows with missing or invalid turns data
    df = df.dropna(subset=['turns'])
    df = df[df['turns'].notna() & (df['turns'] >= 0)]

    if df.empty:
        print("Warning: No valid turns data found. Skipping turns plot.")
        return

    # Create grouped data like other plots for consistent color assignment
    grouped_data = df.groupby(['option', 'dataset']).agg({
        'turns': list  # Keep all turns values for violin plot
    }).reset_index()

    # Create unique labels like other plots
    unique_labels = get_unique_labels(grouped_data)
    grouped_data['label'] = unique_labels

    # Use global color mapping if provided, otherwise create local one
    if global_color_mapping is None:
        color_mapping = get_consistent_color_mapping(grouped_data['option'].unique().tolist())
    else:
        color_mapping = global_color_mapping

    # Create color palette based on option order in grouped data
    colors = [color_mapping[option] for option in grouped_data['option']]

    # Expand the grouped data back to individual rows for violin plot
    expanded_data = []
    for i, row in grouped_data.iterrows():
        expanded_data.extend({
                'option': row['option'],
                'dataset': row['dataset'],
                'label': row['label'],
                'turns': turn_value
            } for turn_value in row['turns'])

    plot_df = pd.DataFrame(expanded_data)

    plt.figure(figsize=(10, 4))

    # Create violin plot with the same label order as other plots
    ax = sns.violinplot(data=plot_df, x='label', y='turns',
                       order=grouped_data['label'], palette=colors,
                       inner=None, legend=False)

    # Add individual points with jitter
    sns.stripplot(data=plot_df, x='label', y='turns',
                  order=grouped_data['label'], color='white', size=6,
                  edgecolor='black', linewidth=0.5)

    # Set all plot elements above grid
    for collection in ax.collections:
        collection.set_zorder(4)

    # Add red diamond mean markers that align correctly with violin plots
    for i, label in enumerate(grouped_data['label']):
        mean_val = plot_df[plot_df['label'] == label]['turns'].mean()
        # Use red diamond markers positioned correctly
        ax.plot(i, mean_val, marker='D', color='red', markersize=8,
                markeredgecolor='white', markeredgewidth=1, zorder=5)

    # Styling
    ax.set_xlabel('')  # Remove automatic seaborn x-axis label
    ax.set_ylabel('Number of Turns', fontsize=14)

    # Rotate labels and improve spacing
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    ax.grid(True, alpha=0.3, zorder=0)
    # Add a subtle background
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Reduced space for rotated labels
    plt.savefig(f"{input_path}/turns_distribution.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{input_path}/turns_distribution.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_clock_seconds_with_std(df: pd.DataFrame, input_path: str, global_color_mapping: Optional[dict[str, Any]] = None) -> None:
    """Create a beautiful horizontal lollipop chart for clock seconds"""
    grouped = (
        df.groupby(["option", "dataset"])["clockSeconds"]
        .agg(["mean", "std"])
        .reset_index()
    )

    unique_labels = get_unique_labels(grouped)
    grouped['label'] = unique_labels

    # Sort data: baselines first, then others by shortest time
    def sort_key(row: pd.Series) -> tuple[int, float]:
        option = row['option'].lower()
        if option.startswith('baseline'):
            return (0, row['mean'])  # Baselines first, sorted by time
        return (1, row['mean'])  # Others after, sorted by time (shortest first)

    grouped['sort_key'] = grouped.apply(sort_key, axis=1)
    grouped = grouped.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
    # Reverse the entire order
    grouped = grouped.iloc[::-1].reset_index(drop=True)

    _fig, ax = plt.subplots(figsize=(10, 4))

    # Create discrete marker chart (no stems)
    y_pos = np.arange(len(grouped))

    # Use global color mapping if provided, otherwise create local one
    if global_color_mapping is None:
        color_mapping = get_consistent_color_mapping(grouped['option'].unique().tolist())
    else:
        color_mapping = global_color_mapping
    colors = [color_mapping[option] for option in grouped['option']]

    # Draw discrete circular markers only
    ax.scatter(grouped['mean'], y_pos,
                        s=250, c=colors,
                        edgecolors='white', linewidth=3, zorder=5)

    # Add error bars
    ax.errorbar(grouped['mean'], y_pos, xerr=grouped['std'],
                fmt='none', color='gray', capsize=6, linewidth=2, zorder=4)

    # Add value labels with better positioning to avoid circle overlap
    for i, (_, row) in enumerate(grouped.iterrows()):
        # Calculate offset to avoid overlap with circle (larger offset)
        offset = max(row['std'] + max(grouped['mean']) * 0.08, max(grouped['mean']) * 0.05)
        ax.text(row['mean'] + offset, i,
                f'{row["mean"]:.1f}s',
                va='center', ha='left', fontsize=14, zorder=6)

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(grouped['label'], fontsize=14)
    ax.set_xlabel('Execution Time (seconds)', fontsize=14)

    # Set x-axis limits with proper margins for labels
    max_val = max(grouped['mean'] + grouped['std'])
    ax.set_xlim(0, max_val * 1.3)  # Extra space for non-overlapping labels

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, alpha=0.3, axis='x', zorder=0)
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.savefig(f"{input_path}/clock_seconds.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{input_path}/clock_seconds.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_decision_success_with_std(df: pd.DataFrame, input_path: str, global_color_mapping: Optional[dict[str, Any]] = None) -> None:
    """Create a beautiful horizontal bar chart for decision success rates"""
    if "decisionSuccess" not in df.columns:
        print(
            "Warning: 'decisionSuccess' column not found. Skipping decision success plot."
        )
        return

    # Filter out rows with missing or invalid decision success data
    df = df.dropna(subset=['decisionSuccess'])
    df = df[df['decisionSuccess'].notna()]

    if df.empty:
        print("Warning: No valid decision success data found. Skipping decision success plot.")
        return

    df["decision_success_numeric"] = df["decisionSuccess"].map({True: 1, False: 0})
    grouped = (
        df.groupby(["option", "dataset"])["decision_success_numeric"]
        .agg(["mean", "std"])
        .reset_index()
    )

    unique_labels = get_unique_labels(grouped)
    grouped['label'] = unique_labels
    grouped = grouped.sort_values('mean')

    _fig, ax = plt.subplots(figsize=(10, 3))

    # Use global color mapping if provided, otherwise create local one
    if global_color_mapping is None:
        color_mapping = get_consistent_color_mapping(grouped['option'].unique().tolist())
    else:
        color_mapping = global_color_mapping
    colors = [color_mapping[option] for option in grouped['option']]

    # Create horizontal bars
    bars = ax.barh(range(len(grouped)), grouped['mean'],
                   color=colors, height=0.6, zorder=3)

    # Add percentage labels on bars
    for i, (_, row) in enumerate(grouped.iterrows()):
        percentage = row['mean'] * 100
        ax.text(row['mean'] + 0.02, i, f'{percentage:.1f}%',
                va='center', ha='left', fontsize=14, zorder=6)

    # Add a subtle pattern to bars
    for bar, rate in zip(bars, grouped['mean']):
        if rate < 0.5:  # Add pattern for low success rates
            bar.set_hatch('///')

    # Styling
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped['label'], fontsize=14)
    ax.set_xlabel('Decision Success Rate', fontsize=14)
    ax.set_xlim(0, 1.1)

    # Add percentage ticks
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=14)

    # Remove spines and add grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x', zorder=0)
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.savefig(f"{input_path}/decision_success.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{input_path}/decision_success.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def get_unique_labels(df: pd.DataFrame) -> list[str]:
    labels = [f"{row['option']}_{row['dataset']}" for _, row in df.iterrows()]
    # Extract unique parts by finding the longest common prefix and suffix
    if not labels:
        return []

    # Find the longest common prefix
    common_prefix = ""
    if labels:
        first_label = labels[0]
        for i in range(len(first_label)):
            if all(label.startswith(first_label[:i + 1]) for label in labels):
                common_prefix = first_label[:i + 1]
            else:
                break

    # Find the longest common suffix
    common_suffix = ""
    if labels:
        first_label = labels[0]
        for i in range(len(first_label)):
            if all(label.endswith(first_label[-(i + 1):]) for label in labels):
                common_suffix = first_label[-(i + 1):]
            else:
                break

    # Extract unique parts by removing common prefix and suffix
    unique_labels = []
    for label in labels:
        unique_part = label
        if common_prefix and label.startswith(common_prefix):
            unique_part = unique_part[len(common_prefix):]
        if common_suffix and unique_part.endswith(common_suffix):
            unique_part = unique_part[:-len(common_suffix)]
        unique_labels.append(format_metric(unique_part))

    return unique_labels


def get_unique_labels_from_conditions(conditions: Union[list[str], np.ndarray[Any, Any]]) -> list[str]:
    """Helper function to get unique labels from condition strings"""
    # Convert to list if it's a numpy array
    condition_list: list[str]
    if hasattr(conditions, 'tolist'):
        condition_list = conditions.tolist()
    else:
        condition_list = conditions

    if len(condition_list) == 0:
        return []

    # Find the longest common prefix
    common_prefix = ""
    if len(condition_list) > 0:
        first_condition = condition_list[0]
        for i in range(len(first_condition)):
            if all(condition.startswith(first_condition[:i + 1]) for condition in condition_list):
                common_prefix = first_condition[:i + 1]
            else:
                break

    # Find the longest common suffix
    common_suffix = ""
    if len(condition_list) > 0:
        first_condition = condition_list[0]
        for i in range(len(first_condition)):
            if all(condition.endswith(first_condition[-(i + 1):]) for condition in condition_list):
                common_suffix = first_condition[-(i + 1):]
            else:
                break

    # Extract unique parts by removing common prefix and suffix
    unique_labels = []
    for condition in condition_list:
        unique_part = condition
        if common_prefix and condition.startswith(common_prefix):
            unique_part = unique_part[len(common_prefix):]
        if common_suffix and unique_part.endswith(common_suffix):
            unique_part = unique_part[:-len(common_suffix)]
        unique_labels.append(format_metric(unique_part))

    return unique_labels


def plot_score_distributions_with_std(df: pd.DataFrame, input_path: str, global_color_mapping: Optional[dict[str, Any]] = None) -> None:
    """Create beautiful enhanced bar charts for score distributions"""
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
        _fig, ax = plt.subplots(figsize=(10, 4))

        # Filter data for the current score type
        score_data = grouped[grouped["Score Type"] == score_type].copy()

        # Sort data: baselines first, then alphabetically
        def sort_key(row: pd.Series) -> tuple[int, str]:
            option = row['option'].lower()
            if option.startswith('baseline'):
                return (0, option)  # Baselines first
            return (1, option)  # Others after

        score_data['sort_key'] = score_data.apply(sort_key, axis=1)
        score_data = score_data.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)

        score_data.to_csv(
            f'{input_path}/{score_type.replace(" ", "_").lower()}_score.csv',
            index=False,
        )

        # Create beautiful bar plot with consistent colors
        x = np.arange(len(score_data))

        # Use global color mapping if provided, otherwise create local one
        if global_color_mapping is None:
            color_mapping = get_consistent_color_mapping(score_data['option'].unique().tolist())
        else:
            color_mapping = global_color_mapping
        colors = [color_mapping[option] for option in score_data['option']]

        bars = ax.bar(x, score_data["mean"],
                     yerr=score_data["std"],
                     capsize=8,
                     color=colors,
                     width=0.6, zorder=3)  # Slightly narrower bars for more discrete look

        # Calculate proper y-axis limits
        max_height = max(score_data["mean"] + score_data["std"])
        min_val = min(0, *(score_data['mean'] - score_data['std']))
        y_range = max_height - min_val

        # Add value labels on top of each bar with better positioning
        for _i, (bar, mean_val, std_val) in enumerate(zip(bars, score_data["mean"], score_data["std"])):
            height = mean_val + std_val
            ax.text(bar.get_x() + bar.get_width() / 2., height + y_range * 0.05,
                   f'{mean_val:.3f}', ha='center', va='bottom',
                   fontsize=14, zorder=6)

        # Styling
        ax.set_ylabel('Average Score', fontsize=14)

        # Set x-axis with proper spacing and labels
        ax.set_xticks(x)
        ax.set_xticklabels(get_unique_labels(score_data), rotation=45, ha='right', fontsize=14)

        # Set proper axis limits to prevent cut-off and add margins
        ax.set_xlim(-0.6, len(x) - 0.4)  # Add margins on both sides
        ax.set_ylim(min_val - y_range * 0.05, max_height + y_range * 0.15)

        # Add grid and styling
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        ax.set_facecolor('#fafafa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add a subtle shadow effect
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#cccccc')

        # Improve layout with better margins
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Reduced space for rotated labels
        plt.savefig(f'{input_path}/{score_type.replace(" ", "_").lower()}_score.png',
                   dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(f'{input_path}/{score_type.replace(" ", "_").lower()}_score.pdf',
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()


def create_plots_for_path(input_dir_path: str, output_dir_path: str) -> None:
    files = [f for f in os.listdir(input_dir_path) if f.endswith(".json")]
    eval_df, stats_df = aggregate_data(files, input_dir_path)

    print("Shape of eval_df:", eval_df.shape)
    print("Columns in eval_df:", eval_df.columns)
    print("First few rows of eval_df:")
    print(eval_df.head())

    # Create global color mapping for all options across all plots
    all_options = set()
    if not eval_df.empty and 'option' in eval_df.columns:
        all_options.update(eval_df['option'].unique())
    if not stats_df.empty and 'option' in stats_df.columns:
        all_options.update(stats_df['option'].unique())

    global_color_mapping = get_consistent_color_mapping(list(all_options))

    available_columns = eval_df.columns

    if "turns" in available_columns:
        plot_turns_with_std(eval_df, output_dir_path, global_color_mapping)
    else:
        print("Warning: 'turns' column not found. Skipping turns plot.")

    if "clockSeconds" in available_columns:
        plot_clock_seconds_with_std(eval_df, output_dir_path, global_color_mapping)
    else:
        print("Warning: 'clockSeconds' column not found. Skipping clock seconds plot.")

    plot_decision_success_with_std(eval_df, output_dir_path, global_color_mapping)

    if not stats_df.empty:
        plot_score_distributions_with_std(stats_df, output_dir_path, global_color_mapping)
    else:
        print("Warning: No stats data available. Skipping score distributions plot.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM discussion data and create plots."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing JSON files"
    )
    args = parser.parse_args()
    input_folder: str = args.input_folder.removesuffix("/")

    create_plots_for_path(input_folder, input_folder)


if __name__ == "__main__":
    main()
