import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Set the style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define a beautiful color palette
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

def get_colors(n_colors):
    """Generate enough colors for n_colors by cycling or using colormap"""
    if n_colors <= len(COLORS):
        return COLORS[:n_colors]
    else:
        # Use a colormap for more colors
        return plt.cm.Set3(np.linspace(0, 1, n_colors))

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


def plot_turns_with_std(df: pd.DataFrame, input_path: str) -> None:
    """Create a beautiful violin plot for turns distribution"""
    # Filter out rows with missing or invalid turns data
    df = df.dropna(subset=['turns'])
    df = df[df['turns'].notna() & (df['turns'] >= 0)]
    
    if df.empty:
        print("Warning: No valid turns data found. Skipping turns plot.")
        return
    
    # Create combination labels for better grouping
    df['condition'] = df['option'] + '_' + df['dataset']
    unique_labels = get_unique_labels_from_conditions(df['condition'].unique())
    
    # Create a mapping from full condition to unique label
    condition_to_label = dict(zip(df['condition'].unique(), unique_labels))
    df['condition_label'] = df['condition'].map(condition_to_label)
    
    plt.figure(figsize=(10, 4))
    
    # Create violin plot with individual points
    ax = sns.violinplot(data=df, x='condition_label', y='turns', 
                       hue='condition_label', palette=get_colors(len(df['condition_label'].unique())), 
                       inner=None, alpha=0.7, legend=False)
    
    # Add individual points with jitter
    sns.stripplot(data=df, x='condition_label', y='turns', 
                  color='white', size=6, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add red diamond mean markers that align correctly with violin plots
    unique_conditions = df['condition_label'].unique()
    
    for i, condition in enumerate(unique_conditions):
        mean_val = df[df['condition_label'] == condition]['turns'].mean()
        # Use red diamond markers positioned correctly
        ax.plot(i, mean_val, marker='D', color='red', markersize=8, 
                markeredgecolor='white', markeredgewidth=1, zorder=10)
    
    # Styling
    ax.set_xlabel('')  # Remove automatic seaborn x-axis label
    ax.set_ylabel('Number of Turns', fontsize=14, fontweight='bold')
    
    # Rotate labels and improve spacing
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    # Add a subtle background
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Reduced space for rotated labels
    plt.savefig(f"{input_path}/turns_distribution.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{input_path}/turns_distribution.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_clock_seconds_with_std(df: pd.DataFrame, input_path: str) -> None:
    """Create a beautiful horizontal lollipop chart for clock seconds"""
    grouped = (
        df.groupby(["option", "dataset"])["clockSeconds"]
        .agg(["mean", "std"])
        .reset_index()
    )
    
    unique_labels = get_unique_labels(grouped)
    grouped['label'] = unique_labels
    
    # Sort data: baselines first, then others by shortest time
    def sort_key(row):
        option = row['option'].lower()
        if option.startswith('baseline'):
            return (0, row['mean'])  # Baselines first, sorted by time
        else:
            return (1, row['mean'])  # Others after, sorted by time (shortest first)
    
    grouped['sort_key'] = grouped.apply(sort_key, axis=1)
    grouped = grouped.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
    # Reverse the entire order
    grouped = grouped.iloc[::-1].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create discrete marker chart (no stems)
    y_pos = np.arange(len(grouped))
    colors = get_colors(len(grouped))
    
    # Draw discrete circular markers only
    scatter = ax.scatter(grouped['mean'], y_pos, 
                        s=250, c=colors, 
                        alpha=0.9, edgecolors='white', linewidth=3, zorder=10)
    
    # Add subtle error bars
    ax.errorbar(grouped['mean'], y_pos, xerr=grouped['std'], 
                fmt='none', color='gray', alpha=0.5, capsize=6, linewidth=2)
    
    # Add value labels with better positioning to avoid circle overlap
    for i, (_, row) in enumerate(grouped.iterrows()):
        # Calculate offset to avoid overlap with circle (larger offset)
        offset = max(row['std'] + max(grouped['mean']) * 0.08, max(grouped['mean']) * 0.05)
        ax.text(row['mean'] + offset, i, 
                f'{row["mean"]:.1f}s', 
                va='center', ha='left', fontweight='bold', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(grouped['label'], fontsize=14)
    ax.set_xlabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    
    # Set x-axis limits with proper margins for labels
    max_val = max(grouped['mean'] + grouped['std'])
    ax.set_xlim(0, max_val * 1.3)  # Extra space for non-overlapping labels
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(f"{input_path}/clock_seconds.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{input_path}/clock_seconds.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_decision_success_with_std(df: pd.DataFrame, input_path: str) -> None:
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
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create gradient colors based on success rate
    colors = plt.cm.RdYlGn(grouped['mean'])
    
    # Create horizontal bars
    bars = ax.barh(range(len(grouped)), grouped['mean'], 
                   color=colors, alpha=0.8, height=0.6)
    
    # Add percentage labels on bars
    for i, (_, row) in enumerate(grouped.iterrows()):
        percentage = row['mean'] * 100
        ax.text(row['mean'] + 0.02, i, f'{percentage:.1f}%', 
                va='center', ha='left', fontweight='bold', fontsize=14)
    
    # Add a subtle pattern to bars
    for bar, rate in zip(bars, grouped['mean']):
        if rate < 0.5:  # Add pattern for low success rates
            bar.set_hatch('///')
    
    # Styling
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped['label'], fontsize=14)
    ax.set_xlabel('Decision Success Rate', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    
    # Add percentage ticks
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=14)
    
    # Remove spines and add grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')
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


def get_unique_labels_from_conditions(conditions) -> list[str]:
    """Helper function to get unique labels from condition strings"""
    # Convert to list if it's a numpy array
    if hasattr(conditions, 'tolist'):
        conditions = conditions.tolist()
    
    if len(conditions) == 0:
        return []

    # Find the longest common prefix
    common_prefix = ""
    if len(conditions) > 0:
        first_condition = conditions[0]
        for i in range(len(first_condition)):
            if all(condition.startswith(first_condition[:i + 1]) for condition in conditions):
                common_prefix = first_condition[:i + 1]
            else:
                break

    # Find the longest common suffix
    common_suffix = ""
    if len(conditions) > 0:
        first_condition = conditions[0]
        for i in range(len(first_condition)):
            if all(condition.endswith(first_condition[-(i + 1):]) for condition in conditions):
                common_suffix = first_condition[-(i + 1):]
            else:
                break

    # Extract unique parts by removing common prefix and suffix
    unique_labels = []
    for condition in conditions:
        unique_part = condition
        if common_prefix and condition.startswith(common_prefix):
            unique_part = unique_part[len(common_prefix):]
        if common_suffix and unique_part.endswith(common_suffix):
            unique_part = unique_part[:-len(common_suffix)]
        unique_labels.append(format_metric(unique_part))

    return unique_labels


def plot_score_distributions_with_std(df: pd.DataFrame, input_path: str) -> None:
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
        fig, ax = plt.subplots(figsize=(10, 4))

        # Filter data for the current score type
        score_data = grouped[grouped["Score Type"] == score_type].copy()
        
        # Sort data: baselines first, then alphabetically
        def sort_key(row):
            option = row['option'].lower()
            if option.startswith('baseline'):
                return (0, option)  # Baselines first
            else:
                return (1, option)  # Others after
        
        score_data['sort_key'] = score_data.apply(sort_key, axis=1)
        score_data = score_data.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
        
        score_data.to_csv(
            f'{input_path}/{score_type.replace(" ", "_").lower()}_score.csv',
            index=False,
        )

        # Create beautiful bar plot with gradient colors
        x = np.arange(len(score_data))
        colors = plt.cm.viridis(np.linspace(0, 1, len(score_data)))
        
        bars = ax.bar(x, score_data["mean"], 
                     yerr=score_data["std"],
                     capsize=8,
                     color=colors, alpha=0.8,
                     edgecolor='white', linewidth=2,
                     width=0.6)  # Slightly narrower bars for more discrete look

        # Calculate proper y-axis limits
        max_height = max(score_data["mean"] + score_data["std"])
        min_val = min(0, min(score_data["mean"] - score_data["std"]))
        y_range = max_height - min_val
        
        # Add value labels on top of each bar with better positioning
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, score_data["mean"], score_data["std"])):
            height = mean_val + std_val
            ax.text(bar.get_x() + bar.get_width()/2., height + y_range * 0.05,
                   f'{mean_val:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Add gradient effect to bars
            gradient = np.linspace(0, 1, 256).reshape(256, -1)
            ax.imshow(gradient, extent=[bar.get_x(), bar.get_x() + bar.get_width(), 
                                      0, bar.get_height()], 
                     aspect='auto', alpha=0.3, cmap='viridis')

        # Styling
        ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
        
        # Set x-axis with proper spacing and labels
        ax.set_xticks(x)
        ax.set_xticklabels(get_unique_labels(score_data), rotation=45, ha='right', fontsize=14)
        
        # Set proper axis limits to prevent cut-off and add margins
        ax.set_xlim(-0.6, len(x) - 0.4)  # Add margins on both sides
        ax.set_ylim(min_val - y_range * 0.05, max_height + y_range * 0.15)
        
        # Add grid and styling
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3, axis='y')
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

    available_columns = eval_df.columns

    if "turns" in available_columns:
        plot_turns_with_std(eval_df, output_dir_path)
    else:
        print("Warning: 'turns' column not found. Skipping turns plot.")

    if "clockSeconds" in available_columns:
        plot_clock_seconds_with_std(eval_df, output_dir_path)
    else:
        print("Warning: 'clockSeconds' column not found. Skipping clock seconds plot.")

    plot_decision_success_with_std(eval_df, output_dir_path)

    if not stats_df.empty:
        plot_score_distributions_with_std(stats_df, output_dir_path)
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
