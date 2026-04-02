import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot(csv_file_path, output_image_path='token_frequency_analysis_grouped_avg_with_stats.png'):
    """
    Analyze CSV file and draw a bar chart where each bar represents the average 'weight'
    of a group of 1000 consecutive tokens from the CSV file. Adds a text box on the right
    showing statistics for token frequencies 0-9.
    
    Args:
        csv_file_path (str): Path to the CSV file.
        output_image_path (str): Path for the output image, default is 'token_frequency_analysis_grouped_avg_with_stats.png'.
    """
    # 1. Read CSV file
    print(f"Reading file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    total_codes = len(df)

    # Ensure 'weight' column exists
    if 'weight' not in df.columns:
        raise ValueError("CSV file must contain a 'weight' column.")

    # 2. Count tokens with frequency 0
    zero_count_codes = df[df['count'] == 0]
    num_zero_count = len(zero_count_codes)
    percentage_zero = (num_zero_count / total_codes) * 100

    # 3. Count tokens with frequency 1 to 9
    counts_of_interest = list(range(0, 10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    stats_dict = {}
    for target_count in counts_of_interest:
        specific_count_codes = df[df['count'] == target_count]
        num_specific_count = len(specific_count_codes)
        percentage_specific = (num_specific_count / total_codes) * 100
        stats_dict[target_count] = {
            'num': num_specific_count,
            'pct': percentage_specific
        }

    # 4. Print all statistics
    print(f"Total number of codes: {total_codes}")
    for count_val in counts_of_interest:
        num = stats_dict[count_val]['num']
        pct = stats_dict[count_val]['pct']
        print(f"Number of codes with frequency {count_val}: {num}, Percentage: {pct:.2f}%")

    # 5. Group every 1000 'weight' values and calculate their average
    weights = df['weight'].values
    group_size = 1000
    
    # Calculate number of complete groups
    num_complete_groups = len(weights) // group_size
    # Trim the array to fit complete groups only
    trimmed_weights = weights[:num_complete_groups * group_size]
    # Reshape the array to have 'group_size' columns
    reshaped_weights = trimmed_weights.reshape(-1, group_size)
    # Calculate the mean along axis 1 (across columns of each row/group)
    avg_weights_per_group = np.mean(reshaped_weights, axis=1)

    # If there are leftover items that don't fill a complete group,
    # we calculate the average of the remaining items separately.
    leftover_start_idx = num_complete_groups * group_size
    leftover_weights = weights[leftover_start_idx:]
    avg_leftover_weight = None
    if len(leftover_weights) > 0:
        avg_leftover_weight = np.mean(leftover_weights)
        print(f"Found {len(leftover_weights)} leftover codes. Their average weight is {avg_leftover_weight:.6f}")

    # Prepare text for the statistics box
    stats_text_lines = [f'Total Codes: {total_codes}', 'Frequency Stats (0-9):']
    for count_val in counts_of_interest:
        num = stats_dict[count_val]['num']
        pct = stats_dict[count_val]['pct']
        stats_text_lines.append(f'  Freq {count_val}: {num} ({pct:.2f}%)')
    stats_text = '\n'.join(stats_text_lines)

    # 6. Create the bar chart
    fig, ax = plt.subplots(figsize=(16, 8)) # Wider figure to accommodate the text box

    # X-axis: Group indices (0, 1, 2, ...)
    group_indices = list(range(len(avg_weights_per_group)))
    if avg_leftover_weight is not None:
        # Add an extra index for the leftover group
        group_indices.append(len(avg_weights_per_group))
        # Append the average weight of the leftover group
        avg_weights_for_plot = np.append(avg_weights_per_group, avg_leftover_weight)
    else:
        avg_weights_for_plot = avg_weights_per_group

    ax.bar(group_indices, avg_weights_for_plot, width=0.8, edgecolor='black', linewidth=0.3)
    
    ax.set_title(f'Average Weight per Group of 1000 Tokens\n(Total Groups: {len(avg_weights_per_group)} + 1 leftover if applicable)')
    ax.set_xlabel('Group Index (Each group contains 1000 tokens)')
    ax.set_ylabel('Average Weight of Group')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the statistics text box on the right side of the plot area
    ax.text(0.75, 0.98, stats_text,
             transform=ax.transAxes, # Use axes coordinates (0,0 is bottom left, 1,1 is top right of plot area)
             verticalalignment='top',
             horizontalalignment='left', # Left align the text within the bbox
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             fontsize=10,
             family='monospace') # Monospace font for better alignment of numbers

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # 7. Save image
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Grouped average weight bar chart with stats saved to: {output_image_path}")
    plt.close() # Close the figure to release memory

if __name__ == "__main__":
    csv_file = "token_frequencies_sorted_by_count_desc.csv"
    output_img = "token_frequency_grouped_avg_with_stats.png" # Specify the output image filename
    analyze_and_plot(csv_file, output_img)
