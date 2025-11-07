import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

def main():
    # Get the path to span_ratios.jsonl (one level up from scripts directory)
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / 'artifacts' / 'span_ratios.jsonl'
    df = pd.read_json(data_file, lines=True)
    df = df.groupby(['dataset', 'format', 'loss_masking']).mean().reset_index()
    df['span_ratio'] = df['span_labels'] / df['span_loss_mask']
    df['start_ratio'] = df['start_labels'] / df['start_loss_mask']
    df['end_ratio'] = df['end_labels'] / df['end_loss_mask']
    labels = df['dataset'] + ' | ' + df['format'] + ' | ' + df['loss_masking']

    # Create color mapping based on dataset (using standard muted colors)
    unique_datasets = df['dataset'].unique()
    # Use tab10 colormap which has standard, less bright colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_datasets)))
    dataset_to_color = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}
    df['color'] = df['dataset'].map(dataset_to_color)

    # Create marker mapping based on format
    format_to_marker = {'text': 'o', 'tokens': 's'}
    df['marker'] = df['format'].map(format_to_marker)
    
    # Create edge style mapping based on loss_masking (filled vs unfilled)
    # 'none' = filled, 'subwords' = unfilled (hollow)
    df['facecolor'] = df.apply(lambda row: row['color'] if row['loss_masking'] == 'none' else 'white', axis=1)
    df['edgewidth'] = df.apply(lambda row: 0.5 if row['loss_masking'] == 'none' else 1.5, axis=1)

    plt.figure(figsize=(10, 6))
    
    # Track which datasets we've already added to legend
    seen_datasets = set()
    
    # Add small jitter to separate overlapping points (text vs tokens with same ratios)
    np.random.seed(42)  # For reproducibility
    jitter_factor = 0.02  # 2% jitter
    df['start_ratio_jittered'] = df['start_ratio'] * (1 + np.random.uniform(-jitter_factor, jitter_factor, len(df)))
    df['span_ratio_jittered'] = df['span_ratio'] * (1 + np.random.uniform(-jitter_factor, jitter_factor, len(df)))
    
    # Plot each point with its color, marker, and fill style
    for idx, row in df.iterrows():
        label = row['dataset'] if row['dataset'] not in seen_datasets else ""
        if row['dataset'] not in seen_datasets:
            seen_datasets.add(row['dataset'])
        
        plt.scatter(
            row['start_ratio_jittered'], 
            row['span_ratio_jittered'],
            c=[row['facecolor']],
            marker=row['marker'],
            s=100,
            label=label,
            edgecolors=row['color'],
            linewidths=row['edgewidth'],
            alpha=0.7  # Add transparency so overlapping points are visible
        )
    
    plt.xscale('log')
    plt.xlabel('Start Ratio')
    plt.yscale('log')
    plt.ylabel('Span Ratio')
    
    # Get the current legend handles and labels from the scatter plots
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    
    # Create marker legend entries for format
    format_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=7, markeredgecolor='black', markeredgewidth=0.5, linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=7, markeredgecolor='black', markeredgewidth=0.5, linestyle='None')
    ]
    format_labels = ['text', 'tokens']
    
    # Create legend entries for loss_masking (filled vs unfilled)
    loss_masking_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=7, markeredgecolor='black', markeredgewidth=0.5, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markersize=7, markeredgecolor='gray', markeredgewidth=1.5, linestyle='None')
    ]
    loss_masking_labels = ['none', 'subwords']
    
    # Combine all legends
    all_handles = handles + format_handles + loss_masking_handles
    all_labels = legend_labels + format_labels + loss_masking_labels
    
    # Create the legend with dataset colors, format markers, and loss_masking styles
    # Position it outside the plot area to the right
    plt.legend(all_handles, all_labels, 
               title='Dataset | Format | Loss Masking', 
               loc='center left', 
               bbox_to_anchor=(1.02, 0.5),
               ncol=1, 
               frameon=True,
               fontsize=9,
               title_fontsize=10,
               handlelength=1.5,
               columnspacing=0.5)
    plt.grid(True, alpha=0.3)
    # Adjust layout to make room for the legend on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save to artifacts directory
    output_dir = script_dir.parent / 'artifacts'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'span_ratio.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()