import pandas as pd
import matplotlib.pyplot as plt


def plot_dbscan_sensitivity_heatmap(sensitivity_df: pd.DataFrame, output_path: str) -> None:
    if sensitivity_df.empty:
        return

    pivot = sensitivity_df.pivot_table(
        index='min_samples',
        columns='eps_meters',
        values='silhouette_score',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_title('DBSCAN Sensitivity (Silhouette Score)')
    ax.set_xlabel('eps (meters)')
    ax.set_ylabel('min_samples')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label='Silhouette')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_event_timeline(events_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title('Event Candidates Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Photo Count')

    if events_df.empty:
        ax.text(0.5, 0.5, 'No event candidates detected',
                ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    events_sorted = events_df.sort_values('period_start')
    ax.plot(events_sorted['period_start'], events_sorted['photo_count'], marker='o')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
