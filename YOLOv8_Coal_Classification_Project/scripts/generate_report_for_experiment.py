#!/usr/bin/env python3
"""
Generate a printable PDF report for a specific experiment using the saved metrics and visualizations.
Usage: python scripts/generate_report_for_experiment.py --experiment coal_yolov8_m_balanced2
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import yaml
from datetime import datetime


def humanize(name):
    return name.replace('_', ' ').title()


def create_pdf_report(experiment_name: str):
    root = Path('.')
    reports_dir = root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    metrics_csv = root / 'coal_classification_metrics.csv'
    summary_csv = root / 'coal_classification_metrics_summary.csv'

    # Prefer per-experiment files if available
    exp_metrics_csv = reports_dir / f"{experiment_name}_metrics.csv"
    exp_summary_csv = reports_dir / f"{experiment_name}_metrics_summary.csv"

    if exp_metrics_csv.exists():
        metrics_csv = exp_metrics_csv
    if exp_summary_csv.exists():
        summary_csv = exp_summary_csv

    if not metrics_csv.exists() or not summary_csv.exists():
        print('Error: metrics CSV or summary CSV not found. Expected at:', metrics_csv, summary_csv)
        return

    df_metrics = pd.read_csv(metrics_csv)
    df_summary = pd.read_csv(summary_csv)
    # format numeric columns for display
    for col in ['Precision', 'Recall', 'F1_Score', 'AP_Score']:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].map(lambda x: float(x))
            df_metrics[col] = df_metrics[col].map(lambda x: f"{x:.3f}")

    # Confusion matrix image (from advanced_metrics)
    cm_image_path = root / 'confusion_matrix_advanced.png'
    if not cm_image_path.exists():
        # try in reports dir
        cm_image_path = reports_dir / 'confusion_matrix_advanced.png'

    # Output PDF path
    out_pdf = reports_dir / f"{experiment_name}_report.pdf"

    with PdfPages(out_pdf) as pdf:
        # --- Title / Cover Page ---
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
        ax.axis('off')
        title = 'YOLOv8 Coal Classification — Comprehensive Report'
        ax.text(0.5, 0.78, title, ha='center', va='center', fontsize=26, weight='bold')
        ax.text(0.5, 0.72, f'Experiment: {experiment_name}', ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.68, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ha='center', va='center', fontsize=10)

        # Short summary box
        summary_items = []
        for _, row in df_summary.iterrows():
            summary_items.append(f"{row['Metric']}: {row['Value']}")
        summary_text = '\n'.join(summary_items)
        ax.text(0.08, 0.42, summary_text, fontsize=10, family='monospace', bbox=dict(facecolor='#f0f4f7', edgecolor='#c9d6df', boxstyle='round'))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # --- Experiment metadata page (args.yaml) ---
        args_yaml = root / 'coal_classification_runs' / experiment_name / 'args.yaml'
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
        ax.axis('off')
        ax.set_title('Experiment Configuration', fontsize=16, weight='bold')
        if args_yaml.exists():
            try:
                with open(args_yaml, 'r') as f:
                    args = yaml.safe_load(f)
                # render args as two-column table text
                txt = ''
                for k, v in sorted(args.items()):
                    txt += f"{k}: {v}\n"
                ax.text(0.02, 0.98, txt, va='top', fontsize=10, family='monospace')
            except Exception as e:
                ax.text(0.02, 0.98, f'Failed to load args.yaml: {e}', va='top', fontsize=10)
        else:
            ax.text(0.02, 0.98, 'No args.yaml found for experiment.', va='top', fontsize=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Per-class metrics bar chart (styled)
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
        classes = df_metrics['Class'].tolist()
        prec = df_metrics['Precision'].astype(float).values
        rec = df_metrics['Recall'].astype(float).values
        f1 = df_metrics['F1_Score'].astype(float).values

        x = np.arange(len(classes))
        width = 0.24
        ax.bar(x - width, prec, width, label='Precision', color='#4E79A7')
        ax.bar(x, rec, width, label='Recall', color='#F28E2B')
        ax.bar(x + width, f1, width, label='F1', color='#59A14F')
        ax.set_xticks(x)
        # Wrap/shorten long class names for display
        display_names = [c if len(c) < 25 else (c[:22] + '...') for c in classes]
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title('Per-class Metrics', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.25)
        # annotate bars
        def annotate_bars(bars, vals):
            for bar, v in zip(bars, vals):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        bars1 = ax.bar(x - width, prec, width)
        bars2 = ax.bar(x, rec, width)
        bars3 = ax.bar(x + width, f1, width)
        annotate_bars(bars1, prec)
        annotate_bars(bars2, rec)
        annotate_bars(bars3, f1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Confusion matrix page (embed image if present)
        if cm_image_path.exists():
            try:
                img = Image.open(cm_image_path)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                ax.imshow(img)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print('Warning: failed to embed confusion matrix image:', e)

        # Training curves (loss / accuracy) if training logs are available
        results_csv = root / 'coal_classification_runs' / experiment_name / 'results.csv'
        if results_csv.exists():
            try:
                df_res = pd.read_csv(results_csv)
                # Normalize column names (they contain slashes)
                # Expected columns: epoch, train/loss, val/loss, metrics/accuracy_top1
                epochs = df_res['epoch'].values

                train_loss_col = 'train/loss' if 'train/loss' in df_res.columns else None
                val_loss_col = 'val/loss' if 'val/loss' in df_res.columns else None
                acc_col = 'metrics/accuracy_top1' if 'metrics/accuracy_top1' in df_res.columns else None

                # Plot loss curves
                if train_loss_col and val_loss_col:
                    fig, ax = plt.subplots(figsize=(8.5, 6))
                    ax.plot(epochs, df_res[train_loss_col], marker='o', label='Train Loss')
                    ax.plot(epochs, df_res[val_loss_col], marker='o', label='Val Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training and Validation Loss')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Plot accuracy curve
                if acc_col:
                    fig, ax = plt.subplots(figsize=(8.5, 6))
                    ax.plot(epochs, df_res[acc_col], marker='o', color='green', label='Top-1 Accuracy')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_ylim(0, 1.0)
                    ax.set_title('Validation Top-1 Accuracy')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Learning rate visualization if present
                lr_cols = [c for c in df_res.columns if c.startswith('lr/') or c.startswith('lr\\')]
                if lr_cols:
                    fig, ax = plt.subplots(figsize=(8.5, 4))
                    for c in lr_cols:
                        ax.plot(epochs, df_res[c], marker='.', label=c)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Learning Rate')
                    ax.set_title('Learning Rate Schedule')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Summary of best values
                best_idx = None
                if acc_col:
                    best_idx = int(df_res[acc_col].idxmax())
                elif val_loss_col:
                    best_idx = int(df_res[val_loss_col].idxmin())

                if best_idx is not None:
                    best_row = df_res.iloc[best_idx]
                    fig, ax = plt.subplots(figsize=(8.5, 4))
                    ax.axis('off')
                    txt = 'Best training epoch summary:\n\n'
                    for k, v in best_row.items():
                        txt += f"{k}: {v}\n"
                    ax.text(0.01, 0.99, txt, va='top', fontsize=9, family='monospace')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                print('Warning: failed to read/plot training results.csv:', e)

        # Detailed metrics table (split into two columns if wide)
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
        ax.axis('off')
        # prepare table data with better formatting
        table_data = [df_metrics.columns.tolist()] + df_metrics.values.tolist()
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        ax.set_title('Detailed Per-Class Metrics', fontsize=14, weight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add summary table page with nicer layout
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
        ax.axis('off')
        # format summary into two columns
        rows = []
        for _, r in df_summary.iterrows():
            rows.append([r['Metric'], f"{r['Value']}"])
        table = ax.table(cellText=rows, colLabels=['Metric', 'Value'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('Summary Metrics', fontsize=14, weight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f'PDF report generated: {out_pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name (e.g. coal_yolov8_m_balanced2)')
    args = parser.parse_args()
    create_pdf_report(args.experiment)
