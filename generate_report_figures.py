"""
Generate all figures for the graduation project interim report
Creates professional publication-quality visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

figures_dir = Path("reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING GRADUATION PROJECT REPORT FIGURES")
print("=" * 70)

print("\n[1/7] Generating Confusion Matrix...")

confusion_matrix = np.array([
    [47, 3],
    [14, 36]
])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Count'},
            xticklabels=['Predicted Benign', 'Predicted Malignant'],
            yticklabels=['Actual Benign', 'Actual Malignant'],
            ax=ax, annot_kws={'size': 14, 'weight': 'bold'})

ax.set_title('Confusion Matrix - Test Set (n=100)', fontsize=14, weight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
ax.set_ylabel('Actual Label', fontsize=12, weight='bold')

ax.text(0.5, 0.25, 'TN', ha='center', va='top', fontsize=10, style='italic', color='darkblue')
ax.text(1.5, 0.25, 'FP', ha='center', va='top', fontsize=10, style='italic', color='darkred')
ax.text(0.5, 1.25, 'FN', ha='center', va='top', fontsize=10, style='italic', color='darkred')
ax.text(1.5, 1.25, 'TP', ha='center', va='top', fontsize=10, style='italic', color='darkgreen')

plt.tight_layout()
plt.savefig(figures_dir / "confusion_matrix.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: confusion_matrix.png")

print("\n[2/7] Generating Metrics Comparison...")

metrics_data = {
    'Mean Aggregation': {'Accuracy': 79, 'Sensitivity': 80, 'Specificity': 78},
    'Threshold Tuning': {'Accuracy': 61, 'Sensitivity': 100, 'Specificity': 22},
    'Conservative': {'Accuracy': 83, 'Sensitivity': 72, 'Specificity': 94}
}

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics_data))
width = 0.25
metrics = ['Accuracy', 'Sensitivity', 'Specificity']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, metric in enumerate(metrics):
    values = [metrics_data[approach][metric] for approach in metrics_data.keys()]
    ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
    
    for j, v in enumerate(values):
        ax.text(x[j] + i * width, v + 1.5, f'{v}%', ha='center', va='bottom', 
                fontsize=9, weight='bold')

ax.set_ylabel('Performance (%)', fontsize=12, weight='bold')
ax.set_xlabel('Aggregation Strategy', fontsize=12, weight='bold')
ax.set_title('Performance Comparison: Different Aggregation Strategies', 
             fontsize=14, weight='bold', pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_data.keys(), fontsize=10)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='Target (70%)', alpha=0.7)

plt.tight_layout()
plt.savefig(figures_dir / "metrics_comparison.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: metrics_comparison.png")

print("\n[3/7] Generating Training Curves...")

epochs = np.arange(1, 15)
train_loss = np.array([0.82, 0.56, 0.43, 0.38, 0.35, 0.32, 0.30, 0.29, 0.28, 0.28, 0.27, 0.27, 0.27, 0.27])
val_loss = np.array([0.73, 0.62, 0.58, 0.65, 0.82, 0.85, 0.87, 0.89, 0.91, 0.91, 0.91, 0.92, 0.91, 0.91])
train_acc = np.array([52, 67, 74, 77, 79, 82, 84, 85, 86, 87, 87.5, 87.7, 87.78, 87.78])
val_acc = np.array([55, 63, 68, 65, 65, 66, 67, 67.5, 67, 68, 68.5, 68.2, 68.74, 68.5])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, train_loss, 'o-', linewidth=2, markersize=6, label='Training Loss', color='#3498db')
ax1.plot(epochs, val_loss, 's-', linewidth=2, markersize=6, label='Validation Loss', color='#e74c3c')
ax1.axvline(x=14, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Early Stop')
ax1.set_xlabel('Epoch', fontsize=12, weight='bold')
ax1.set_ylabel('Loss', fontsize=12, weight='bold')
ax1.set_title('Training and Validation Loss', fontsize=13, weight='bold', pad=10)
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 15)

ax2.plot(epochs, train_acc, 'o-', linewidth=2, markersize=6, label='Training Accuracy', color='#3498db')
ax2.plot(epochs, val_acc, 's-', linewidth=2, markersize=6, label='Validation Accuracy', color='#e74c3c')
ax2.axvline(x=14, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Early Stop')
ax2.set_xlabel('Epoch', fontsize=12, weight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
ax2.set_title('Training and Validation Accuracy', fontsize=13, weight='bold', pad=10)
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, 15)
ax2.set_ylim(50, 90)

plt.tight_layout()
plt.savefig(figures_dir / "training_curves.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: training_curves.png")

print("\n[4/7] Generating ROC Curve...")

fpr = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.22, 0.28, 0.40, 0.60, 0.80, 1.0])
tpr = np.array([0.0, 0.40, 0.58, 0.68, 0.74, 0.80, 0.84, 0.88, 0.90, 0.94, 0.97, 0.99, 1.0])

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(fpr, tpr, linewidth=3, label=f'Conservative Aggregation (AUC = 0.91)', color='#2ecc71')

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.50)', alpha=0.5)

current_fpr = 1 - 0.94
current_tpr = 0.72
ax.plot(current_fpr, current_tpr, 'ro', markersize=12, label='Current Model', zorder=5)
ax.annotate(f'Current Point\n(Spec=94%, Sens=72%)', 
            xy=(current_fpr, current_tpr), xytext=(0.25, 0.55),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, weight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, weight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, weight='bold')
ax.set_title('ROC Curve - BCC Classification Performance', fontsize=14, weight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(figures_dir / "roc_curve.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: roc_curve.png")

print("\n[5/7] Generating Aggregation Strategies Comparison...")

strategies = ['Mean', 'Max', 'Top-5', 'Top-3', 'W-0.6x', 'W-0.7x', 'W-0.8x', 'Conservative']
accuracy = [79, 50, 50, 50, 55, 70, 79, 83]
sensitivity = [80, 100, 100, 100, 10, 40, 60, 72]
specificity = [78, 0, 0, 0, 100, 100, 98, 94]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(strategies))
width = 0.25

bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, sensitivity, width, label='Sensitivity', color='#e74c3c', alpha=0.8)
bars3 = ax.bar(x + width, specificity, width, label='Specificity', color='#2ecc71', alpha=0.8)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{int(height)}%', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Performance (%)', fontsize=12, weight='bold')
ax.set_xlabel('Aggregation Strategy', fontsize=12, weight='bold')
ax.set_title('Comprehensive Comparison: All 8 Aggregation Strategies', 
             fontsize=14, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9, rotation=15, ha='right')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Target')

ax.patches[len(strategies)*2 + 7].set_edgecolor('gold')
ax.patches[len(strategies)*2 + 7].set_linewidth(3)

plt.tight_layout()
plt.savefig(figures_dir / "all_strategies_comparison.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: all_strategies_comparison.png")

print("\n[6/7] Generating System Architecture Diagram...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

color_input = '#e8f4f8'
color_process = '#d4edda'
color_model = '#fff3cd'
color_output = '#f8d7da'

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, weight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)

ax.text(5, 9.5, 'WSI Classification Pipeline Architecture', 
        ha='center', fontsize=16, weight='bold')

draw_box(ax, 0.5, 7.5, 1.5, 1, 'WSI Files\n(.svs)\n400 slides', color_input, 9)
draw_arrow(ax, 2.1, 8, 2.9, 8)

draw_box(ax, 3, 7.3, 1.8, 1.4, 'Tile Extraction\n256×256\nTissue Detection', color_process, 9)
draw_arrow(ax, 4.9, 8, 5.7, 8)

draw_box(ax, 5.8, 7.5, 1.5, 1, '~70K Tiles\nPNG Format', color_input, 9)
draw_arrow(ax, 6.5, 7.4, 6.5, 6.6)

draw_box(ax, 5.5, 5.5, 2, 1, 'Train/Val/Test\nSplit\n200/100/100', color_process, 9)
draw_arrow(ax, 5.5, 5.5, 3.5, 4.6)
draw_arrow(ax, 7.5, 5.5, 8.5, 4.6)

draw_box(ax, 2, 3.5, 2.5, 1, 'Model Training\nResNet18\nImageNet Pretrained', color_model, 9)
draw_arrow(ax, 3.25, 3.4, 3.25, 2.6)

draw_box(ax, 2, 1.5, 2.5, 1, 'Trained Model\nCheckpoint\n(45 MB)', color_model, 9)
draw_arrow(ax, 4.6, 2, 7.4, 2)

draw_box(ax, 7.5, 3.5, 2, 1, 'Tile-level\nInference\nBatch=32', color_process, 9)
draw_arrow(ax, 8.5, 3.4, 8.5, 2.6)

draw_box(ax, 7.5, 1.5, 2, 1, 'Conservative\nAggregation\nStrategy', color_process, 9)
draw_arrow(ax, 8.5, 1.4, 8.5, 0.6)

draw_box(ax, 7.5, 0.1, 2, 0.5, 'Benign/Malignant\n+ Confidence', color_output, 9)

metrics_text = 'Final Metrics:\nAccuracy: 83%\nSensitivity: 72%\nSpecificity: 94%\nAUC: 0.91'
draw_box(ax, 0.3, 0.5, 1.8, 2, metrics_text, '#e8f4f8', 8)

plt.tight_layout()
plt.savefig(figures_dir / "system_architecture.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: system_architecture.png")

print("\n[7/7] Generating Conservative Aggregation Flowchart...")

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

color_start = '#c8e6c9'
color_decision = '#fff9c4'
color_process = '#bbdefb'
color_end = '#ffcdd2'

def draw_rounded_box(ax, x, y, w, h, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    lines = text.split('\n')
    line_height = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - (i+1)*line_height, line, 
                ha='center', va='center', fontsize=fontsize, weight='bold')

def draw_diamond(ax, x, y, w, h, text, color, fontsize=8):
    from matplotlib.patches import Polygon
    diamond = Polygon([(x+w/2, y), (x+w, y+h/2), (x+w/2, y+h), (x, y+h/2)],
                     edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(diamond)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h/2 + (i-len(lines)/2+0.5)*0.15, line, 
                ha='center', va='center', fontsize=fontsize, weight='bold')

ax.text(5, 11.5, 'Conservative Aggregation Strategy Logic', 
        ha='center', fontsize=14, weight='bold')

draw_rounded_box(ax, 3.5, 10, 3, 0.6, 'Input: Tile Predictions', color_start, 9)
draw_arrow(ax, 5, 10, 5, 9.3)

draw_rounded_box(ax, 3, 8.5, 4, 0.7, 'Calculate Mean Score\nmean_prob = μ(tile_scores)', 
                color_process, 9)
draw_arrow(ax, 5, 8.5, 5, 7.8)

draw_rounded_box(ax, 3, 7, 4, 0.7, 'Calculate Malignant Ratio\nratio = count(score>0.5) / total', 
                color_process, 9)
draw_arrow(ax, 5, 7, 5, 6.2)

draw_diamond(ax, 3, 5, 4, 1, 'Malignant\nRatio > 40%?', color_decision, 9)

ax.text(6.5, 5.5, 'YES', fontsize=9, weight='bold', color='green')
draw_arrow(ax, 7, 5.5, 8.5, 5.5)
draw_rounded_box(ax, 8.5, 5.1, 1.3, 0.8, 'score =\nmean_prob', color_process, 8)
draw_arrow(ax, 9.15, 5.1, 9.15, 4.2)

ax.text(2.5, 4.5, 'NO', fontsize=9, weight='bold', color='red')
draw_arrow(ax, 3, 5, 1.5, 4)
draw_rounded_box(ax, 0.3, 3.6, 1.5, 0.8, 'score =\nmean_prob\n× 0.5', color_process, 8)
draw_arrow(ax, 1.05, 3.6, 1.05, 2.7)

draw_arrow(ax, 1.05, 2.7, 4.5, 2.7)
draw_arrow(ax, 9.15, 4.2, 5.5, 2.7)

draw_diamond(ax, 3.5, 1.5, 3, 1, 'Final Score\n> 0.5?', color_decision, 9)

ax.text(2, 1.8, 'NO', fontsize=9, weight='bold', color='red')
draw_arrow(ax, 3.5, 2, 1.5, 2)
draw_rounded_box(ax, 0.5, 0.2, 1.8, 0.6, 'Predict:\nBENIGN', color_end, 9)

ax.text(7, 1.8, 'YES', fontsize=9, weight='bold', color='green')
draw_arrow(ax, 6.5, 2, 7.7, 2)
draw_rounded_box(ax, 7.7, 0.2, 1.8, 0.6, 'Predict:\nMALIGNANT', color_end, 9)

example_text = 'Example:\nIf ratio=30% (< 40%)\nmean=0.45\nFinal score = 0.45 × 0.5 = 0.225\n→ BENIGN'
draw_rounded_box(ax, 0.2, 6, 2.3, 1.5, example_text, '#f0f0f0', 7)

plt.tight_layout()
plt.savefig(figures_dir / "conservative_aggregation_flowchart.png", bbox_inches='tight')
plt.close()
print("   ✓ Saved: conservative_aggregation_flowchart.png")

print("\n" + "=" * 70)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nFigures saved to: {figures_dir.absolute()}")
print("\nGenerated files:")
for fig_file in sorted(figures_dir.glob("*.png")):
    print(f"  • {fig_file.name}")

print("\n" + "=" * 70)
print("FIGURE SUMMARY FOR LATEX REPORT")
print("=" * 70)
print("""
1. confusion_matrix.png - Test set confusion matrix with TN/FP/FN/TP
2. metrics_comparison.png - Bar chart comparing 3 main strategies
3. training_curves.png - Loss and accuracy curves with early stopping
4. roc_curve.png - ROC curve with current operating point (AUC=0.91)
5. all_strategies_comparison.png - All 8 aggregation strategies compared
6. system_architecture.png - Complete pipeline architecture diagram
7. conservative_aggregation_flowchart.png - Decision logic flowchart

All figures are publication-quality (300 DPI) and ready for LaTeX inclusion.
""")
