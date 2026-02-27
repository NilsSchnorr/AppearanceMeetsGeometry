"""
Probability Visualization for Archaeological Masonry Segmentation
==================================================================
Styled to EXACTLY match the paper's figure format (Schnorr & Leimkühler)
See Figures 7-10 in the paper for reference.

Addresses reviewer comment Line 417 about demonstrating the "subjective" 
character of manual interpretation vs. probabilistic machine segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import os

# ============================================================================
# PAPER STYLE CONFIGURATION - Matches your published figures exactly
# ============================================================================

# Your exact color scheme (RGB for matplotlib)
CLASS_COLORS_RGB = [
    [0.0, 0.0, 0.0],      # Background - Black
    [0.0, 0.0, 1.0],      # Ashlar - Blue
    [1.0, 0.0, 0.0],      # Polygonal - Red
    [1.0, 1.0, 0.0]       # Quarry - Yellow
]

CLASS_NAMES = ['Background', 'Ashlar', 'Polygonal', 'Quarry']


def add_corner_label(ax, text, position='top-right', fontsize=11):
    """
    Add a label with semi-transparent white background in the corner.
    Matches the style in Figures 7-10 of your paper exactly.
    """
    positions = {
        'top-right': (0.98, 0.96, 'right', 'top'),
        'top-left': (0.02, 0.96, 'left', 'top'),
        'bottom-right': (0.98, 0.04, 'right', 'bottom'),
        'bottom-left': (0.02, 0.04, 'left', 'bottom')
    }
    
    x, y, ha, va = positions.get(position, positions['top-right'])
    
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight='normal',
            ha=ha, va=va,
            bbox=dict(boxstyle='square,pad=0.3',
                     facecolor='white',
                     edgecolor='none',
                     alpha=0.9))


def create_segmentation_colormap():
    """Create colormap matching paper's color scheme exactly."""
    return ListedColormap(CLASS_COLORS_RGB)


def print_sample_point_data(class_scores, sample_points, image_name="image"):
    """
    Print simple text output for sample points.
    
    Output format:
    Image: [name]
    Point 1: (x, y) | Background: X.X% | Ashlar: X.X% | Polygonal: X.X% | Quarry: X.X% | → Predicted
    """
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    
    print("\n" + "=" * 70)
    print("SAMPLE POINT PROBABILITY DATA")
    print("=" * 70)
    print(f"Image: {image_name}")
    print("-" * 70)
    for i, (x, y) in enumerate(sample_points):
        probs = class_scores[y, x, :n_classes]
        predicted = CLASS_NAMES[np.argmax(probs)]
        prob_strs = [f"{CLASS_NAMES[j]}: {probs[j]*100:.1f}%" for j in range(n_classes)]
        print(f"Point {i+1}: ({x}, {y}) | {' | '.join(prob_strs)} | → {predicted}")
    print("=" * 70 + "\n")
    
    return [(x, y, class_scores[y, x, :n_classes].tolist()) for x, y in sample_points]


# ============================================================================
# MAIN FIGURE FOR PAPER - 2x2 Grid Style (like Figures 7-10)
# ============================================================================

def create_probability_comparison_2x2(class_scores, segmentation, rgb_image,
                                       sample_point=None, save_path=None,
                                       image_name="image"):
    """
    Create a 2x2 comparison figure matching your paper's style (Figures 7-10).
    
    Layout:
        Orthomosaic      |  Segmentation
        Confidence Map   |  Probability Distribution
    
    Also prints simple text output for the sample point.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    h, w = segmentation.shape
    if sample_point is None:
        sample_point = (w // 2, h // 2)
    
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    
    # ========== SIMPLE TEXT OUTPUT ==========
    x, y = sample_point
    probs = class_scores[y, x, :n_classes]
    predicted = CLASS_NAMES[np.argmax(probs)]
    prob_strs = [f"{CLASS_NAMES[j]}: {probs[j]*100:.1f}%" for j in range(n_classes)]
    
    print("\n" + "=" * 70)
    print("SAMPLE POINT PROBABILITY DATA")
    print("=" * 70)
    print(f"Image: {image_name}")
    print("-" * 70)
    print(f"Point 1: ({x}, {y}) | {' | '.join(prob_strs)} | → {predicted}")
    print("=" * 70 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    # ========== TOP LEFT: Orthomosaic ==========
    ax1 = axes[0, 0]
    rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(rgb_display)
    ax1.axis('off')
    add_corner_label(ax1, 'Orthomosaic')
    
    # Mark sample point
    x, y = sample_point
    circle = plt.Circle((x, y), radius=max(h, w)//40, 
                        fill=True, facecolor='white', edgecolor='black', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x, y, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ========== TOP RIGHT: Segmentation ==========
    ax2 = axes[0, 1]
    cmap = create_segmentation_colormap()
    ax2.imshow(segmentation, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax2.axis('off')
    add_corner_label(ax2, 'Full Model')
    
    # Mark same point
    circle2 = plt.Circle((x, y), radius=max(h, w)//40,
                         fill=True, facecolor='white', edgecolor='black', linewidth=2)
    ax2.add_patch(circle2)
    ax2.text(x, y, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ========== BOTTOM LEFT: Confidence Map ==========
    ax3 = axes[1, 0]
    max_probs = np.max(class_scores, axis=2)
    im = ax3.imshow(max_probs, cmap='RdYlGn', vmin=0.25, vmax=1.0)
    ax3.axis('off')
    add_corner_label(ax3, 'Confidence')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=9)
    
    # ========== BOTTOM RIGHT: Probability Bar Chart ==========
    ax4 = axes[1, 1]
    probs = class_scores[y, x, :n_classes]
    predicted_class = np.argmax(probs)
    
    bar_colors = ['black', 'blue', 'red', 'gold'][:n_classes]
    bars = ax4.bar(range(n_classes), probs * 100, color=bar_colors, 
                   edgecolor='black', linewidth=0.5)
    
    # Highlight winner
    bars[predicted_class].set_edgecolor('lime')
    bars[predicted_class].set_linewidth(2.5)
    
    ax4.set_ylim(0, 105)
    ax4.set_xticks(range(n_classes))
    ax4.set_xticklabels(['BG', 'Ashlar', 'Polygonal', 'Quarry'][:n_classes], fontsize=10)
    ax4.set_ylabel('Probability (%)', fontsize=10)
    
    # Add percentage labels
    for bar, prob in zip(bars, probs):
        if prob > 0.03:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{prob*100:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    pred_name = CLASS_NAMES[predicted_class]
    ax4.set_title(f'Point 1: {pred_name} ({probs[predicted_class]*100:.0f}%)', 
                  fontsize=11, pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    return fig


def create_multi_point_analysis(class_scores, segmentation, rgb_image,
                                 sample_points=None, save_path=None,
                                 image_name="image"):
    """
    Create figure with multiple sample points showing probability distributions.
    
    Layout:
        Row 1: Orthomosaic | Segmentation | Confidence
        Row 2: Bar charts for each sample point
    
    Also prints simple text output for each sample point.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    
    h, w = segmentation.shape
    if sample_points is None:
        sample_points = [
            (w // 4, h // 3),
            (w // 2, h // 2),
            (3 * w // 4, 2 * h // 3)
        ]
    
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    
    # ========== SIMPLE TEXT OUTPUT ==========
    print("\n" + "=" * 70)
    print("SAMPLE POINT PROBABILITY DATA")
    print("=" * 70)
    print(f"Image: {image_name}")
    print("-" * 70)
    for i, (x, y) in enumerate(sample_points):
        probs = class_scores[y, x, :n_classes]
        predicted = CLASS_NAMES[np.argmax(probs)]
        prob_strs = [f"{CLASS_NAMES[j]}: {probs[j]*100:.1f}%" for j in range(n_classes)]
        print(f"Point {i+1}: ({x}, {y}) | {' | '.join(prob_strs)} | → {predicted}")
    print("=" * 70 + "\n")
    
    n_points = len(sample_points)
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    
    fig = plt.figure(figsize=(4 * max(3, n_points), 8), facecolor='white')
    
    # ========== TOP ROW: Images ==========
    
    # Orthomosaic
    ax1 = fig.add_subplot(2, 3, 1)
    rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(rgb_display)
    ax1.axis('off')
    add_corner_label(ax1, 'Orthomosaic')
    
    # Mark all sample points
    for i, (x, y) in enumerate(sample_points):
        circle = plt.Circle((x, y), radius=max(h, w)//50,
                           fill=True, facecolor='white', edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, str(i+1), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Segmentation
    ax2 = fig.add_subplot(2, 3, 2)
    cmap = create_segmentation_colormap()
    ax2.imshow(segmentation, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax2.axis('off')
    add_corner_label(ax2, 'Full Model')
    
    for i, (x, y) in enumerate(sample_points):
        circle = plt.Circle((x, y), radius=max(h, w)//50,
                           fill=True, facecolor='white', edgecolor='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, y, str(i+1), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Confidence map
    ax3 = fig.add_subplot(2, 3, 3)
    max_probs = np.max(class_scores, axis=2)
    im = ax3.imshow(max_probs, cmap='RdYlGn', vmin=0.25, vmax=1.0)
    ax3.axis('off')
    add_corner_label(ax3, 'Confidence')
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    # ========== BOTTOM ROW: Probability distributions ==========
    
    bar_colors = ['black', 'blue', 'red', 'gold'][:n_classes]
    
    for i, (x, y) in enumerate(sample_points):
        ax = fig.add_subplot(2, n_points, n_points + i + 1)
        
        probs = class_scores[y, x, :n_classes]
        predicted_class = np.argmax(probs)
        
        bars = ax.bar(range(n_classes), probs * 100, color=bar_colors,
                     edgecolor='black', linewidth=0.5)
        
        bars[predicted_class].set_edgecolor('lime')
        bars[predicted_class].set_linewidth(2.5)
        
        ax.set_ylim(0, 105)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(['BG', 'Ash', 'Poly', 'Qua'][:n_classes], fontsize=9)
        ax.set_ylabel('Probability (%)', fontsize=9)
        
        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{prob*100:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        pred_name = CLASS_NAMES[predicted_class]
        ax.set_title(f'Point {i+1}: {pred_name}', fontsize=10, pad=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    return fig


def create_confidence_comparison_2x2(class_scores, segmentation, save_path=None):
    """
    Create 2x2 confidence/uncertainty comparison in paper style.
    
    Layout:
        Segmentation     |  Manual (placeholder)
        Confidence Map   |  Uncertainty Map
    """
    plt.rcParams['font.family'] = 'sans-serif'
    
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    max_probs = np.max(class_scores, axis=2)
    
    # Entropy
    eps = 1e-10
    entropy = -np.sum(class_scores * np.log(class_scores + eps), axis=2)
    normalized_entropy = entropy / np.log(n_classes)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    # Top left: Segmentation
    ax1 = axes[0, 0]
    cmap = create_segmentation_colormap()
    ax1.imshow(segmentation, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax1.axis('off')
    add_corner_label(ax1, 'Full Model')
    
    # Top right: Per-class dominant probability
    ax2 = axes[0, 1]
    # Show the probability of the assigned class
    assigned_probs = np.zeros_like(max_probs)
    for i in range(n_classes):
        mask = segmentation == i
        assigned_probs[mask] = class_scores[mask, i]
    im2 = ax2.imshow(assigned_probs, cmap='viridis', vmin=0, vmax=1)
    ax2.axis('off')
    add_corner_label(ax2, 'Assigned Class Prob.')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.85)
    cbar2.ax.tick_params(labelsize=9)
    
    # Bottom left: Confidence
    ax3 = axes[1, 0]
    im3 = ax3.imshow(max_probs, cmap='RdYlGn', vmin=0.25, vmax=1.0)
    ax3.axis('off')
    add_corner_label(ax3, 'Confidence')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, shrink=0.85)
    cbar3.ax.tick_params(labelsize=9)
    
    # Bottom right: Uncertainty
    ax4 = axes[1, 1]
    im4 = ax4.imshow(normalized_entropy, cmap='hot', vmin=0, vmax=1)
    ax4.axis('off')
    add_corner_label(ax4, 'Uncertainty')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, shrink=0.85)
    cbar4.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    return fig, max_probs, normalized_entropy


def create_per_class_probability_maps(class_scores, save_path=None):
    """
    Create per-class probability heatmaps in paper style.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    
    fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4), facecolor='white')
    if n_classes == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, CLASS_NAMES[:n_classes])):
        im = ax.imshow(class_scores[:, :, i], cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        add_corner_label(ax, name)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def create_single_pixel_bar_chart(class_scores, x, y, save_path=None):
    """
    Create a single bar chart for one pixel's probability distribution.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    probs = class_scores[y, x, :n_classes]
    predicted_class = np.argmax(probs)
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    
    bar_colors = ['black', 'blue', 'red', 'gold'][:n_classes]
    bars = ax.bar(CLASS_NAMES[:n_classes], probs * 100, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    
    bars[predicted_class].set_edgecolor('lime')
    bars[predicted_class].set_linewidth(3)
    
    ax.set_ylim(0, 105)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_xlabel('Masonry Class', fontsize=11)
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    pred_name = CLASS_NAMES[predicted_class]
    ax.set_title(f'Pixel ({x}, {y}) — Predicted: {pred_name}', fontsize=12, pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return {name: f"{prob*100:.1f}%" for name, prob in zip(CLASS_NAMES[:n_classes], probs)}


def print_probability_statistics(class_scores, segmentation):
    """
    Print statistics about probability distributions for use in paper text.
    """
    max_probs = np.max(class_scores, axis=2)
    non_bg_mask = segmentation > 0
    n_classes = min(class_scores.shape[2], len(CLASS_NAMES))
    
    print("=" * 70)
    print("PROBABILITY DISTRIBUTION STATISTICS FOR PAPER")
    print("=" * 70)
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  COMPARISON: Manual Classification vs. ML Segmentation          │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  Manual:  Each region → single categorical label (100% implied) │")
    print("│  ML:      Each pixel  → probability distribution over classes   │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    
    print("Overall Statistics (all pixels):")
    print(f"  Mean confidence:   {max_probs.mean()*100:.1f}%")
    print(f"  Median confidence: {np.median(max_probs)*100:.1f}%")
    print(f"  Std deviation:     {max_probs.std()*100:.1f}%")
    print()
    
    if non_bg_mask.sum() > 0:
        stone_probs = max_probs[non_bg_mask]
        print("Stone Pixels Only (excluding background):")
        print(f"  Pixel count:       {non_bg_mask.sum():,}")
        print(f"  Mean confidence:   {stone_probs.mean()*100:.1f}%")
        print(f"  Median confidence: {np.median(stone_probs)*100:.1f}%")
        print(f"  Min confidence:    {stone_probs.min()*100:.1f}%")
        print(f"  Max confidence:    {stone_probs.max()*100:.1f}%")
        print()
    
    print("Confidence Distribution (all pixels):")
    for thresh in [0.50, 0.70, 0.80, 0.90, 0.95]:
        pct = (max_probs >= thresh).sum() / max_probs.size * 100
        print(f"  ≥{thresh*100:3.0f}% confidence: {pct:5.1f}% of pixels")
    print()
    
    print("Per-Class Analysis:")
    for i, name in enumerate(CLASS_NAMES[:n_classes]):
        class_mask = segmentation == i
        if class_mask.sum() > 0:
            class_probs = class_scores[class_mask, i]
            print(f"\n  {name}:")
            print(f"    Pixel count:      {class_mask.sum():,}")
            print(f"    Mean probability: {class_probs.mean()*100:.1f}%")
            print(f"    Std deviation:    {class_probs.std()*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("SUGGESTED TEXT FOR PAPER (addresses reviewer Line 417):")
    print("=" * 70)
    
    stone_mean = stone_probs.mean()*100 if non_bg_mask.sum() > 0 else max_probs.mean()*100
    high_conf_pct = (max_probs >= 0.90).sum() / max_probs.size * 100
    
    print(f"""
The probabilistic nature of deep learning segmentation offers fundamental 
transparency regarding classification certainty. While manual interpretation 
assigns categorical labels (effectively 100% confidence), our neural network 
outputs probability distributions across all classes for every pixel.

Across our test data, mean prediction confidence was {stone_mean:.1f}%, with 
only {high_conf_pct:.1f}% of pixels exceeding 90% confidence. This quantification 
of uncertainty—impossible with traditional manual methods—enables more nuanced 
archaeological interpretation and identifies regions where expert review may 
be beneficial.
""")


# ============================================================================
# CONVENIENCE FUNCTION - Run all visualizations
# ============================================================================

def run_all_probability_visualizations(class_scores, segmentation, rgb_image,
                                        output_dir, image_name,
                                        sample_points=None):
    """
    Run all probability visualizations and save to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = segmentation.shape
    if sample_points is None:
        sample_points = [
            (w // 4, h // 3),
            (w // 2, h // 2),
            (3 * w // 4, 2 * h // 3)
        ]
    
    print("=" * 60)
    print("GENERATING PROBABILITY VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Main 2x2 comparison
    print("\n1. Creating 2x2 probability comparison...")
    create_probability_comparison_2x2(
        class_scores, segmentation, rgb_image,
        sample_point=sample_points[0],
        save_path=os.path.join(output_dir, f"{image_name}_prob_comparison_2x2.png"),
        image_name=image_name
    )
    
    # 2. Multi-point analysis
    print("\n2. Creating multi-point analysis...")
    create_multi_point_analysis(
        class_scores, segmentation, rgb_image,
        sample_points=sample_points,
        save_path=os.path.join(output_dir, f"{image_name}_multi_point_analysis.png"),
        image_name=image_name
    )
    
    # 3. Confidence comparison
    print("\n3. Creating confidence/uncertainty maps...")
    create_confidence_comparison_2x2(
        class_scores, segmentation,
        save_path=os.path.join(output_dir, f"{image_name}_confidence_uncertainty.png")
    )
    
    # 4. Per-class probability maps
    print("\n4. Creating per-class probability maps...")
    create_per_class_probability_maps(
        class_scores,
        save_path=os.path.join(output_dir, f"{image_name}_per_class_probabilities.png")
    )
    
    # 5. Statistics
    print("\n5. Generating statistics for paper text...")
    print_probability_statistics(class_scores, segmentation)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print(f"Files saved to: {output_dir}")
    print("=" * 60)
