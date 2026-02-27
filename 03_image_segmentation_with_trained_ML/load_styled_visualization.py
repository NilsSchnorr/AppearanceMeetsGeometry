# ============================================================================
# PAPER-STYLED PROBABILITY VISUALIZATION
# ============================================================================
# 
# This cell imports the styled visualization functions that match your paper's
# figure format (corner labels with white backgrounds, proper colors, etc.)
#
# Run this cell AFTER running the segmentation with:
#   full_segmentation, class_scores, rgb, normals, n_classes, image_name = run_segmentation_with_probabilities()
# ============================================================================

import sys
sys.path.insert(0, '.')  # Ensure current directory is in path

from probability_visualization_styled import (
    create_probability_figure_for_paper,
    create_probability_heatmaps_styled,
    create_confidence_uncertainty_figure,
    create_single_pixel_analysis,
    print_probability_statistics
)

print("Styled visualization functions loaded successfully!")
print("\nAvailable functions:")
print("  - create_probability_figure_for_paper()  → Main figure for paper")
print("  - create_probability_heatmaps_styled()   → Per-class probability maps")
print("  - create_confidence_uncertainty_figure() → Confidence & entropy maps")
print("  - create_single_pixel_analysis()         → Single pixel bar chart")
print("  - print_probability_statistics()         → Stats for paper text")
