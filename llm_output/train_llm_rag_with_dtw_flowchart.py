"""
DTW-Based RAG Forecasting Flowchart Generator
Creates publication-ready flowchart visualizing the complete forecasting methodology:
Retrieval (DTW) → Augmentation (Prompt) → Generation (LLM) pipeline.

Author: Flaredown Research Team
Date: November 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Wedge, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
import os
from pathlib import Path

def create_box(ax, x, y, width, height, text, color, style='round'):
    """Create a styled box for flowchart"""
    if style == 'cylinder':
        # Database/file symbol - proper 3D cylinder
        ellipse_height = height * 0.15  # Height of the ellipse
        
        # Main body (rectangle without top/bottom edges)
        body = Rectangle((x - width/2, y - height/2), width, height,
                        facecolor=color, edgecolor='none')
        ax.add_patch(body)
        
        # Bottom ellipse - draw back half first (hidden by cylinder body perspective)
        theta_back = np.linspace(0, np.pi, 100)
        ellipse_x_back = x + (width/2) * np.cos(theta_back)
        ellipse_y_back = y - height/2 + (ellipse_height/2) * np.sin(theta_back)
        ax.fill(ellipse_x_back, ellipse_y_back, color=color, edgecolor='none', zorder=1)
        ax.plot(ellipse_x_back, ellipse_y_back, 'k-', linewidth=1.5, zorder=2)
        
        # Left vertical line
        ax.plot([x - width/2, x - width/2], [y - height/2, y + height/2],
                'k-', linewidth=1.5, zorder=3)
        
        # Right vertical line
        ax.plot([x + width/2, x + width/2], [y - height/2, y + height/2],
                'k-', linewidth=1.5, zorder=3)
        
        # Bottom ellipse - draw front half (visible, on top of body)
        theta_front = np.linspace(np.pi, 2*np.pi, 100)
        ellipse_x_front = x + (width/2) * np.cos(theta_front)
        ellipse_y_front = y - height/2 + (ellipse_height/2) * np.sin(theta_front)
        ax.fill(ellipse_x_front, ellipse_y_front, color=color, edgecolor='none', zorder=4)
        ax.plot(ellipse_x_front, ellipse_y_front, 'k-', linewidth=1.5, zorder=4)
        
        # Top ellipse (full, visible)
        top_ellipse = mpatches.Ellipse((x, y + height/2), width, ellipse_height,
                                       facecolor=color, edgecolor='black', 
                                       linewidth=1.5, zorder=4)
        ax.add_patch(top_ellipse)
        
    else:
        bbox = FancyBboxPatch((x - width/2, y - height/2), width, height,
                               boxstyle="round,pad=0.05", 
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(bbox)
    
    ax.text(x, y, text, ha='center', va='center', fontsize=7, wrap=True,
            multialignment='center', fontweight='normal', zorder=5)

def create_arrow(ax, x1, y1, x2, y2, label='', style='solid', color='black', spacing=0.1, linewidth=1.2):
    """Create an arrow between boxes with spacing"""
    # Calculate direction and add spacing
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    
    if length > 0 and length > 2 * spacing:  # Only add spacing if arrow is long enough
        # Normalize and add spacing at both ends
        dx_norm = dx / length
        dy_norm = dy / length
        
        x1_spaced = x1 + dx_norm * spacing
        y1_spaced = y1 + dy_norm * spacing
        x2_spaced = x2 - dx_norm * spacing
        y2_spaced = y2 - dy_norm * spacing
    else:
        # For short arrows, don't add spacing or it disappears
        x1_spaced, y1_spaced = x1, y1
        x2_spaced, y2_spaced = x2, y2
    
    arrow = FancyArrowPatch((x1_spaced, y1_spaced), (x2_spaced, y2_spaced),
                            arrowstyle='->', mutation_scale=12.75,
                            linewidth=linewidth,
                            linestyle='--',
                            color=color, zorder=1)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=6, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

def create_curved_arrow(ax, x1, y1, x2, y2, label='', color='#FF6B6B', linewidth=1.2, spacing=0.3, rad=0.4):
    """Create a curved arrow using ConnectionPatch with quadratic Bezier curve
    
    Args:
        rad: curve radius. Positive values curve counterclockwise, negative clockwise
    """
    from matplotlib.patches import ConnectionPatch
    
    # Calculate direction and add spacing
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    
    if length > 0 and length > 2 * spacing:
        dx_norm = dx / length
        dy_norm = dy / length
        
        x1_spaced = x1 + dx_norm * spacing
        y1_spaced = y1 + dy_norm * spacing
        x2_spaced = x2 - dx_norm * spacing
        y2_spaced = y2 - dy_norm * spacing
    else:
        x1_spaced, y1_spaced = x1, y1
        x2_spaced, y2_spaced = x2, y2
    
    # Create curved arrow using connectionpatch
    arrow = FancyArrowPatch((x1_spaced, y1_spaced), (x2_spaced, y2_spaced),
                            connectionstyle=f"arc3,rad={rad}",
                            arrowstyle='->', mutation_scale=15.3,
                            linewidth=linewidth, linestyle='--', color=color, zorder=2)
    ax.add_patch(arrow)
    
    if label:
        # Calculate midpoint of curve for label placement
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        # Offset label perpendicular to the general direction
        offset_x = -dy / length * 0.4 if length > 0 else 0
        offset_y = dx / length * 0.4 if length > 0 else 0
        ax.text(mid_x + offset_x, mid_y + offset_y, label, fontsize=6, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9),
                ha='center')

def create_dual_curve_arrow(ax, x1, y1, x2, y2, label='', color='#FF6B6B', linewidth=1.2, spacing=0.3, 
                            outer_rad=0.4, inner_rad=-0.5, transition_point=0.5, curve_origin=True):
    """Create two connected arrow segments: outer curve (no tip) connected to inner curve (with tip)
    
    Args:
        outer_rad: curve radius for the outer segment
        inner_rad: curve radius for the inner segment (opposite direction)
        transition_point: where to split (0.0-1.0)
        curve_origin: if True, outer curve at origin with inner curve at tip
                     if False, inner curve at origin with outer curve at tip
    """
    
    # Calculate direction and add spacing
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    
    if length > 0 and length > 2 * spacing:
        dx_norm = dx / length
        dy_norm = dy / length
        
        x1_spaced = x1 + dx_norm * spacing
        y1_spaced = y1 + dy_norm * spacing
        x2_spaced = x2 - dx_norm * spacing
        y2_spaced = y2 - dy_norm * spacing
    else:
        x1_spaced, y1_spaced = x1, y1
        x2_spaced, y2_spaced = x2, y2
    
    # Calculate transition point
    trans_x = x1_spaced + (x2_spaced - x1_spaced) * transition_point
    trans_y = y1_spaced + (y2_spaced - y1_spaced) * transition_point
    
    if curve_origin:
        # Outer curve at origin (no tip), inner curve at end (with tip)
        # First segment: outer curve without arrowhead
        arrow1 = FancyArrowPatch((x1_spaced, y1_spaced), (trans_x, trans_y),
                                 connectionstyle=f"arc3,rad={outer_rad}",
                                 arrowstyle='-', mutation_scale=15.3,
                                 linewidth=linewidth, linestyle='--', color=color, zorder=2)
        ax.add_patch(arrow1)
        
        # Second segment: inner curve with arrowhead
        arrow2 = FancyArrowPatch((trans_x, trans_y), (x2_spaced, y2_spaced),
                                 connectionstyle=f"arc3,rad={inner_rad}",
                                 arrowstyle='->', mutation_scale=15.3,
                                 linewidth=linewidth, linestyle='--', color=color, zorder=2)
        ax.add_patch(arrow2)
    else:
        # Inner curve at origin (no tip), outer curve at end (with tip)
        # First segment: inner curve without arrowhead
        arrow1 = FancyArrowPatch((x1_spaced, y1_spaced), (trans_x, trans_y),
                                 connectionstyle=f"arc3,rad={inner_rad}",
                                 arrowstyle='-', mutation_scale=15.3,
                                 linewidth=linewidth, linestyle='--', color=color, zorder=2)
        ax.add_patch(arrow1)
        
        # Second segment: outer curve with arrowhead
        arrow2 = FancyArrowPatch((trans_x, trans_y), (x2_spaced, y2_spaced),
                                 connectionstyle=f"arc3,rad={outer_rad}",
                                 arrowstyle='->', mutation_scale=15.3,
                                 linewidth=linewidth, linestyle='--', color=color, zorder=2)
        ax.add_patch(arrow2)
    
    if label:
        # Calculate midpoint for label placement
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        # Offset label perpendicular to the general direction
        offset_x = -dy / length * 0.4 if length > 0 else 0
        offset_y = dx / length * 0.4 if length > 0 else 0
        ax.text(mid_x + offset_x, mid_y + offset_y, label, fontsize=6, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9),
                ha='center')

def create_dtw_rag_flowchart():
    """Create flowchart for DTW-based RAG forecasting methodology"""
    
    fig, ax = plt.subplots(figsize=(11, 8.5))  # Letter size in landscape
    ax.set_xlim(-1, 16)  # Extended left margin to accommodate Left Diamond Box
    ax.set_ylim(0, 12)  # Increased upper limit to provide more vertical space
    ax.axis('off')
    
    # Minimize margins to prevent white border on left
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Color scheme matching data pipeline flowchart
    colors = {
        'data': '#C8E6C9',         # Light green for data inputs/outputs
        'retrieval': '#FFF9C4',    # Light yellow for retrieval phase
        'augmentation': '#BBDEFB', # Light blue for augmentation
        'generation': '#C8E6C9',   # Light green for LLM generation
        'process': '#BBDEFB',      # Light blue for processing
        'yellow': '#FFF9C4',       # Light yellow for above-augmentation boxes
        'blue': '#BBDEFB',         # Light blue for augmentation boxes
        'green': '#C8E6C9',        # Light green for below-augmentation boxes
    }
    
    y = 12.5  # Start from top (shifted up for better bottom margin)
    x_center = 8  # Center of diagram
    box_width = 3.0
    box_height = 0.65
    dy = 1.1  # Vertical spacing between boxes (increased by 10%)
    phase_label_offset = 0.45  # Consistent offset for phase labels
    
    # ========================================================================
    # TITLE
    # ========================================================================
    ax.text(x_center, y - 0.3, 'DTW-Based RAG Forecasting Methodology',
            ha='center', va='center', fontsize=13, fontweight='bold',
            family='arial')
    
    y -= 1.5
    
    # ========================================================================
    # PHASE 0: INPUT DATA - REDESIGNED LAYOUT
    # ========================================================================
    
    # Position constants for cleaner layout
    x_left_col = 3.5      # Left column (training data)
    x_center_col = 8      # Center column (main flow)
    x_right_col = 8       # Right column (test data) - aligned with center
    
    # Training data (left side) - narrower and taller for barrel-like appearance
    # Positioned at midpoint between Test Data Pool and Single Test Window
    y_training_db = y - dy / 2  # Midpoint between the two test boxes
    create_box(ax, x_left_col, y_training_db, 1.8, 1.4,
               'Training Database\n20-Day Windows\nAnxiety: n=25,945\nDepression: n=21,034',
               colors['yellow'], 'cylinder')
    
    # Test data pool (right side)
    create_box(ax, x_right_col, y, 2.8, 0.7,
               'Test Data Pool\nDays 1-10 Windows\nn=1,500 per condition',
               colors['yellow'])
    y_test_pool = y
    
    y -= dy
    
    # Sample single test window
    create_box(ax, x_right_col, y, 2.8, 0.65,
               'Single Test Window\n(Days 1-10)',
               colors['yellow'])
    y_single_test = y
    
    # Solid arrow between Test Data Pool and Single Test Window
    create_arrow(ax, x_right_col, y_test_pool - box_height/2, x_right_col, y_single_test + 0.65/2,
                 label='', style='solid', color='black', spacing=0.1, linewidth=1.2)
    
    y -= dy
    
    # ========================================================================
    # PHASE 1: RETRIEVAL (DTW-Based Matching) - REDESIGNED
    # ========================================================================
    
    # Main DTW retrieval box (center)
    create_box(ax, x_center_col, y, box_width, box_height,
               'DTW-Based Similar Patient Matching',
               colors['yellow'])
    y_dtw_matching = y
    
    # Solid arrow between Single Test Window and DTW-Based Similar Patient Matching
    create_arrow(ax, x_center_col, y_single_test - 0.65/2, x_center_col, y_dtw_matching + box_height/2,
                 label='', style='solid', color='black', spacing=0.1, linewidth=1.2)
    
    # Main vertical flow continues here
    y -= dy
    
    # Retrieved results box
    create_box(ax, x_center_col, y, box_width, box_height,
               'Top-25 Similar Patients\n(DTW-ranked by trajectory)',
               colors['yellow'])
    y_retrieved = y
    
    # Solid arrow between Top-25 Similar Patients and Prompt Construction
    # (positioned here before y_augmentation is defined below)
    
    y -= dy
    
    # ========================================================================
    # PHASE 2: AUGMENTATION (Prompt Construction)
    # ========================================================================
    
    # Main augmentation box
    create_box(ax, x_center_col, y, box_width, box_height,
               'Prompt Construction\n(Target + Cases + Statistics)',
               colors['blue'])
    
    y_augmentation = y
    
    # Solid arrow between Top-25 Similar Patients and Prompt Construction
    create_arrow(ax, x_center_col, y_retrieved - box_height/2, x_center_col, y_augmentation + box_height/2,
                 label='', style='solid', color='black', spacing=0.1, linewidth=1.2)
    
    # ========================================================================
    # DIAMOND STRUCTURE ON LEFT SIDE
    # ========================================================================
    
    # Calculate positions for diamond structure to create a baseball diamond shape
    # with uniform spacing between all points
    # 
    # Diamond points:
    # - Top: Training Database at (x_left_col=3.5, y_training_db)
    # - Right: DTW-Based Similar Patient Matching at (x_center_col=8, y_dtw_matching)
    # - Left: New box at same y as DTW Matching (y_dtw_matching)
    # - Bottom: New box at same x as Top point, positioned between retrieved and augmentation
    
    # Smaller box dimensions for diamond structure
    diamond_box_width = 2.2
    diamond_box_height = 0.55
    
    # Left point: positioned to the left of training database
    x_left_diamond = 0.5  # Positioned within visible area (xlim goes to -1)
    y_left_diamond = y_dtw_matching  # Same horizontal axis as DTW-Based Matching
    
    # Bottom point: positioned to create proper diamond spacing
    x_bottom_diamond = x_left_col  # Vertically aligned with Training DB (top point)
    y_bottom_diamond = y_retrieved - 0.3  # Moved upward, between retrieved and augmentation
    
    # Create left diamond box (smaller)
    create_box(ax, x_left_diamond, y_left_diamond, diamond_box_width, diamond_box_height,
               'Euclidean Pre-Filter\n(Top-500 candidates)',
               colors['yellow'])
    
    # Create bottom diamond box (smaller)
    create_box(ax, x_bottom_diamond, y_bottom_diamond, diamond_box_width, diamond_box_height,
               'Primary-Only DTW\n(Compute distances)',
               colors['yellow'])
    
    # ========================================================================
    # ARROWS: LEFT DIAMOND CYCLE (Training DB → Left Diamond → Bottom Diamond)
    # ========================================================================
    
    # Arrow 1: Training DB cylinder (left midline) to Left Diamond (top midline)
    # Regular outer curve only
    cylinder_exit_x = x_left_col - (1.8/2 + 0.3)  # Left edge of cylinder
    cylinder_exit_y = y_training_db  # Midline of cylinder
    left_diamond_entry_x = x_left_diamond  # Midline (center x)
    left_diamond_entry_y = y_left_diamond + (diamond_box_height/2 + 0.3)  # Top edge
    create_curved_arrow(ax, cylinder_exit_x, cylinder_exit_y, left_diamond_entry_x, left_diamond_entry_y,
                       label='', color='black', linewidth=1.2, spacing=0.25, rad=0.4)
    
    # Arrow 2: Left Diamond (bottom midline) to Bottom Diamond (left midline)
    # Regular outer curve only
    left_diamond_exit_x = x_left_diamond  # Midline (center x)
    left_diamond_exit_y = y_left_diamond - (diamond_box_height/2 + 0.3)  # Bottom edge
    bottom_diamond_entry_x = x_bottom_diamond - (diamond_box_width/2 + 0.3)  # Left edge
    bottom_diamond_entry_y = y_bottom_diamond  # Midline
    create_curved_arrow(ax, left_diamond_exit_x, left_diamond_exit_y, 
                       bottom_diamond_entry_x, bottom_diamond_entry_y,
                       label='', color='black', linewidth=1.2, spacing=0.25, rad=0.4)
    
    # ========================================================================
    # ARROWS: CONNECTIONS TO/FROM DTW-BASED MATCHING
    # ========================================================================
    
    # Arrow 3: DTW-Based to Training DB cylinder (straight arrow)
    dtw_exit_x = x_center_col - (box_width/2 + 0.3)  # Left edge with spacing
    dtw_exit_y = y_dtw_matching  # Midline
    cylinder_entry_x = x_left_col + (1.8/2 + 0.15)  # Right edge of cylinder
    cylinder_entry_y = y_training_db  # Midline of cylinder
    create_arrow(ax, dtw_exit_x, dtw_exit_y, cylinder_entry_x, cylinder_entry_y,
                 label='', style='dashed', color='black', spacing=0.25, linewidth=1.2)
    
    # Arrow 4: Bottom Diamond Box to Top-25 Similar Patients (straight arrow)
    # Using direct box edge positions without extra spacing offset
    bottom_diamond_exit_x = x_bottom_diamond + (diamond_box_width/2)  # Right edge
    bottom_diamond_exit_y = y_bottom_diamond  # Midline
    retrieved_entry_x = x_center_col - (box_width/2)  # Left edge
    retrieved_entry_y = y_retrieved  # Midline of retrieved box
    create_arrow(ax, bottom_diamond_exit_x, bottom_diamond_exit_y, retrieved_entry_x, retrieved_entry_y,
                 label='', style='dashed', color='black', spacing=0.35, linewidth=1.2)
    
    # ========================================================================
    # END ARROW CONNECTIONS
    # ========================================================================
    
    # DTW-ONLY FORECASTING BOX (right side) - NEW BOX ABOVE PROMPT COMPONENTS
    x_prompt_detail = 13.5
    dtw_only_box_width = 3.5
    dtw_only_box_height = 0.7
    # Position at same vertical level as Top-25 Similar Patients box for horizontal arrow
    y_dtw_only = y_retrieved
    
    # Duplicate "Compare to Ground Truth" box positioned above DTW-only Forecasting
    y_compare_dtw_only = y_dtw_only + dy
    create_box(ax, x_prompt_detail, y_compare_dtw_only, dtw_only_box_width, dtw_only_box_height,
               'Compare to Ground Truth\n(Days 11-20 Accuracy)', colors['yellow'])
    
    create_box(ax, x_prompt_detail, y_dtw_only, dtw_only_box_width, dtw_only_box_height,
               'DTW-only Forecasting:\nInverse Distance Weighted Averaging',
               colors['yellow'])
    
    # Arrow from Top-25 Similar Patients to DTW-only box (horizontal arrow)
    # Using direct box edge positions without extra spacing offset for equal spacing
    retrieved_exit_x = x_center_col + (box_width/2)  # Right edge of Top-25 box
    retrieved_exit_y = y_retrieved  # Midline of retrieved box
    dtw_only_entry_x = x_prompt_detail - (dtw_only_box_width/2)  # Left edge of DTW-only box
    dtw_only_entry_y = y_dtw_only  # Midline of DTW-only box (same as retrieved_exit_y for horizontal arrow)
    create_arrow(ax, retrieved_exit_x, retrieved_exit_y, dtw_only_entry_x, dtw_only_entry_y,
                 label='', style='dashed', color='black', spacing=0.35, linewidth=1.2)
    
    # Solid arrow pointing UP between DTW-only Forecasting and Compare to Ground Truth (DTW-only)
    create_arrow(ax, x_prompt_detail, y_dtw_only + dtw_only_box_height/2, x_prompt_detail, y_compare_dtw_only - dtw_only_box_height/2,
                 label='', style='solid', color='black', spacing=0.1, linewidth=1.2)
    
    # PROMPT DETAIL BOX (right side)
    # Box dimensions and positioning
    prompt_box_width = 4.4
    prompt_box_height = 1.7
    # Position box so its top border aligns with top border of Prompt Construction box
    # Prompt Construction box: center at y_augmentation, height = box_height (0.65)
    # Top of Prompt Construction = y_augmentation + box_height/2 = y_augmentation + 0.325
    # For Prompt Components box (height 1.7) to have top at same position:
    # Center should be at: y_augmentation + 0.325 - prompt_box_height/2 = y_augmentation - 0.525
    y_prompt_detail = y_augmentation - 0.525
    
    # Create dashed box showing prompt components
    prompt_detail_box = FancyBboxPatch((x_prompt_detail - prompt_box_width / 2, y_prompt_detail - prompt_box_height / 2), 
                                        prompt_box_width, prompt_box_height,
                                        boxstyle="round,pad=0.05", 
                                        facecolor=colors['blue'], edgecolor='#424242', 
                                        linewidth=1.5, linestyle='--')
    ax.add_patch(prompt_detail_box)
    
    # Prompt detail text - header
    ax.text(x_prompt_detail, y_prompt_detail + 0.7, 'Prompt Components', 
            fontsize=8, fontweight='bold', ha='center')
    
    # Component 1
    ax.text(x_prompt_detail - 2.0, y_prompt_detail + 0.4, '• Target patient summary', 
            fontsize=6.5, ha='left', fontweight='bold')
    ax.text(x_prompt_detail - 1.8, y_prompt_detail + 0.2, '(Days 1-10, demographics)', 
            fontsize=6, ha='left', color='#666')
    
    # Component 2
    ax.text(x_prompt_detail - 2.0, y_prompt_detail - 0.05, '• Top-3 DTW cases', 
            fontsize=6.5, ha='left', fontweight='bold')
    ax.text(x_prompt_detail - 1.8, y_prompt_detail - 0.25, '(Full trajectories + outcomes)', 
            fontsize=6, ha='left', color='#666')
    
    # Component 3
    ax.text(x_prompt_detail - 2.0, y_prompt_detail - 0.5, '• Empirical statistics', 
            fontsize=6.5, ha='left', fontweight='bold')
    ax.text(x_prompt_detail - 1.8, y_prompt_detail - 0.7, '(From all top-25 cases)', 
            fontsize=6, ha='left', color='#666')
    
    # Calculate y_prompt before using it in dashed arrows
    y -= dy
    
    # Output: Augmented prompt
    create_box(ax, x_center_col, y, box_width, box_height,
               'Augmented LLM Prompt', colors['blue'])
    y_prompt = y
    
    # Solid arrow between Augmented LLM Prompt and Prompt Sent to LLM
    # (positioned here before y_generation is defined below)
    
    y -= dy
    
    # Dashed arrows to/from prompt detail
    y_upper_prompt = y_augmentation  # From midpoint of Prompt Construction box
    x_start_prompt = x_center_col + box_width/2 + 0.2
    x_end_prompt = x_prompt_detail - 2.2 - 0.1
    
    straight_out_prompt = FancyArrowPatch(
        (x_start_prompt, y_upper_prompt),
        (x_end_prompt, y_upper_prompt),
        arrowstyle='->', mutation_scale=12,
        linewidth=1.2, color='black',
        linestyle='--',
        zorder=1
    )
    ax.add_patch(straight_out_prompt)
    
    ax.text((x_start_prompt + x_end_prompt) / 2, y_upper_prompt + 0.10, 
            'Build prompt', fontsize=6, style='italic', color='black', ha='center')
    
    y_lower_prompt = y_prompt  # Point toward midpoint of Augmented LLM Prompt box
    straight_in_prompt = FancyArrowPatch(
        (x_end_prompt, y_lower_prompt),
        (x_start_prompt, y_lower_prompt),
        arrowstyle='->', mutation_scale=12,
        linewidth=1.2, color='black',
        linestyle='--',
        zorder=1
    )
    ax.add_patch(straight_in_prompt)
    
    ax.text((x_start_prompt + x_end_prompt) / 2, y_lower_prompt - 0.20, 
            'Augmented prompt', fontsize=6, style='italic', color='black', ha='center')
    
    # ========================================================================
    # PHASE 3: GENERATION (LLM Forecasting)
    # ========================================================================
    
    # Main generation box
    create_box(ax, x_center_col, y, box_width, box_height,
               'Prompt Sent to LLM\n(via OpenRouter API)',
               colors['green'])
    
    y_generation = y
    
    # Solid arrow between Augmented LLM Prompt and Prompt Sent to LLM
    create_arrow(ax, x_center_col, y_prompt - box_height/2, x_center_col, y_generation + box_height/2,
                 label='', style='solid', color='black', spacing=0.1, linewidth=1.2)
    
    y -= dy
    
    # Response parsing box
    create_box(ax, x_center_col, y, box_width, box_height,
               'Response Parsing & Validation\n(Extract structured predictions)',
               colors['green'])
    
    y_parsing = y
    y -= dy
    
    # Final output
    create_box(ax, x_center_col, y, box_width, box_height,
               'Compare to Ground Truth\n(Days 11-20 Accuracy)', colors['green'])
    
    y_compare = y
    
    # Solid arrow between Response Parsing and Compare to Ground Truth
    create_arrow(ax, x_center_col, y_parsing - box_height/2, x_center_col, y_compare + box_height/2,
                 label='', style='solid', color='black', spacing=0.1, linewidth=1.2)
    
    # ========================================================================
    # LLM FORECASTING DETAIL BOX (left side)
    # ========================================================================
    
    # Box dimensions and positioning
    llm_box_width = 4.4
    llm_box_height = 1.5
    # Position box so its vertical center aligns between Prompt Sent to LLM and Response Parsing
    # This will create a balanced layout
    y_llm_detail = (y_generation + y_parsing) / 2
    x_llm_detail = 1.5  # Left side of flowchart
    
    # Create dashed box showing LLM forecasting details
    llm_detail_box = FancyBboxPatch((x_llm_detail - llm_box_width / 2, y_llm_detail - llm_box_height / 2), 
                                     llm_box_width, llm_box_height,
                                     boxstyle="round,pad=0.05", 
                                     facecolor=colors['green'], edgecolor='#424242', 
                                     linewidth=1.5, linestyle='--')
    ax.add_patch(llm_detail_box)
    
    # LLM detail text - header
    ax.text(x_llm_detail, y_llm_detail + 0.55, 'LLM Forecasting', 
            fontsize=8, fontweight='bold', ha='center')
    
    # Structured response requirement
    ax.text(x_llm_detail - 2.0, y_llm_detail + 0.3, '• THINKING block', 
            fontsize=6.5, ha='left', fontweight='bold')
    ax.text(x_llm_detail - 1.8, y_llm_detail + 0.1, '(reasoning, draft, adjustments)', 
            fontsize=6, ha='left', color='#666')
    
    ax.text(x_llm_detail - 2.0, y_llm_detail - 0.15, '• JSON block', 
            fontsize=6.5, ha='left', fontweight='bold')
    ax.text(x_llm_detail - 1.8, y_llm_detail - 0.35, '(day_11...day_20: 0-4 scale)', 
            fontsize=6, ha='left', color='#666')
    
    # Dashed arrows to/from LLM detail
    # Arrow from Prompt Sent to LLM to LLM detail box
    generation_exit_x = x_center_col - (box_width/2 + 0.3)  # Left edge of generation box
    generation_exit_y = y_generation  # Midline of generation box
    llm_detail_entry_x = x_llm_detail + (llm_box_width/2 + 0.3)  # Right edge of LLM detail box
    llm_detail_entry_y = y_generation  # Align with generation box
    
    straight_out_llm = FancyArrowPatch(
        (generation_exit_x, generation_exit_y),
        (llm_detail_entry_x, llm_detail_entry_y),
        arrowstyle='->', mutation_scale=12,
        linewidth=1.2, color='black',
        linestyle='--',
        zorder=1
    )
    ax.add_patch(straight_out_llm)
    
    ax.text((generation_exit_x + llm_detail_entry_x) / 2, generation_exit_y + 0.10, 
            'LLM processes', fontsize=6, style='italic', color='black', ha='center')
    
    # Arrow from LLM detail box to Response Parsing
    llm_detail_exit_x = x_llm_detail + (llm_box_width/2 + 0.3)  # Right edge of LLM detail box
    llm_detail_exit_y = y_parsing  # Align with parsing box
    parsing_entry_x = x_center_col - (box_width/2 + 0.3)  # Left edge of parsing box
    parsing_entry_y = y_parsing  # Midline of parsing box
    
    straight_in_llm = FancyArrowPatch(
        (llm_detail_exit_x, llm_detail_exit_y),
        (parsing_entry_x, parsing_entry_y),
        arrowstyle='->', mutation_scale=12,
        linewidth=1.2, color='black',
        linestyle='--',
        zorder=1
    )
    ax.add_patch(straight_in_llm)
    
    ax.text((llm_detail_exit_x + parsing_entry_x) / 2, parsing_entry_y - 0.15, 
            'Structured response', fontsize=6, style='italic', color='black', ha='center')
    
    # ========================================================================
    # LEGEND: Color Coding
    # ========================================================================
    
    legend_x = 13.5
    legend_y = 2.8  # Shifted up for better bottom margin
    legend_width = 3.5
    legend_height = 1.4
    
    # Legend container box
    legend_box = FancyBboxPatch((legend_x - legend_width/2, legend_y - legend_height/2), 
                                legend_width, legend_height,
                                boxstyle="round,pad=0.1", 
                                facecolor='white', edgecolor='#424242', 
                                linewidth=1.2)
    ax.add_patch(legend_box)
    
    # Legend title
    ax.text(legend_x, legend_y + 0.55, 'Pipeline Phase', 
            fontsize=9, fontweight='bold', ha='center')
    
    # Legend items - Yellow (Retrieval)
    yellow_box = FancyBboxPatch((legend_x - 1.4, legend_y + 0.15), 0.35, 0.25,
                                boxstyle="round,pad=0.02", 
                                facecolor=colors['yellow'], edgecolor='black', linewidth=0.8)
    ax.add_patch(yellow_box)
    ax.text(legend_x - 0.95, legend_y + 0.275, 'Retrieval (DTW Matching)', 
            fontsize=6, ha='left', va='center')
    
    # Legend items - Blue (Augmentation)
    blue_box = FancyBboxPatch((legend_x - 1.4, legend_y - 0.2), 0.35, 0.25,
                              boxstyle="round,pad=0.02", 
                              facecolor=colors['blue'], edgecolor='black', linewidth=0.8)
    ax.add_patch(blue_box)
    ax.text(legend_x - 0.95, legend_y - 0.075, 'Augmentation (Prompt)', 
            fontsize=6, ha='left', va='center')
    
    # Legend items - Green (Generation)
    green_box = FancyBboxPatch((legend_x - 1.4, legend_y - 0.55), 0.35, 0.25,
                               boxstyle="round,pad=0.02", 
                               facecolor=colors['green'], edgecolor='black', linewidth=0.8)
    ax.add_patch(green_box)
    ax.text(legend_x - 0.95, legend_y - 0.425, 'Generation (LLM Forecast)', 
            fontsize=6, ha='left', va='center')
    
    # Don't use tight_layout() to preserve the full xlim range for the Left Diamond Box
    return fig

def main():
    """Generate and save DTW-RAG flowchart"""
    
    print("="*70)
    print("GENERATING DTW-BASED RAG FORECASTING FLOWCHART")
    print("="*70)
    
    # Determine output directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / 'publication_ready_tables_and_figures' / 'new_flowcharts'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create flowchart
    print("\n[1/2] Building flowchart structure...")
    fig = create_dtw_rag_flowchart()
    
    # Save files
    print("[2/2] Saving flowchart...")
    
    png_path = output_dir / 'dtw_rag_methodology_flowchart.png'
    pdf_path = output_dir / 'dtw_rag_methodology_flowchart.pdf'
    
    # Save as PNG (high resolution for presentations)
    fig.savefig(png_path, dpi=300, pad_inches=0.2, facecolor='white')
    print(f"   ✓ PNG saved: {png_path}")
    
    # Save as PDF (vector format for publication)
    fig.savefig(pdf_path, format='pdf', pad_inches=0.2, facecolor='white')
    print(f"   ✓ PDF saved: {pdf_path}")
    
    plt.close(fig)
    
    print("\n" + "="*70)
    print("✅ FLOWCHART GENERATION COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print(f"  • PNG (high-res, 300 DPI): {png_path}")
    print(f"  • PDF (vector, publication): {pdf_path}")
    
    print("\n📊 Flowchart visualizes:")
    print("  • Three-phase RAG pipeline (Retrieval → Augmentation → Generation)")
    print("  • DTW-based temporal similarity matching (primary condition only)")
    print("  • Prompt construction with top-3 cases + empirical statistics")
    print("  • LLM forecasting with structured reasoning (Llama 3.3 70B)")
    print("  • Parallel processing architecture")
    
    print("\n💡 To view:")
    print(f"   open {png_path}")
    print(f"   open {pdf_path}")
    
    print("\n📝 Design features:")
    print("  • Publication-ready quality (300 DPI)")
    print("  • Clear phase separation with color coding")
    print("  • Explicit data flow from training set through DTW algorithm")
    print("  • Compact layout (8.5\" x 11\" landscape)")
    print("  • Matches data pipeline flowchart style")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
