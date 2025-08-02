import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set Times New Roman as the default font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Data for the clinical conditions
notes = [
    "severe dehydration", "moderate liver failure", "moderate renal failure",
    "normal thrombocytopenia", "moderate liver failure", "moderate hypoglycemia",
    "severe liver failure", "moderate anemia"
]

# Colors for each condition
note_colors = {
    "severe dehydration": "tab:blue",
    "moderate liver failure": "tab:orange",
    "moderate renal failure": "tab:green",
    "normal thrombocytopenia": "tab:red",
    "moderate hypoglycemia": "tab:purple",
    "severe liver failure": "tab:brown",
    "moderate anemia": "tab:cyan"
}

# Data
labs = {
    'creatinine_mgdl': {
        'true': [1.57, 1.04, 2.87, 1.06, 1.35, 0.98, 1.19, 1.11],
        'pred': [1.54, 0.96, 3.01, 1.01, 0.96, 0.93, 0.80, 0.93],
    },
    'hemoglobin_gdl': {
        'true': [14.29, 11.39, 14.49, 13.89, 11.02, 13.86, 9.37, 9.77],
        'pred': [13.79, 11.13, 14.09, 14.22, 11.08, 14.17, 9.56, 10.01],
    },
    'sodium_mEqL': {
        'true': [147.66, 140.09, 139.81, 140.16, 140.17, 140.11, 139.92, 140.02],
        'pred': [148.62, 141.24, 140.62, 141.59, 140.84, 141.67, 140.92, 140.60],
    },
    'WBC_kul': {
        'true': [7.55, 6.91, 6.92, 6.94, 6.87, 7.36, 7.31, 6.96],
        'pred': [7.03, 6.71, 7.27, 7.07, 6.69, 7.00, 7.57, 7.18],
    },
    'platelet_kul': {
        'true': [249.66, 249.93, 249.84, 94.89, 250.02, 249.98, 250.11, 250.06],
        'pred': [250.89, 251.46, 250.86, 101.76, 251.53, 250.89, 252.96, 250.52],
    },
    'BUN_mgdl': {
        'true': [15.23, 11.90, 20.02, 12.10, 12.00, 12.01, 11.67, 11.94],
        'pred': [14.63, 12.28, 20.46, 12.08, 12.23, 12.10, 11.49, 12.13],
    },
    'RBC_million_ul': {
        'true': [4.58, 4.80, 4.45, 4.40, 4.67, 4.64, 4.50, 4.53],
        'pred': [4.64, 4.62, 4.64, 4.60, 4.61, 4.66, 4.61, 4.60],
    },
    'hematocrit_percent': {
        'true': [39.81, 40.10, 40.11, 40.12, 40.14, 40.17, 39.85, 39.82],
        'pred': [40.36, 40.40, 40.40, 40.41, 40.29, 40.43, 40.32, 40.36],
    },
    'MCV_fl': {
        'true': [89.96, 90.13, 90.18, 89.89, 90.44, 89.80, 89.95, 90.22],
        'pred': [90.69, 90.78, 90.76, 90.79, 90.53, 90.74, 90.55, 90.73],
    },
    'MCH_pg': {
        'true': [30.05, 50.46, 29.99, 29.85, 49.99, 29.97, 61.03, 30.07],
        'pred': [30.35, 50.80, 30.84, 30.32, 50.79, 30.49, 59.50, 29.86],
    },
    'MCHC_gdl': {
        'true': [32.92, 33.02, 33.11, 32.86, 33.21, 33.17, 32.94, 33.02],
        'pred': [33.27, 33.31, 33.35, 33.36, 33.22, 33.35, 33.27, 33.31],
    },
    'RDW_percent': {
        'true': [13.02, 13.25, 12.74, 12.73, 12.62, 12.80, 12.96, 12.69],
        'pred': [13.19, 13.17, 13.20, 13.21, 13.13, 13.22, 13.17, 13.20],
    },
    'potassium_mEqL': {
        'true': [5.04, 5.11, 4.65, 4.59, 4.85, 5.06, 4.61, 5.23],
        'pred': [5.06, 5.07, 5.07, 5.07, 5.06, 5.07, 5.05, 5.03],
    },
    'chloride_mEqL': {
        'true': [100.02, 100.24, 100.62, 99.66, 99.69, 85.34, 100.05, 99.96],
        'pred': [99.97, 100.30, 101.05, 100.87, 100.07, 85.77, 101.46, 100.98],
    },
    'bicarbonate_mEqL': {
        'true': [24.34, 23.64, 24.22, 24.31, 24.10, 24.14, 23.71, 23.90],
        'pred': [24.13, 24.17, 24.16, 24.20, 24.10, 24.19, 24.12, 24.11],
    },
    'glucose_mgdl': {
        'true': [110.18, 110.03, 110.12, 110.01, 110.02, 110.05, 110.34, 109.98],
        'pred': [110.93, 111.07, 111.01, 111.11, 110.77, 111.01, 110.81, 110.95],
    },
    'calcium_mgdl': {
        'true': [9.25, 9.17, 9.10, 9.27, 8.76, 9.16, 9.06, 9.23],
        'pred': [9.14, 9.12, 9.16, 9.11, 9.10, 9.16, 9.27, 9.14],
    }
}

labs_to_plot = list(labs.keys())

# Calculate and print percentage differences between true and predicted values
print("True vs Predicted Percentage Differences:")
print("=" * 60)

all_percentage_diffs = []  # To collect all percentage differences for overall statistics
lab_avg_diffs = []  # To collect average differences for each lab for plotting

for lab in labs_to_plot:
    true_vals = labs[lab]['true']
    pred_vals = labs[lab]['pred']
    
    # Calculate percentage differences for each sample
    percentage_diffs = []
    for true_val, pred_val in zip(true_vals, pred_vals):
        if true_val != 0:  # Avoid division by zero
            percentage_diff = ((pred_val - true_val) / true_val) * 100
            percentage_diffs.append(percentage_diff)
            all_percentage_diffs.append(percentage_diff)  # Add to overall collection
        else:
            percentage_diffs.append(0)  # Handle zero true values
            all_percentage_diffs.append(0)
    
    # Calculate average for this lab
    avg_percentage_diff = np.mean(percentage_diffs)
    lab_avg_diffs.append(avg_percentage_diff)
    
    print(f"{lab}: {avg_percentage_diff:.2f}%")

# Calculate overall average across all labs
overall_avg = np.mean(all_percentage_diffs)

print("=" * 60)
print(f"OVERALL AVERAGE PERCENTAGE DIFFERENCE: {overall_avg:.2f}%")
print("=" * 60)
print()

# Create dot plot of all individual accuracy percentages
fig_dot, ax_dot = plt.subplots(figsize=(7, 6))  # Changed to narrower ratio (7:6)

# Convert to absolute values for accuracy visualization (closer to 0% = more accurate)
all_abs_percentage_diffs = [abs(diff) for diff in all_percentage_diffs]

# Create colors for different lab measurements
color_palette = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 
                 'olive', 'cyan', 'navy', 'maroon', 'lime', 'teal', 'indigo', 'gold', 'coral']

# Prepare data for boxplots - group by lab measurement
boxplot_data = []
measurement_labels = []

for i, lab in enumerate(labs_to_plot):
    # Get the 8 absolute percentage differences for this lab
    start_idx = i * 8
    end_idx = start_idx + 8
    lab_data = all_abs_percentage_diffs[start_idx:end_idx]
    boxplot_data.append(lab_data)
    
    # Clean up measurement names
    clean_name = lab.replace('_', ' ').replace(' mgdl', '').replace(' gdl', '').replace(' mEqL', '').replace(' kul', '').replace(' percent', '').replace(' fl', '').replace(' pg', '')
    
    # Handle special medical abbreviations before applying title case
    if 'BUN' in clean_name:
        clean_name = clean_name.replace('BUN', 'BUN')  # Keep as is
    elif 'WBC' in clean_name:
        clean_name = clean_name.replace('WBC', 'WBC')  # Keep as is
    elif 'MCHC' in clean_name:
        clean_name = clean_name.replace('MCHC', 'MCHC')  # Keep as is
    elif 'MCH' in clean_name:
        clean_name = clean_name.replace('MCH', 'MCH')  # Keep as is
    elif 'MCV' in clean_name:
        clean_name = clean_name.replace('MCV', 'MCV')  # Keep as is
    elif 'RDW' in clean_name:
        clean_name = clean_name.replace('RDW', 'RDW')  # Keep as is
    else:
        clean_name = clean_name.title()  # Apply title case only to non-abbreviations
    
    measurement_labels.append(clean_name)

# Create vertical boxplots
box_plot = ax_dot.boxplot(boxplot_data, patch_artist=True, 
                         showfliers=True, flierprops=dict(marker='x', markersize=10, alpha=0.7, markeredgewidth=3))

# Make each boxplot hollow with colored outlines and colored outliers
for i, (box, fliers) in enumerate(zip(box_plot['boxes'], box_plot['fliers'])):
    color = color_palette[i % len(color_palette)]
    
    # Make box hollow with colored outline
    box.set_facecolor('white')
    box.set_edgecolor(color)
    box.set_linewidth(2)
    
    # Color the median line
    box_plot['medians'][i].set_color(color)
    box_plot['medians'][i].set_linewidth(2)
    
    # Color the whiskers
    box_plot['whiskers'][i*2].set_color(color)
    box_plot['whiskers'][i*2+1].set_color(color)
    box_plot['whiskers'][i*2].set_linewidth(2)
    box_plot['whiskers'][i*2+1].set_linewidth(2)
    
    # Color the caps
    box_plot['caps'][i*2].set_color(color)
    box_plot['caps'][i*2+1].set_color(color)
    box_plot['caps'][i*2].set_linewidth(2)
    box_plot['caps'][i*2+1].set_linewidth(2)
    
    # Color the outliers (X markers)
    fliers.set_markerfacecolor(color)
    fliers.set_markeredgecolor(color)
    fliers.set_markersize(12)
    fliers.set_markeredgewidth(3)
    fliers.set_alpha(0.8)

# Customize the plot with larger font sizes
ax_dot.set_xlabel('Lab Measurement', fontsize=14, fontfamily='serif')
ax_dot.set_ylabel('Absolute Percentage Difference (%)', fontsize=14, fontfamily='serif')
ax_dot.set_title('Prediction Error by Lab Measurement\n(Lower values = Better accuracy)', fontsize=16, fontfamily='serif')
ax_dot.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines

# Set x-axis labels manually
ax_dot.set_xticklabels(measurement_labels, rotation=45, ha='right', fontsize=10, fontfamily='serif')
ax_dot.tick_params(axis='y', which='major', labelsize=12)

# Set y-axis to start at 0 with no gap
ax_dot.set_ylim(bottom=0)

# Add horizontal line at overall average
overall_abs_avg = float(np.mean(all_abs_percentage_diffs))
ax_dot.axhline(y=overall_abs_avg, color='black', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Average Error: {overall_abs_avg:.2f}%')
ax_dot.legend(fontsize=12, prop={'family': 'serif'})

plt.tight_layout()
plt.savefig('accuracy_dotplot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Dot plot saved as 'accuracy_dotplot.png'")
print()

# Create poster scatterplots for specific lab measurements
selected_labs = ['platelet_kul', 'BUN_mgdl', 'hemoglobin_gdl', 'MCH_pg']
selected_indices = [labs_to_plot.index(lab) for lab in selected_labs]

# Get the colors for these specific labs from the dot plot color palette
selected_colors = [color_palette[i % len(color_palette)] for i in selected_indices]

# Plot setup for poster scatterplots (2x2 grid)
fig_poster, axes_poster = plt.subplots(2, 2, figsize=(12, 10))
axes_poster = axes_poster.flatten()

for i, lab in enumerate(selected_labs):
    ax = axes_poster[i]
    true_vals = labs[lab]['true']
    pred_vals = labs[lab]['pred']
    max_val = max(true_vals + pred_vals)
    lab_color = selected_colors[i]

    for j in range(len(true_vals)):
        ax.scatter(true_vals[j], pred_vals[j], 
                   color=lab_color, marker='x', s=200, alpha=0.9, linewidths=3)

    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.5)
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    
    # Clean up lab name for title with proper capitalization
    clean_title = lab.replace('_', ' ').replace(' mgdl', '').replace(' gdl', '').replace(' mEqL', '').replace(' kul', '').replace(' percent', '').replace(' fl', '').replace(' pg', '')
    
    # Handle special medical abbreviations before applying title case
    if 'BUN' in clean_title:
        clean_title = clean_title.replace('BUN', 'BUN')  # Keep as is
    elif 'WBC' in clean_title:
        clean_title = clean_title.replace('WBC', 'WBC')  # Keep as is
    elif 'MCHC' in clean_title:
        clean_title = clean_title.replace('MCHC', 'MCHC')  # Keep as is
    elif 'MCH' in clean_title:
        clean_title = clean_title.replace('MCH', 'MCH')  # Keep as is
    elif 'MCV' in clean_title:
        clean_title = clean_title.replace('MCV', 'MCV')  # Keep as is
    elif 'RDW' in clean_title:
        clean_title = clean_title.replace('RDW', 'RDW')  # Keep as is
    else:
        clean_title = clean_title.title()  # Apply title case only to non-abbreviations
    
    ax.set_title(clean_title, fontsize=24, fontfamily='serif')
    ax.set_xlabel('True Lab Value', fontsize=20, fontfamily='serif')
    ax.set_ylabel('CNN-Predicted Lab Value', fontsize=20, fontfamily='serif')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('poster_scatters.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Poster scatterplots saved as 'poster_scatters.png'")
print()

# Plot setup
fig, axes = plt.subplots(5, 4, figsize=(16, 14))  # Adjust figure size
axes = axes.flatten()

for i, lab in enumerate(labs_to_plot):
    ax = axes[i]
    true_vals = labs[lab]['true']
    pred_vals = labs[lab]['pred']
    max_val = max(true_vals + pred_vals)

    for j in range(len(true_vals)):
        ax.scatter(true_vals[j], pred_vals[j], 
                   color=note_colors[notes[j]], marker='x', s=150, linewidths=3)

    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--')
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    ax.set_title(lab, fontfamily='serif')
    ax.set_xlabel('True Lab Value', fontfamily='serif')
    ax.set_ylabel('CNN-Predicted Lab Value', fontfamily='serif')
    ax.grid(True)

# Add custom legend to the right side of the plot
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in note_colors.items()]
fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.15, 0.5), title="Condition", prop={'family': 'serif'})

plt.tight_layout(rect=(0, 0, 0.95, 1))  # Leave room for the right-side legend
plt.savefig('scatterplots.tif', dpi=800, bbox_inches='tight')
plt.show()

