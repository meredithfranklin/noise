import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Initial Preprocessing ---
try:
    file_path = r"C:\Users\franc\PycharmProjects\Summer_research\Codes\sound_pollutant_met_data.csv"
    df = pd.read_csv(file_path)
    print("--- Initial Data Head ---")
    print(df.head())
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please make sure the file exists in the specified path.")
    exit()

df.rename(columns=lambda col: col.replace(" ", "_"), inplace=True)
print("\n--- Column Names After Renaming ---")
print(df.columns)

df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
df = df.dropna(subset=['time_utc'])
df_subset = df[(df['time_utc'] >= '2023-07-01') & (df['time_utc'] <= '2024-05-31')].copy()


# --- 2. Define Variables and Create Plot Labels ---
vars_original = [
    "LAFmax", "LAFmin", "LCpeak", "LCeq", "LAeq",
    "LZeq_12.5Hz", "LZeq_16Hz", "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz",
    "LZeq_50Hz", "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz",
    "LZeq_160Hz", "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz",
    "LZeq_500Hz", "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz",
    "LZeq_1.6kHz", "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz",
    "LZeq_5kHz", "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz",
    "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr",
    "nox", "ch4", "co", "co2_ppm", "h2s"
]

plot_labels = [
    "LAFmax", "LAFmin", "LCpeak", "LCeq", "LAeq",
    "LZeq 12.5Hz", "LZeq 16Hz", "LZeq 20Hz", "LZeq 25Hz", "LZeq 31.5Hz", "LZeq 40Hz",
    "LZeq 50Hz", "LZeq 63Hz", "LZeq 80Hz", "LZeq 100Hz", "LZeq 125Hz",
    "LZeq 160Hz", "LZeq 200Hz", "LZeq 250Hz", "LZeq 315Hz", "LZeq 400Hz",
    "LZeq 500Hz", "LZeq 630Hz", "LZeq 800Hz", "LZeq 1kHz", "LZeq 1.25kHz",
    "LZeq 1.6kHz", "LZeq 2kHz", "LZeq 2.5kHz", "LZeq 3.15kHz", "LZeq 4kHz",
    "LZeq 5kHz", "LZeq 6.3kHz", "LZeq 8kHz", "LZeq 10kHz", "LZeq 12.5kHz", "LZeq 16kHz",
    "Temperature (°F)", "Wind Speed (m/s)", "Wind Direction (°)", "Relative Humidity (%)", "Atmospheric Pressure", # Meteorology renamed
    "NOx", "CH4", "CO", "CO2", "H2S" # Pollutants renamed
]

# check for mismatch between original variable count and plot label count
print(len(vars_original) == len(plot_labels))
# True

# --- 3. Calculate Correlation Matrix ---
df_corr = df_subset[vars_original].copy()
corr_matrix = df_corr.corr(numeric_only=True)

# --- 4. Apply Combined Plotting Style (Text Lower, Bubble Upper) ---
try:
    mpl.rcParams['font.family'] = 'Calibri'
except Exception as e:
    print(f"Warning: Could not set font to 'Calibri'. Using default font. Error: {e}")

corr_matrix.index = plot_labels
corr_matrix.columns = plot_labels

# Mask for the upper triangle (heatmap function will ignore this part)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(18, 16))
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw the heatmap
# IMPORTANT: Set annot=True to show text in the lower triangle
# The mask ensures heatmap only draws data in the lower triangle
sns.heatmap(
    corr_matrix,
    mask=mask,          # Apply the mask to hide the upper triangle & diagonal
    cmap=cmap,          # Use the diverging blue-red colormap
    center=0,           # Center the colormap at 0 for correlations
    vmin=-1, vmax=1,    # Set color limits to [-1, 1]
    square=True,        # Ensure cells are square
    annot=True,         # Show text annotations (in lower triangle)
    fmt=".2f",          # Format annotations to 2 decimal places
    linewidths=0.5,     # Add lines between cells
    cbar_kws={"shrink": .75}, # Adjust color bar size
    annot_kws={"size": 7}
)

# Create the bubbles: overlay the circles manually for the UPPER triangle.
num_vars = len(corr_matrix.columns)
for i in range(num_vars):
    for j in range(num_vars):
        # Only plot circles in the UPPER triangle (i < j)
        if i < j: # <--- Condition changed back to i < j
            value = corr_matrix.iloc[i, j]
            # Adjust the scaling factor (e.g., 400) to change bubble size range
            circle_size = np.abs(value) * 400
            # Normalize correlation value from [-1, 1] to [0, 1] for color mapping
            color_val_normalized = (value + 1) / 2
            color = cmap(color_val_normalized)
            # Plot the circle (scatter point) - ensures it's drawn on top
            ax.scatter(j + 0.5, i + 0.5, s=circle_size, color=color, alpha=0.7, edgecolors='white', linewidth=0.5)

# --- 5. Customize Axes and Labels ---
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(rotation=90, ha='center', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
# plt.title("Correlation Heatmap: Sound, Meteorology & Pollutants (Text Lower, Bubble Upper)", fontsize=16, pad=20) # Updated title
plt.tight_layout(rect=[0, 0, 1, 0.97])

# --- 6. Save the plot as a png and Show Plot ---
try:
    plt.savefig('combined_correlation_matrix.png',  # Changed filename extension
                format='png',                      # Changed format to png
                dpi=300,                           # Added DPI setting
                bbox_inches='tight')               # Kept tight bounding box
    print("\nCombined correlation heatmap saved as 'combined_correlation_matrix.png' (300 DPI)") # Updated print message
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show()
print("\nScript finished.")