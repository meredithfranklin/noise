import pandas as pd
import numpy as np # import for politeness :)
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# --- Existing Data Loading and Preprocessing ---

# Note, this is url is the absolute path in my machine.
# You might want to change it into the dataset in your own absolute path.
try:
    df = pd.read_csv(r"/Users/meredith/Library/CloudStorage/GoogleDrive-mereditf@usc.edu/Shared drives/HEI Energy/papers/noise/analysis/sound_pollutant_met_data.csv")
    print("--- Initial Data Head ---")
    print(df.head())
except FileNotFoundError:
    print("Error: 'sound_pollutant_met_data.csv' not found. Please make sure the file exists.")
    exit() # Exit if file not found

df.rename(columns=lambda col: col.replace(" ", "_"), inplace=True)
print("\n--- Column Names After Renaming ---")
print(df.columns)

df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
df.dropna(subset=['time_utc'], inplace=True)

# Create the binary day night variable
print("\nCreating 'day_night' variable (day means 1 = 11am to <5pm)...")
df['day_night'] = ((df['time_utc'].dt.hour >= 11) & (df['time_utc'].dt.hour < 17)).astype(int)
print("Value counts for 'day_night':")
print(df['day_night'].value_counts(normalize=True))

df_subset = df[(df['time_utc'] >= '2023-07-01') & (df['time_utc'] <= '2024-05-31')].copy()

# Print initial means (optional, won't be in report)
print("\n--- Mean Values for Subset ---")
for var in ['nox', 'ch4', 'co', 'h2s', 'co2_ppm', 'no2', 'so2']:
     if var in df_subset.columns:
        mean_val = df_subset[var].mean()
        print(f"Mean {var.upper():<7}: {mean_val:.3f}")

print(f"\nSubset shape: {df_subset.shape}")

df_subset['LAF_max_min_diff'] = df_subset['LAFmax'] - df_subset['LAFmin']
df_subset['LZeq_25_2k_diff'] = df_subset['LZeq_25Hz'] - df_subset['LZeq_2kHz']

vars = ["LCpeak", "LCeq","LAFmax","LAFmin", 'LAF_max_min_diff', 'LZeq_25_2k_diff',
    "LZeq_12.5Hz", "LZeq_16Hz", "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz",
    "LZeq_50Hz", "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz",
    "LZeq_160Hz", "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz",
    "LZeq_500Hz", "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz",
    "LZeq_1.6kHz", "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz",
    "LZeq_5kHz", "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz",
    "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr", "day_night"]

# --- Define a dictionary for the mapping for better variable names in importance plot ---
variable_name_map = {
    "temp_f": "Temperature",
    "wsp_ms": "Wind Speed",
    "wdr_deg": "Wind Direction",
    "relh_percent": "Relative Humidity",
    "pressure_altcorr": "Atmospheric Pressure",
    "nox": "NOx",
    "ch4": "CH4",
    "co": "CO",
    "co2_ppm": "CO2",
    "h2s": "H2S",
    "no2": "NO2",
    "so2": "SO2",
    "day_night": "Day/Night (11am-5pm)"
}

# --- Function to clean variable names for plots ---
def clean_label(var_name, name_map):
    name = name_map.get(var_name, var_name) # Use value in dictionary corresponding to the key variable if available
    # if the variable key is not within dictionary key, then report the variable (such as frequency)
    return name.replace("_", " ") # Replace underscores with blank space.


# --- Loop Through Each Gas ---
for response in ['nox', 'ch4', 'h2s', 'co', 'co2_ppm', 'no2', 'so2']:

    # --- Create an empty list to store output lines for the report ---
    output_lines = []
    output_lines.append(f"========================= {response.upper()} MODEL =========================")

    # Drop rows with NA in predictors or target
    df_model = df_subset[vars + [response]].dropna()

    X = df_model[vars]
    selected_predictors = X.columns.tolist()
    y = df_model[response] # Define y here for consistency

    # Train/valid/test split
    # Combine X and y for splitting to ensure alignment
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    output_lines.append(f"\nObservations for {response.upper()}:")
    output_lines.append(f"  Train: {len(X_train)}")
    output_lines.append(f"  Valid: {len(X_valid)}")
    output_lines.append(f"  Test : {len(X_test)}")
    output_lines.append(f"  Total: {len(df_model)}")

    # --- Randomized Search CV ---
    param_grid = {
        'n_estimators': randint(300, 1201),
        'max_depth': randint(3, 13),
        'eta': uniform(0.01, 0.19)
    }
    xgb_model = xgb.XGBRegressor(
        objective = "reg:squarederror", eval_metric = "rmse", random_state = 42,
        tree_method = 'hist', n_jobs = -1
    )
    random_search = RandomizedSearchCV(
        estimator = xgb_model, param_distributions = param_grid, n_iter=20,
        scoring='neg_root_mean_squared_error', cv=5, verbose = 1, # Verbose=1 prints CV progress to console
        random_state=42, n_jobs = -1 # allow multicore computation in Prof's computer
    )

    print(f"\nStarting Randomized Search CV for {response.upper()}...") # Keep console progress
    random_search.fit(X_train, y_train)
    print(f"Randomized Search CV for {response.upper()} complete.") # Keep console progress

    best_params = random_search.best_params_
    output_lines.append("\nBest Hyperparameters Found by RandomizedSearchCV:")
    output_lines.append(str(best_params)) # Convert dict to string for report

    # --- Final Model Training ---
    final_model = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", random_state=42,
        n_jobs = -1, tree_method='hist', **best_params
    )
    print(f"\nTraining final model for {response.upper()}...") # Keep console progress
    final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=10)
    # If the performance on the validation set (RMSE metric) doesn't improve for 10 consecutive trees (rounds),
    # stop training early. This is t prevent overfitting.
    print(f"Final model training for {response.upper()} complete.") # Keep console progress

    # --- Evaluation ---
    valid_preds = final_model.predict(X_valid)
    test_preds = final_model.predict(X_test)
    valid_r2 = r2_score(y_valid, valid_preds)
    valid_rmse = root_mean_squared_error(y_valid, valid_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)

    output_lines.append(f"\nModel Performance:")
    output_lines.append(f"Validation R²:   {valid_r2:.3f}") # 3 decimals.
    output_lines.append(f"Validation RMSE: {valid_rmse:.3f}")
    output_lines.append(f"Test R²:         {test_r2:.3f}")
    output_lines.append(f"Test RMSE:       {test_rmse:.3f}")

    # Report the iteration number about early stopping if it occurred
    if final_model.best_iteration:
    # if early stopping occurs, then best_iteration is nonzero, then boolean True.
        output_lines.append(f"Best Iteration (Early Stopping): {final_model.best_iteration}")

    # --- Variable Importance (with F score aka importance_type = weight) ---
    importance_scores = final_model.get_booster().get_score(importance_type='weight') # use F score (weight)
    # F score: corresponds to the number of times a feature is used to split the data across all trees
    # it is just frequency scores.
    all_features = X_train.columns
    # # Create a list of F-scores for all input features, using 0 for unused features in the model.
    f_scores = [importance_scores.get(feature, 0) for feature in all_features]
    importance_df = pd.DataFrame({
        'Variable': all_features,
        'F-Score': f_scores
    })
    # Sort by F-Score descending: sort() does not change the original index
    # and we need reset_index() to manually change index
    importance_df = importance_df.sort_values(by='F-Score', ascending=False).reset_index(drop=True)

    output_lines.append("\nVariable Importance (F-Score - All Variables):")
    # Use to_string() to get a formatted table representation for the report
    output_lines.append(importance_df.to_string(index=False))

    # --- Frequency Variable Analysis (based on F-Score) ---
    importance_df_top20 = importance_df.head(20) # select top 20 important variables.
    top_20_vars = importance_df_top20['Variable'].tolist()

    ## analyze frequency variables and two difference variables within the importance set.
    freq_vars = [var for var in top_20_vars if
                 var.startswith("L") and var not in ['LAFmax', 'LAFmin', 'LCpeak', 'LCeq']]
    output_lines.append("\nFrequency Variables within Top 20 (by F-Score):")
    output_lines.append(str([clean_label(v, variable_name_map) for v in freq_vars]))
    output_lines.append(f"Number of top frequency variables: {len(freq_vars)}")
    if 'LAF_max_min_diff' in top_20_vars:
        output_lines.append(f"\n'{clean_label('LAF_max_min_diff', variable_name_map)}' is in top 20")
    if 'LZeq_25_2k_diff' in top_20_vars:
        output_lines.append(f"\n'{clean_label('LZeq_25_2k_diff', variable_name_map)}' is in top 20")

    # --- Generate and Save Importance Plot (Top 20, use F-Score, Clean Labels) ---
    if not importance_df_top20.empty and importance_df_top20['F-Score'].max() > 0:
        # logical operator precedence:
        # not: Evaluated first.
        # and: Evaluated after not.
        # or: Evaluated last (after not & and).

        plot_data = importance_df_top20.sort_values(by='F-Score', ascending=True)
        cleaned_labels = [clean_label(var, variable_name_map) for var in plot_data['Variable']]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(cleaned_labels, plot_data['F-Score'], color='skyblue')
        ax.set_xlabel("Feature Importance (F-Score)")
        ax.set_ylabel("Features")
        ax.set_title(f"Top 20 Feature Importance for {response.upper()} (F-Score)")
        fig.tight_layout()

        # Save plot directly to PNG file in the current directory
        plot_filename = f"xgboost_plot_{response}_top20_fscore.png"
        try:
            fig.savefig(plot_filename, format='png', bbox_inches='tight', dpi=300)
            print(f"\nTop 20 F-Score importance plot saved to '{plot_filename}'")
            output_lines.append(f"\nTop 20 F-Score importance plot saved to '{plot_filename}'")
        except Exception as e:
            print(f"\nError saving plot for {response.upper()}: {e}")
            output_lines.append(f"\nError saving plot for {response.upper()}: {e}")

        plt.close(fig)

    else:
        print(f"\nSkipping plot for {response.upper()}: No features with F-Score > 0 in top 20.")
        output_lines.append(f"\nSkipping plot for {response.upper()}: No features with F-Score > 0 in top 20.")

    # --- Save captured text output to a TXT file in the current directory ---
    report_text_filename = f"xgboost_report_{response}.txt"
    try:
    # Combine list of lines into one string,  each list element is separated by a new line
        full_report_text = "\n".join(output_lines)
        with open(report_text_filename, 'w', encoding='utf-8') as f:
            f.write(full_report_text) # Write the string to the new file
        print(f"Text report saved to '{report_text_filename}'")
    except Exception as e:
        print(f"\nError saving text report for {response.upper()}: {e}")

# --- End of Loop ---

print("\n\nScript finished processing all specified gas models.")