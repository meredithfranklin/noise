### Note. This XGBoost model is fitted with all variables excluded 'LAeq' and
### without addressing the chronological gap.
## TOP INSTRUCTION for readers: this file is using full variables without VIF greedy
## selection to regress on 7 gas (including new NO2 and SO2) and two difference
## variables and the new binary day-night time categorical variable
# (11am to 5 pm -> day) with a random grid search on XGB parameters to
# find an optimal fit.

import pandas as pd
import numpy as np  ## import for politeness lol
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
# CV website: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
from scipy.stats import uniform, randint  ## this is for random grid search CV input


# read in flare_well_sound_pollutant_met_data.csv
# this is the merged data at 1-minute intervals (sound, pollutants, and meteorology are 1-minute; flares daily, well production monthly)
# per instructions, focus on SELECT SET XGBoost for NOx and CH4

try:
    # Note, this is url is the absolute path in my machine.
    # You might want to change it into 'sound_pollutant_met_data.csv'
    df = pd.read_csv(r"C:\Users\franc\PycharmProjects\Summer_research\Codes\sound_pollutant_met_data.csv")
    print(df.head())
except FileNotFoundError:
    print("Error: 'sound_pollutant_met_data.csv' not found. Please make sure the file exists in the current directory or provide the correct path.")

df.rename(columns=lambda col: col.replace(" ", "_"), inplace=True)

# Output the updated column names
print(df.columns)

# subset from July 1st, 2023 through end of study May 31st, 2024
df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
df.dropna(subset=['time_utc'], inplace=True)

# --- Create Day/Night Binary Variable ---
# Define Day as 11:00:00 up to (but not including) 17:00:00 (5 PM)
# Assign 1 for Day, 0 for Night
# Note: If 5 PM should be inclusive, change '< 17' to '<= 17'
print("\nCreating 'day_night' variable (day means 1 = 11am to <5pm)...")
df['day_night'] = ((df['time_utc'].dt.hour >= 11) & (df['time_utc'].dt.hour < 17)).astype(int)

# Verify creation
print("Value counts for 'day_night':")
print(df['day_night'].value_counts(normalize=True))


df_subset = df[(df['time_utc'] >= '2023-07-01') & (df['time_utc'] <= '2024-05-31')].copy()

# variables to predict, calculate means for table
for var in ['nox', 'ch4', 'co', 'h2s', 'co2_ppm', 'no2', 'so2']:
    mean_val = df_subset[var].mean()
    print(f"Mean {var.upper():<7}: {mean_val:.3f}")

print(df_subset.shape)

# we now insert two difference frequency variables
# 1: LAFmax - LAFmin (these two frequency variables are highly important variables
# across all models and gas types.)
# 2: LZeq_25Hz - LZeq_2kHz (this is just a try.)

df_subset['LAF_max_min_diff'] = df_subset['LAFmax'] - df_subset['LAFmin']
df_subset['LZeq_25_2k_diff'] = df_subset['LZeq_25Hz'] - df_subset['LZeq_2kHz']


## use full variables except the 'LAeq'

vars = ["LCpeak", "LCeq","LAFmax","LAFmin", 'LAF_max_min_diff', 'LZeq_25_2k_diff',
    "LZeq_12.5Hz", "LZeq_16Hz", "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz",
    "LZeq_50Hz", "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz",
    "LZeq_160Hz", "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz",
    "LZeq_500Hz", "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz",
    "LZeq_1.6kHz", "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz",
    "LZeq_5kHz", "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz",
    "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr", "day_night"]

for response in ['nox', 'ch4', 'h2s', 'co', 'co2_ppm', 'no2', 'so2']:
    ## Here, we are additionally modelling two more gas: NO2 and SO2

    print(f"\n========================= {response.upper()} MODEL =========================")

    # Drop rows with NA in predictors or target
    df_model = df_subset[vars + [response]].dropna()

    ## use full variables except the 'LAeq'
    X = df_model[vars]
    selected_predictors = X.columns.tolist()

    # Train/valid/test split
    train_data, temp_data = train_test_split(df_model, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Report the number of observations per model for each gas type
    print(f"Observations for {response.upper()}:")
    print(f"  Train: {len(train_data)}")
    print(f"  Valid: {len(valid_data)}")
    print(f"  Test : {len(test_data)}")
    print(f"  Total: {len(df_model)}")

    # define X and Y sets for random grid search CV fitting
    X_train = train_data[selected_predictors]
    y_train = train_data[response]
    X_valid = valid_data[selected_predictors]
    y_valid = valid_data[response]
    X_test = test_data[selected_predictors]
    y_test = test_data[response]

    # Using random grid search to find the optimal XGBoost model
    # link for xgb parameters: https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html
    param_grid = {
        'n_estimators': randint(300, 1201),  # num_boost_round
        'max_depth': randint(3, 13),   # range: [0,∞], default=6
        'eta': uniform(0.01, 0.19)    # aka learning_rate, float, range: [0,1], default=0.3
    }

    # Define the base XGBoost model using the scikit-learn API
    xgb_model = xgb.XGBRegressor(
        objective = "reg:squarederror",
        eval_metric = "rmse",
        random_state = 42,
        tree_method = 'hist',  # Use histogram-based method for better memory efficiency
        n_jobs = 1
    )

    # begin the random grid search CV for tuning the cb model.
    random_search = RandomizedSearchCV(
        estimator = xgb_model,
        param_distributions = param_grid,
        n_iter=20,  # Number of random samples of parameter grid to try, we can try 50 or even 100.
        scoring='neg_root_mean_squared_error',  # Metric for CV evaluation
        cv=5,  # Number of cross-validation folds
        verbose=1,  # Show progress
        random_state=42,
        n_jobs=1
    )

    print("\nStarting Randomized Search CV for hyperparameter tuning...")
    random_search.fit(X_train, y_train)

    # Initialize the CatBoost model with the best parameters
    best_params = random_search.best_params_

    # Train the best model

    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,
        tree_method='hist',
        **best_params  # use the best tuned parameters
    )

    print("\nTraining final model with best parameters and early stopping...")
    # Train the final model using early stopping based on the validation set
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False  # Set to True to see progress if needed
    )
    print("Final model training complete.")

    # Predict
    valid_preds = final_model.predict(X_valid)
    test_preds = final_model.predict(X_test)

    # Evaluate
    valid_r2 = r2_score(y_valid, valid_preds)
    valid_rmse = root_mean_squared_error(y_valid, valid_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)

    print(f"\n Model Performance:")
    print(f"Validation R²:   {valid_r2:.3f}")
    print(f"Validation RMSE: {valid_rmse:.3f}")
    print(f"Test R²:         {test_r2:.3f}")
    print(f"Test RMSE:       {test_rmse:.3f}")

    # Variable Importance
    feature_importance_values = final_model.feature_importances_
    # Create a DataFrame
    importance_df = pd.DataFrame({
        'Variable': X_train.columns,  # Get feature names from training data
        'Importance': feature_importance_values
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n Top 20 Variable Importances:")
    print(importance_df.head(20).to_string(index=False))

    ## Here, we subtract the Frequency variables only (starting with L)
    # from the top 20 variables, but we exclude 'LAeq'.
    top_20_vars = importance_df.head(20)['Variable'].to_list()
    freq_vars = []
    for var in top_20_vars:
        if var.startswith("L") and var != 'LAeq':
            freq_vars.append(var)

    print("\n Frequency Variables within Top 20 important variables:")
    print(freq_vars)

    print('\n The number of top frequency variables is :')
    print(len(freq_vars))

    if 'LAF_max_min_diff' in top_20_vars:
        print(" 'LAF_max_min_diff' is in top 20")
    if 'LZeq_25_2k_diff' in top_20_vars:
        print(" 'LZeq_25_2k_diff' is in top 20")

    # Plotting top important variables
    plt.figure(figsize=(10, 8))
    # Select top 20 features for plotting, sort ascending for barh
    plot_df = importance_df.head(20).sort_values(by='Importance', ascending=True)
    plt.barh(plot_df['Variable'], plot_df['Importance'], color='green')  # Use plt.barh
    plt.xlabel("Feature Importance (XGBoost Default)")
    plt.ylabel("Features")
    plt.title(f"Top 20 Feature Importance for {response.upper()} (XGBoost)")
    plt.tight_layout()
    plt.show()