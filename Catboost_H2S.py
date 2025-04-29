### Note. This CatBoost model is fitted with all variables excluded 'LAeq' and
### without addressing the chronological gap.

import pandas as pd
import numpy as np  ## import master numpy for politeness lol
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import catboost
from catboost import CatBoostRegressor, Pool
# Catboost website: https://catboost.ai/
from sklearn.model_selection import RandomizedSearchCV
# CV website: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
from scipy.stats import uniform, randint  ## this is for random grid search CV input

# read in flare_well_sound_pollutant_met_data.csv
# this is the merged data at 1-minute intervals (sound, pollutants, and meteorology are 1-minute; flares daily, well production monthly)

## Here, we are not doing any variable selection, and we use full variables.
## We are using Catboost to regress on H2S only for testing purposes.

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
df_subset = df[(df['time_utc'] >= '2023-07-01') & (df['time_utc'] <= '2024-05-31')]

# variables to predict, calculate means for table
for var in ['h2s']:
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
    "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr"]

for response in ['h2s']:
    ## Here, we are just using H2S as response for testing purpose

    print(f"\n========================= {response.upper()} MODEL =========================")

    # Drop rows with NA in predictors or target
    df_model = df_subset[vars + [response]].dropna()

    # Standardize predictors for VIF calculation
    X = df_model[vars]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=vars)

    ## use full variables except the 'LAeq'
    selected_predictors = X_scaled.columns.tolist()

    # Final model dataset
    df_model = df_model[selected_predictors + [response]].dropna()

    # Train/valid/test split
    train_data, temp_data = train_test_split(df_model, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Report the number of observations per model for each gas type
    print(f"Observations for {response.upper()}:")
    print(f"  Train: {len(train_data)}")
    print(f"  Valid: {len(valid_data)}")
    print(f"  Test : {len(test_data)}")
    print(f"  Total: {len(df_model)}")

    # Create Pool Matrices for Catboost
    dtrain = Pool(data = train_data[selected_predictors], label = train_data[response])
    dvalid = Pool(data = valid_data[selected_predictors], label = valid_data[response])
    dtest = Pool(data = test_data[selected_predictors], label = test_data[response])

    # define X and Y sets for random grid search CV fitting
    X_train = train_data[selected_predictors]
    y_train = train_data[response]

    # Train model

    # Define the parameter grid for the random grid search
    param_grid = {
        'iterations': randint(900, 1101),   # int, default = 1000
        # randint(low, high) gives range [low, high-1]

        'depth': randint(5, 8),     # int, dafault = 6
        'learning_rate': uniform(loc = 0.01, scale = 0.09),  # float, default 0.03
        # uniform(loc, scale) gives range [loc, loc + scale)
        'l2_leaf_reg': uniform(loc = 2.5, scale = 1.0),    # float, default = 3.0
        'border_count': randint(30, 256) # int, default = 254.
        # The number of splits for numerical features.
        # Allowed values are integers from 1 to 65535 inclusively.
    }

    # define the catboost model
    model = CatBoostRegressor(
        loss_function = 'RMSE',
        random_seed = 42,
        verbose = 0,
        eval_metric = 'RMSE'
    )

    # begin the random grid search CV for tuning the cb model.
    random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = 30,  # Number of random samples of parameter grid to try, we can try 50 or even 100.
        scoring = 'neg_root_mean_squared_error',  # Metric for CV evaluation
        cv = 5,  # Number of cross-validation folds
        verbose=1,  # Show progress
        random_state=42,  # Reproducibility
        n_jobs=-1  # parallel processing
    )

    print("\nStarting Randomized Search CV for hyperparameter tuning...")
    random_search.fit(X_train, y_train)

    # Initialize the CatBoost model with the best parameters
    best_params = random_search.best_params_
    final_model = CatBoostRegressor(**best_params, loss_function = 'RMSE',
                              verbose = 100, eval_metric = 'RMSE',
                              early_stopping_rounds = 50)
    # verbose = 100: message every 100 iterations
    # add early stopping to prevent overfitting

    print("\nTraining final model with best parameters...")
    # Train the model using the training Pool and evaluate on the validation Pool
    final_model.fit(dtrain, eval_set=dvalid, verbose=100)  # Use Pool objects, eval_set, adjusted verbose

    # Predict
    valid_preds = final_model.predict(dvalid)
    test_preds = final_model.predict(dtest)

    # Evaluate the R square and RMSE for validation set and test set on catboost.
    valid_r2 = r2_score(valid_data[response], valid_preds)
    valid_rmse = root_mean_squared_error(valid_data[response], valid_preds)
    test_r2 = r2_score(test_data[response], test_preds)
    test_rmse = root_mean_squared_error(test_data[response], test_preds)

    print(f"\n Model Performance:")
    print(f"Validation R²:   {valid_r2:.3f}")
    print(f"Validation RMSE: {valid_rmse:.3f}")
    print(f"Test R²:         {test_r2:.3f}")
    print(f"Test RMSE:       {test_rmse:.3f}")

    # --- Variable Importance (CatBoost Method) ---
    # Get feature importance scores from the trained model
    # The default importance type is 'FeatureImportance' (prediction value change)
    feature_importance_values = final_model.get_feature_importance()
    # default, type = 'FeatureImportance' for get_feature_importance()
    # It returns a list of length [n_features] with float feature
    # importances values for each feature.

    # Get feature names (should match the order of columns in X_train/dtrain)
    feature_names = X_train.columns  # Or use final_model.feature_names_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Variable': feature_names,
        'Importance': feature_importance_values
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\nTop 20 Variable Importances (CatBoost):")
    print(importance_df.head(20).to_string(index=False))

    # --- Analyze Top Variables ---
    # Extract top 20 variable names
    top_20_vars = importance_df.head(20)['Variable'].tolist()

    # Identify frequency variables (starting with 'L') within the top 20
    # Exclude 'LAeq' if it were present (it's not in your 'predictors' list)
    freq_vars_in_top20 = [var for var in top_20_vars if
                          var.startswith("L")]

    print("\nFrequency Variables within Top 20 important variables:")
    print(freq_vars_in_top20)
    print(f'\nNumber of top frequency variables: {len(freq_vars_in_top20)}')

    # Check if the engineered features are in the top 20
    if 'LAF_max_min_diff' in top_20_vars:
        print("\n'LAF_max_min_diff' is in the top 20 important variables.")
    else:
        print("\n'LAF_max_min_diff' is NOT in the top 20 important variables.")

    if 'LZeq_25_2k_diff' in top_20_vars:
        print("'LZeq_25_2k_diff' is in the top 20 important variables.")
    else:
        print("'LZeq_25_2k_diff' is NOT in the top 20 important variables.")

    # --- Plotting Feature Importance (Using Matplotlib) ---
    plt.figure(figsize=(10, 8))
    # Select top 20 features for plotting
    plot_df = importance_df.head(20).sort_values(by='Importance',
                                                 ascending=True)  # Sort ascending for horizontal bar plot

    plt.barh(plot_df['Variable'], plot_df['Importance'], color='skyblue')  # Use barh for horizontal plot
    plt.xlabel("Feature Importance (CatBoost Prediction Value Change)")
    plt.ylabel("Features")
    plt.title(f"Top 20 Feature Importance for {response.upper()} (CatBoost)")
    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    plt.show()