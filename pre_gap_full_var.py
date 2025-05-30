### Note. This XGBoost model is fitted with all variables excluded 'LAeq' and
### for the data before the gap.

import pandas as pd
import numpy as np  ## import for politeness lol
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# read in flare_well_sound_pollutant_met_data.csv
# this is the merged data at 1-minute intervals (sound, pollutants, and meteorology are 1-minute; flares daily, well production monthly)
# per instructions, focus on SELECT SET XGBoost for NOx and CH4

try:
    # Read only the first 253605 rows (pre-gap data)
    ## The data is sliced for time between 2023-07-01 and "2023-11-13 17:25:00 UTC"
    df_pre = pd.read_csv(r"C:\Users\franc\PycharmProjects\Summer_research\Codes\sound_pollutant_met_data.csv", nrows=253605)
    print(df_pre.head())
except FileNotFoundError:
    print("Error: 'sound_pollutant_met_data.csv' not found. Please make sure the file exists in the current directory or provide the correct path.")

# Step 2: Clean column names
df_pre.rename(columns=lambda col: col.replace(" ", "_"), inplace=True)

# Step 3: Convert time_utc to datetime
df_pre['time_utc'] = pd.to_datetime(df_pre['time_utc'], errors='coerce')
print("Shape of the data before gap:", df_pre.shape)
print(df_pre.shape)

# subset from July 1st, 2023 to Nov 13th, 2023 (before the gap)
df_subset = df_pre[(df_pre['time_utc'] >= '2023-07-01')]


# variables to predict, calculate means for table
for var in ['nox', 'ch4', 'co', 'h2s', 'co2_ppm']:
    mean_val = df_subset[var].mean()
    print(f"Mean {var.upper():<7}: {mean_val:.3f}")

print(df_subset.shape)

## use full variables except the 'LAeq'

vars = ["LCpeak", "LCeq","LAFmax","LAFmin",
    "LZeq_12.5Hz", "LZeq_16Hz", "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz",
    "LZeq_50Hz", "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz",
    "LZeq_160Hz", "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz",
    "LZeq_500Hz", "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz",
    "LZeq_1.6kHz", "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz",
    "LZeq_5kHz", "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz",
    "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr"]

for response in ['nox', 'ch4', 'h2s', 'co', 'co2_ppm']:
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

    # Create DMatrices
    dtrain = xgb.DMatrix(train_data[selected_predictors], label=train_data[response])
    dvalid = xgb.DMatrix(valid_data[selected_predictors], label=valid_data[response])
    dtest = xgb.DMatrix(test_data[selected_predictors], label=test_data[response])

    # Train model
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.1,
        "eval_metric": "rmse",
        "seed": 42
    }
    model = xgb.train(params, dtrain, num_boost_round=1000)

    # Predict
    valid_preds = model.predict(dvalid)
    test_preds = model.predict(dtest)

    # Evaluate
    valid_r2 = r2_score(valid_data[response], valid_preds)
    valid_rmse = root_mean_squared_error(valid_data[response], valid_preds)
    test_r2 = r2_score(test_data[response], test_preds)
    test_rmse = root_mean_squared_error(test_data[response], test_preds)

    print(f"\n Model Performance:")
    print(f"Validation R²:   {valid_r2:.3f}")
    print(f"Validation RMSE: {valid_rmse:.3f}")
    print(f"Test R²:         {test_r2:.3f}")
    print(f"Test RMSE:       {test_rmse:.3f}")

    # Variable Importance
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame(list(importance.items()), columns=['Variable', 'F_score'])
    importance_df = importance_df.sort_values(by='F_score', ascending=False)

    print("\n Top 20 Variable Importances:")
    print(importance_df.head(20).to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='weight', max_num_features=20, height=0.6)
    plt.title(f"Top 20 Feature Importances for {response.upper()} (F Score)")
    plt.tight_layout()
    plt.show()