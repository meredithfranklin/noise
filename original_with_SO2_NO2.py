import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# read in flare_well_sound_pollutant_met_data.csv
# this is the merged data at 1-minute intervals (sound, pollutants, and meteorology are 1-minute; flares daily, well production monthly)
# per instructions, focus on SELECT SET XGBoost for NOx and CH4
## Here, we are additionally modelling two more gas: NO2 and SO2

try:
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
for var in ['nox', 'ch4', 'co', 'h2s', 'co2_ppm']:
    mean_val = df_subset[var].mean()
    print(f"Mean {var.upper():<7}: {mean_val:.3f}")

# Compute correlation matrix of all variables (sound, meteorology, pollutants)

vars = ["LAFmax", "LAFmin", "LCpeak", "LCeq", "LAeq",
        "LZeq_12.5Hz", "LZeq_16Hz", "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz",
        "LZeq_50Hz", "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz",
        "LZeq_160Hz", "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz",
        "LZeq_500Hz", "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz",
        "LZeq_1.6kHz", "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz",
        "LZeq_5kHz", "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz",
        "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr",
        "nox", "ch4", "co", "co2_ppm", "h2s"]

df_corr = df_subset[vars].copy()

corr_matrix = df_corr.corr(numeric_only=True)

# create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# plot heatmap with mask
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix,
            mask=mask,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 6})
plt.title("Correlation Heatmap: Acoustic Variables (Lower Triangle)", fontsize=16)
plt.tight_layout()
plt.show()

# create a mask for upper triangle and show correlations >0.9
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) | (corr_matrix.abs() <= 0.9) | (corr_matrix == 1.0)

# plot heatmap of highly correlated variables
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix,
            mask=mask,
            cmap="Reds",
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 6})
plt.title("Lower Triangle Correlation Heatmap (|r| > 0.9)", fontsize=16)
plt.tight_layout()
plt.show()

# Predictor selection using vars list
# Greedy backward elimination via VIF (remove highly correlated data that causes variance inflation)
# XGBoost training with train/validation/test split
# Evaluation using validation R-squared & RMSE
# Variable importance table and plot, top 20

# Predictor list
vars = ["LCpeak", "LCeq","LAFmax","LAFmin", "LAeq",
    "LZeq_12.5Hz", "LZeq_16Hz", "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz",
    "LZeq_50Hz", "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz",
    "LZeq_160Hz", "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz",
    "LZeq_500Hz", "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz",
    "LZeq_1.6kHz", "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz",
    "LZeq_5kHz", "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz",
    "temp_f", "wsp_ms", "wdr_deg", "relh_percent", "pressure_altcorr"]

def calculate_vif(X):
    """
    Compute Variance Inflation Factor (VIF) for each predictor in X.
    Returns a DataFrame with variable names and their corresponding VIF values.
    VIF(i) = 1/(1- R**2(i)), where R**2(i) is the R_squared value from regressing
    the i-th feature against all the other features.

    :param X: the input dataset what we want to calculate VIF for each variable within.
    :return: a Pandas DataFrame with its first column to be the name of variable,
    and the second column to be the VIF for that variable.
    """
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


for response in ['nox', 'ch4', 'h2s', 'co', 'co2_ppm', 'no2', 'so2']:
    ## Here, we are additionally modelling two more gas: NO2 and SO2

    print(f"\n========================= {response.upper()} MODEL =========================")

    # Drop rows with NA in predictors or target
    df_model = df_subset[vars + [response]].dropna()

    # Standardize predictors for VIF calculation
    X = df_model[vars]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=vars)

    # Calculate VIF for 20 threshold
    vif_data = calculate_vif(X_scaled)
    while vif_data["VIF"].max() > 20:
        max_vif_var = vif_data.sort_values("VIF", ascending=False).iloc[0]["Variable"]
        print(f"Dropping '{max_vif_var}' with VIF: {vif_data['VIF'].max():.2f}")
        X_scaled = X_scaled.drop(columns=[max_vif_var])
        vif_data = calculate_vif(X_scaled)

    selected_predictors = X_scaled.columns.tolist()
    print(f"\n Final selected predictors ({len(selected_predictors)}):")
    print(selected_predictors)

    # Final model dataset
    df_model = df_model[selected_predictors + [response]].dropna()

    # Train/valid/test split
    train_data, temp_data = train_test_split(df_model, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

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
    ## here, we only extract two columns from the importance "Variable and F_score"
    ## F_score = the number of times a feature appears in all the trees.
    importance_df = importance_df.sort_values(by='F_score', ascending=False)

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

    print("\n The number of top frequency variables is:")
    print(len(freq_vars))

    # Plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='weight', max_num_features=20, height=0.6)
    plt.title(f"Top 20 Feature Importances for {response.upper()} (F Score)")
    plt.tight_layout()
    plt.show()

