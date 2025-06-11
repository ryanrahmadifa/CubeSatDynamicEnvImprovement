import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau

# ==== PARSE ARGS ====
parser = argparse.ArgumentParser(description='Predict satellite scores with different models.')
parser.add_argument('--model', type=str, choices=['tree', 'forest', 'linear'], required=True)
parser.add_argument('--dataset', type=str, choices=['low', 'mid', 'high'], required=True)
parser.add_argument('--target', type=str, choices=['score_4orbits', 'score_8orbits', 'score_15orbits'], required=True)
args = parser.parse_args()

MODEL_TYPE = args.model
DATASET_SIZE = args.dataset
TARGET_SCORE = args.target

# ==== SETTINGS ====
CV_SPLITS = 5
N_SEEDS = 10
TEST_SIZE = 0.2
# ===================

# ==== LOAD DATA ====
df = pd.read_csv('./outputs/satellite_config_scores_multi.csv')
df.columns = df.columns.str.strip().str.lower()

X = df.drop(columns=['score_4orbits', 'score_8orbits', 'score_15orbits'])
X = X.select_dtypes(include=[np.number])
y = df[TARGET_SCORE]

sample_sizes = {
    'low': 35,
    'mid': 350,
    'high': 3500
}

# ==== METRICS STORAGE ====
mses = []
rmses = []
r2s = []
rhos = []
kendalls = []
rhos_15 = []

# ==== MULTI-SEED LOOP ====
for seed in range(N_SEEDS):
    np.random.seed(seed)

    # Fix sample size if needed
    N = min(sample_sizes[DATASET_SIZE], len(X))
    X_sampled = X.sample(n=N, random_state=seed)
    y_sampled = y.loc[X_sampled.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=TEST_SIZE, random_state=seed
    )

    if MODEL_TYPE == 'tree':
        model = DecisionTreeRegressor(random_state=seed)
    elif MODEL_TYPE == 'forest':
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
    elif MODEL_TYPE == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("Invalid model type")

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    y_pred_oof = cross_val_predict(model, X_sampled, y_sampled, cv=cv)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    rho, _ = spearmanr(y_test, y_pred)
    tau, _ = kendalltau(y_test, y_pred)

    # Ground truth Spearman with 15 orbits
    y_true_15 = df.loc[X_test.index]['score_15orbits']
    rho_15, _ = spearmanr(y_true_15, y_pred)

    mses.append(mse)
    rmses.append(rmse)
    r2s.append(r2)
    rhos.append(rho)
    kendalls.append(tau)
    rhos_15.append(rho_15)

# ==== RESULTS ====

def mean_std(arr):
    return np.mean(arr), np.std(arr)

mse_mean, mse_std = mean_std(mses)
rmse_mean, rmse_std = mean_std(rmses)
r2_mean, r2_std = mean_std(r2s)
rho_mean, rho_std = mean_std(rhos)
kendall_mean, kendall_std = mean_std(kendalls)
rho15_mean, rho15_std = mean_std(rhos_15)

print("\n========= RESULTS =========")
print(f"🌟 MSE: {mse_mean:.4f} ± {mse_std:.4f}")
print(f"🌟 RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"📈 R²: {r2_mean:.4f} ± {r2_std:.4f}")
print(f"🏆 Spearman ρ (vs 15 orbits ground truth): {rho15_mean:.4f} ± {rho15_std:.4f}")
print("============================")
