import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

# 1. Load data
df = pd.read_csv("satellite_config_scores.csv")

# 2. Normalize column names
df.columns = df.columns.str.strip().str.lower()

# 3. Identify target and features
if 'score' not in df.columns:
    raise KeyError("Could not find a 'score' column in your CSV.")
y = df['score']
X = df.drop(columns=['score']).select_dtypes(include=['number'])

# 4. Experimental settings
sample_sizes = [350, 3500, 35000]
results = []

# 5. Loop over sample sizes
for n_samples in sample_sizes:
    print(f"\n🔵 Sample size: {n_samples}")

    # Uniform random sampling
    df_sampled = df.sample(n=n_samples, random_state=42)
    X_sampled = df_sampled.drop(columns=['score']).select_dtypes(include=['number'])
    y_sampled = df_sampled['score']

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42
    )

    # Train a simple DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    rho, _ = spearmanr(y_test, y_pred)

    # Store results
    results.append({
        'Sample Size': n_samples,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Spearman_rho': rho
    })

    # Print for quick inspection
    print(f"  📈 MSE: {mse:.4f}")
    print(f"  📈 RMSE: {rmse:.4f}")
    print(f"  📈 R²: {r2:.4f}")
    print(f"  📈 Spearman ρ: {rho:.4f}")

# 6. Convert to DataFrame for easy export or table generation
results_df = pd.DataFrame(results)
print("\nFinal Results Table:")
print(results_df)

# 7. Optional: save results to CSV
# results_df.to_csv("preliminary_results.csv", index=False)

# 8. Optional: plot R2 vs Sample Size
plt.figure(figsize=(6, 4))
plt.plot(results_df['Sample Size'], results_df['R2'], marker='o')
plt.xlabel('Sample Size')
plt.ylabel('$R^2$ Score')
plt.title('Prediction Performance vs Sample Size')
plt.grid(True)
plt.tight_layout()
plt.show()
