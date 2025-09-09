import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pdb
# Generate some sample data with outliers
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]

# Initialize and train the Isolation Forest model
# n_estimators: number of trees in the forest
# contamination: proportion of outliers in the data
# random_state: for reproducibility
model = IsolationForest(n_estimators=256, contamination=0.1, random_state=42)
model.fit(X)

# Predict anomaly scores and labels (-1 for outliers, 1 for inliers)
y_pred = model.predict(X)
scores_pred = model.decision_function(X)

pdb.set_trace()
# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', s=50, edgecolors='k')
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Anomaly Label (-1: Outlier, 1: Inlier)")
plt.show()

# Print the number of detected outliers
print(f"Number of detected outliers: {np.sum(y_pred == -1)}")