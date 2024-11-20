import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from ucimlrepo import fetch_ucirepo
import scienceplots

# Set plot style
plt.style.use(['science', 'ieee'])

# Fetch dataset
concrete_compressive_strength = fetch_ucirepo(id=165)
X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

# Decision Tree Regression
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train_scaled, y_train)
dt_predictions = dt_model.predict(X_test_scaled)

# Metrics function
def print_metrics(y_true, y_pred, model_name):
    print(f"{model_name} Metrics:")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_true, y_pred):.4f}")
    print(f"R-squared Score: {r2_score(y_true, y_pred):.4f}\n")

# Print metrics for both models
print_metrics(y_test, lr_predictions, "Linear Regression")
print_metrics(y_test, dt_predictions, "Decision Tree Regression")

# Create scatter plots of predicted vs actual values
plt.figure()

# Linear Regression Scatter Plot

plt.scatter(y_test, lr_predictions, alpha=0.3, s=2.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Linear Regression: Predicted vs Actual')
plt.xlabel('Actual Concrete Strength')
plt.ylabel('Predicted Concrete Strength')
plt.show()

# Decision Tree Regression Scatter Plot
plt.figure()
plt.scatter(y_test, dt_predictions, alpha=0.3, s=2.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Decision Tree Regression: Predicted vs Actual')
plt.xlabel('Actual Concrete Strength')
plt.ylabel('Predicted Concrete Strength')

plt.tight_layout()
plt.show()

# Feature importance for Decision Tree
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure()
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Decision Tree Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Print feature importance
print("Decision Tree Feature Importance:")
print(feature_importance)