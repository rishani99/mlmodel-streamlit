import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('boston_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as boston_model.pkl (trained on California housing dataset)")
