import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8],
    'Age': [22, 25, 28, 30, 32, 35, 38, 40],
    'Salary': [20000, 25000, 30000, 35000,
               40000, 45000, 50000, 55000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Experience', 'Age']]
y = df['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Predicted Salary:", y_pred)
print("Actual Salary:", list(y_test))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))