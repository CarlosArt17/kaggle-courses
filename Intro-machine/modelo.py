import pandas as pd

melbourne_file_path = "melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

#1 - select prediction target y
y = melbourne_data.Price
print(y)

#2 - choose features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X)

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
print("training...")
melbourne_model.fit(X, y)
print("done")

print()
print(X.head())
print("The prediction are")
print(melbourne_model.)