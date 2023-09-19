import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Selecting Data for Modeling
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)

#Selecting The Prediction Target
print(y = melbourne_data.Price)

#Choosing "Features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())

#Building Your Model
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

#Ejercicios
# Path of the file to read
iowa_file_path = 'train.csv'

home_data = pd.read_csv(iowa_file_path)

#Ejercicio 1
y = home_data.SalePrice
print(y)

#Ejercicio 2
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
                "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X=home_data[feature_names]
print(X)

#Ejercicio 3
iowa_model = DecisionTreeRegressor(random_state=1)
print(iowa_model.fit(X, y))

#Ejercicio 4
predictions = iowa_model.predict(X)
print(predictions)