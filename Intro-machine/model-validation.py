#What is Model Validation
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

#Coding It
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#Ejercicios
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

#Ejercicio 1
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
print(train_X, val_X, train_y, val_y )

#Ejercicio 2
iowa_model = DecisionTreeRegressor(random_state=1)
print(iowa_model.fit(train_X, train_y))

#Ejercicio 3
val_predictions = iowa_model.predict(val_X)
print(val_predictions)

#Ejercicio 4
from sklearn.metrics import mean_absolute_error
val_mae = val_mae = mean_absolute_error(val_y, val_predictions)

#uncomment following line to see the validation_mae
print(val_mae)
