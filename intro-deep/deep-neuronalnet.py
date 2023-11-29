import pandas as pd

concrete = pd.read_csv('concrete.csv')
concrete.head()

#Input shape
input_shape = [8]
print(input_shape)

#Define a Model with Hidden Layers
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
print(model)

#Activation Layers
model = keras.Sequential([
    
    layers.Dense(units=32, input_shape=[8]), 
    layers.Activation('relu'),
    layers.Dense(units=32), 
    layers.Activation('relu'),
    layers.Dense(1),
])
print(model)

#Alternatives to ReLU
activation_layer = layers.Activation('relu')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
