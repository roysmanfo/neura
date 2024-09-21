# Neural Network Creation Library (Neura)

Create simple neural network using a lightweight library

Example of usage (create a simple network)
```py
import neura
import numpy as np

# generate a sample (mimic the mnist digits database)
# 28x28 pixel image of a digit
input_shape = (28, 28)

image = np.random.rand(28, 28)

# create the correct output (y_true) 
y = np.zeros(10)
y[int(np.random.rand() * 10)] = 1

# create the model architecture
model = neura.model.Model([
    neura.layers.Flatten(input_shape=input_shape),
    neura.layers.Dense(128, activation=neura.activation.ReLu()),
    neura.layers.Dense(10)
], name="MyModel", learning_rate=0.001)

# compile the model with loss function and optimizer
model.compile(
    loss=neura.losses.CategoricalCrossEntropy(),
    optimizer="adam"
)

# print the model architecture
model.summary()

# try to categorize the image in one of the 10 classes [0, 1, 2, ..., 9]
pred = model.predict(image, verbose=False)

# evaluate the model prediction without training
loss = model.evaluate(image, y)

# print the results
print("\nresults:")
print(pred)
print(y)
print(loss)

```

```
MyModel
===============================================
Total number of parameters: 102416
===============================================

LAYERS:

0. Flatten              nodes: 1        params: 784
1. Dense                nodes: 128      params: 100352
2. Dense                nodes: 10       params: 1280

===============================================

results:
[  55.0807834     8.48622231    7.46788936    6.80623906   -7.329365
 -120.89559978   41.64848654  -64.82835674   29.99597943  -49.55296296]
[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
[9.992007221626415e-17]
```

## Installation
1. clone this repository
```
git clone https://github.com/roysmanfo/neura 
```
2. install using pip
```
pip install .
```

