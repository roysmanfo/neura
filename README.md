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
print(pred)
print(y)
print(loss)

```