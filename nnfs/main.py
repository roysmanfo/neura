import nnfs
import random

NN = nnfs.model.Model([
    nnfs.layers.Dense(16, activation=nnfs.activation.LeakyReLu()),
    nnfs.layers.Dense(8, activation=nnfs.activation.Sigmoid()),
    nnfs.layers.Dense(4, activation="tanh")
], name="Test Neural Network")


X = [random.randint(0, 100) for _ in range(10)]
prediction = NN.predict([random.randint(0, 100) for _ in range(10)], verbose=False)
print(prediction, "\n")

print(NN.summary())
