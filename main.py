from nn import NeuralNetwork, Layer
import activation, random

NN = NeuralNetwork([
    Layer(16, activation=activation.LeakyReLu()),
    Layer(8, activation=activation.Sigmoid()),
    Layer(4, activation="tanh")
], name="Test Neural Network")


X = [random.randint(0, 100) for _ in range(10)]
prediction = NN.predict([random.randint(0, 100) for _ in range(10)] ,verbose=False)
print(prediction, "\n")

print(NN.summary())
