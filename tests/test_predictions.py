import unittest
import numpy as np

class TestPredictions(unittest.TestCase):
    def test_output_shape(self):
        import neura
        
        NN = neura.model.Model([
            neura.layers.Dense(16, activation=neura.activation.LeakyReLu(), input_shape=(16,)),
            neura.layers.Dense(8, activation=neura.activation.Sigmoid()),
            neura.layers.Dense(4, activation="tanh")
        ], name="Test Neural Network")

        prediction = NN.predict(np.random.rand(16), verbose=False)
        self.assertEqual(len(prediction), 4, f"incorrect output shape (output_size: {len(prediction)}, expected: 4)")


if __name__ == '__main__':
    unittest.main()