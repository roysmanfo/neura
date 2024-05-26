import unittest

class TestPredictions(unittest.TestCase):
    def test_output_shape(self):
        import nnfs
        import random
        
        NN = nnfs.model.Model([
            nnfs.layers.Dense(16, activation=nnfs.activation.LeakyReLu(), input_shape=(16,)),
            nnfs.layers.Dense(8, activation=nnfs.activation.Sigmoid()),
            nnfs.layers.Dense(4, activation="tanh")
        ], name="Test Neural Network")

        prediction = NN.predict([random.randint(0, 100) for _ in range(16)], verbose=False)
        self.assertEqual(len(prediction), 4, f"incorrect output shape (output_size: {len(prediction)}, expected: 4)")


if __name__ == '__main__':
    unittest.main()