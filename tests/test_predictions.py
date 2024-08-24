import unittest
import numpy as np

class TestPredictions(unittest.TestCase):
    def test_input_shape(self):
        import neura
        input_shape = (16,)
        model = neura.model.Model([
            neura.layers.Dense(16, activation=neura.activation.LeakyReLu(), input_shape=input_shape),
            neura.layers.Dense(8, activation=neura.activation.Sigmoid()),
            neura.layers.Dense(4, activation="tanh")
        ], name="Test Neural Network")

        self.assertEqual(model.input_shape, input_shape, "incorrect model input shape {}".format(model.output_shape))

    def test_output_shape(self):
        import neura
        
        model = neura.model.Model([
            neura.layers.Dense(16, activation=neura.activation.LeakyReLu(), input_shape=(16,)),
            neura.layers.Dense(8, activation=neura.activation.Sigmoid()),
            neura.layers.Dense(4, activation="tanh")
        ], name="Test Neural Network")

        prediction = model.predict(np.random.rand(16), verbose=False)
        self.assertEqual(len(prediction), 4, f"incorrect output shape (output_size: {len(prediction)}, expected: 4)")

    def test_model_output_shape(self):
        import neura
        
        model = neura.model.Model([
            neura.layers.Dense(16, input_shape=(16,)),
            neura.layers.Dense(8),
        ])

        self.assertEqual(model.output_shape, (8,), "incorrect model output shape {}".format(model.output_shape))

    def test_layer_output_shape(self):
        import neura
        
        model = neura.model.Model()

        model.add_layer(neura.layers.Dense(16, input_shape=(16,)))
        self.assertEqual(model.output_shape, (16,), "incorrect layer output shape {}".format(model.output_shape))
        
        model.add_layer(neura.layers.Dense(8))
        self.assertEqual(model.output_shape, (8,), "incorrect model output shape {}".format(model.output_shape))

if __name__ == '__main__':
    unittest.main()