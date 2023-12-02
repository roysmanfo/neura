import unittest

class Test_Creation(unittest.TestCase):
    def test_model_creation(self):
        import nnfs
        model = None

        try:
            model = nnfs.model.Model()
        except:
            pass

        self.assertTrue(isinstance(model, nnfs.model.Model), "model not created correctly")
    
    def test_model_values_change(self):
        import nnfs
        
        model = nnfs.model.Model([
            nnfs.layers.Dense(16, activation=nnfs.activation.LeakyReLu()),
            nnfs.layers.Dense(8, activation=nnfs.activation.Sigmoid()),
            nnfs.layers.Dense(4, activation="tanh")
        ], name="Test Neural Network")

        self.assertEqual(model.name, "Test Neural Network", "unable to change model name")
    
    def test_model_layers_add(self):
        import nnfs
        
        model = nnfs.model.Model([
            nnfs.layers.Dense(16),
            nnfs.layers.Dense(8),
            nnfs.layers.Dense(4)
        ])

        self.assertEqual(len(model.layers), 3, "number of layers incorrect")




if __name__ == '__main__':
    unittest.main()
