import unittest

class Test_Creation(unittest.TestCase):
    def test_model_creation(self):
        import neura
        model = None

        try:
            model = neura.model.Model()
        except:
            pass

        self.assertTrue(isinstance(model, neura.model.Model), "model not created correctly")
    
    def test_model_values_change(self):
        import neura
        
        model = neura.model.Model([
            neura.layers.Dense(16, activation=neura.activation.LeakyReLu(), input_shape=(16,)),
            neura.layers.Dense(8, activation=neura.activation.Sigmoid()),
            neura.layers.Dense(4, activation="tanh")
        ], name="Test Neural Network")

        self.assertEqual(model.name, "Test Neural Network", "unable to change model name")
    
    def test_model_layers_add(self):
        import neura
        
        model = neura.model.Model([
            neura.layers.Dense(16, input_shape=(16,)),
            neura.layers.Dense(8),
            neura.layers.Dense(4)
        ])

        self.assertEqual(len(model.layers), 3, "number of layers incorrect")
    
    def test_model_layers_add_2(self):
        import neura
        
        model = neura.model.Model([
            neura.layers.Dense(16, input_shape=(16,)),
            neura.layers.Dense(8),
        ])

        model.add_layer(neura.layers.Dense(4))

        self.assertEqual(len(model.layers), 3, "number of layers incorrect")




if __name__ == '__main__':
    unittest.main()
