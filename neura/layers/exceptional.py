"""
A collection of helper classes to specify what a layer is
how it behaves and how it should(n't) be used

example:

    # this is a class that should NOT be used as a first layer
    class MyLayer(Layer, exceptional.NotFirstLayer):
        ...

also more helper classes can be used at the same time:   

    # this is a class that should NOT be used as a first and last layer
    class MyLayer(Layer, exceptional.NotFirstLayer, exceptional.NotLastLayer):
        ...

"""

class ExcetionalLayer:
    """base class for all exceptional layers"""
    errmsg: str = "this is an exceptional layer"

class NotFirstLayer():
    """A layer that should not be used as input layer"""
    errmsg: str = "this layer should not be used as input layer"

class NotLastLayer():
    """A layer that should not be used as output layer"""
    
    errmsg: str = "this layer should not be used as input layer"
    def __init__(self) -> None:
        if (_last_layer := getattr(self, "_last_layer")) and _last_layer == True:
            raise RuntimeError(NotLastLayer.errmsg)

