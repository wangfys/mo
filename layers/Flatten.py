import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    """
    This is the flatten layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array((self.inShapes[0][0], 1))
        for d in self.inShapes[0][1:]:
            self.outShape[1] *= d
    
    def forward(self, feedInput):
        BaseLayer.forward(self, feedInput)
        inputTensor = np.array(self.inNodes[0].output)
        self.output = inputTensor.reshape(self.outShape)
    
    def backward(self):
        pass