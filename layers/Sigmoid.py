import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    """
    This is the Sigmoid layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array(self.inShapes[0])

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        outputTensor = np.array(self.inNodes[0].output)
        self.output = 1 / (1 + np.exp(outputTensor))
    
    def backward(self):
        pass