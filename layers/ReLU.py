import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    """
    This is the ReLU layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array(self.inShapes[0])

    def forward(self, feedInput):
        BaseLayer.forward(self, feedInput)
        outputTensor = np.array(self.inNodes[0].output)
        self.output = np.maximum(outputTensor, 0)
    
    def backward(self):
        pass