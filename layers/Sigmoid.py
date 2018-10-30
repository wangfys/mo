import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    """
    This is the Sigmoid layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array(self.inShapes[0])
        self.inputGradients = [np.zeros(self.inShapes[0])]

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        self.output = 1 / (1 + np.exp(inputTensor))
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        self.inputGradients[0] = self.output * (self.output - 1)
        BaseLayer.backward(self, applyGradient)