import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    """
    This is the Sigmoid layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        self.output = 1 / (1 + np.exp(-inputTensor))
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        inputVector = (self.output * (1 - self.output)).flatten()
        thisGradient = np.diag(inputVector)
        inputGradient = np.zeros((1, columnNumber))
        for outNode in self.outNodes:
            inputGradient = inputGradient + np.dot(outNode.inputGradients[self.name], thisGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)