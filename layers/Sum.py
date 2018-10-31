import numpy as np
from .Base import BaseLayer

class Sum(BaseLayer):
    """
    This is the Sum layer which can add all the element in the tensor.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        self.output = np.sum(inputTensor)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        thisInputGradient = np.ones((1, columnNumber))
        inputGradient = np.zeros((1, columnNumber))
        for outNode in self.outNodes:
            inputGradient += np.dot(outNode.inputGradients[self.name], thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)