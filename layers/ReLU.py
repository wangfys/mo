import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    """
    This is the ReLU layer.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        outputTensor = np.array(self.inNodes[0].output)
        self.output = np.maximum(outputTensor, 0)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        inputVector = np.where(self.output>0, 1, 0).flatten()
        thisGradient = np.diag(inputVector)
        inputGradient = np.zeros((1, columnNumber))
        for outNode in self.outNodes:
            inputGradient = inputGradient + np.dot(outNode.inputGradients[self.name], thisGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)