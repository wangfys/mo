import numpy as np
from ..Base import BaseLayer

class Add(BaseLayer):
    """
    This is an add operator which can add lots of tensors with same shape.
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        for i in range(len(self.inShapes)):
            if (self.inShapes[i] != self.inShapes[0]).all():
                raise Exception("can not add tensors with different shapes in %s" % self.name)
        self.outShape = self.inShapes[0]
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        outputTensor = np.zeros(self.outShape)
        for inNode in self.inNodes:
            outputTensor += inNode.output
        self.output = outputTensor
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        rowNumber = self.outSize
        for inNode in self.inNodes:
            thisInputGradient = np.zeros((rowNumber, rowNumber))
            for i in range(rowNumber):
                thisInputGradient[i][i] = 1
            inputGradient = np.zeros((1, rowNumber))
            for outNode in self.outNodes:
                inputGradient += np.dot(outNode.inputGradients[self.name], thisInputGradient)
            self.inputGradients[inNode.name] = inputGradient
        BaseLayer.backward(self, applyGradient)