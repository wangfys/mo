import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...lib import getNumpyShape

class Add(BaseLayer):
    """
    This is an add operator which can add several tensors. It works like reduce(np.add, datalist).
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = getNumpyShape(reduce(np.add, [np.zeros(inNode.outShape) for inNode in self.inNodes]))
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        self.output = reduce(np.add, [inNode.output for inNode in self.inNodes]).reshape(self.outShape)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        rowNumber = self.outSize
        for i in range(len(self.inNodes)):
            columnNumber = self.inNodes[i].outSize
            thisInputGradient = np.zeros((rowNumber, columnNumber))
            for j in range(columnNumber):
                tmp = [np.zeros(inNode.outShape) for inNode in self.inNodes]
                tmp[i].ravel()[j] = 1
                tmp = reduce(np.add, tmp).reshape(self.outShape).flatten()
                for k in np.argwhere(tmp!=0):
                    thisInputGradient[k, j] = tmp[k]
            inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
            if self.inNodes[i].name in self.inputGradients:
                self.inputGradients[self.inNodes[i].name] += inputGradient
            else:
                self.inputGradients[self.inNodes[i].name] = inputGradient
        BaseLayer.backward(self, applyGradient)