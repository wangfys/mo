import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...lib import getNumpyShape
from ...globalvar import *

class Multiply(BaseLayer):
    """
    This is an multiply operator which can multiply two tensors. It works like np.multiply().
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=2)
        self.outShape = getNumpyShape(np.ones(self.inNodes[0].outShape)*np.ones(self.inNodes[1].outShape))
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        rowNumber = self.outSize
        if self.inNodes[0] == self.inNodes[1]:
            self.inputGradients[self.inNodes[0].name] = np.diag((2 * self.inNodes[0].output).flatten())
        else:
            columnNumber0 = self.inNodes[0].outSize
            columnNumber1 = self.inNodes[1].outSize
            thisInputGradient0 = np.zeros((rowNumber, columnNumber0), dtype=Dtype)
            thisInputGradient1 = np.zeros((rowNumber, columnNumber1), dtype=Dtype)
            for i in range(columnNumber0):
                tmp = np.zeros(self.inNodes[0].outShape, dtype=Dtype)
                tmp.ravel()[i] = 1
                tmp = tmp * self.inNodes[1].output
                for j in np.argwhere(tmp!=0):
                    thisInputGradient0[j, i] = tmp[j]
            for i in range(columnNumber1):
                tmp = np.zeros(self.inNodes[1].outShape, dtype=Dtype)
                tmp.ravel()[i] = 1
                tmp = self.inNodes[0].output * tmp
                for j in np.argwhere(tmp!=0):
                    thisInputGradient1[j, i] = tmp[j]
            inputGradient0 = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient0)
            inputGradient1 = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient1)
            self.inputGradients[self.inNodes[0].name] = inputGradient0
            self.inputGradients[self.inNodes[1].name] = inputGradient1

    def forward(self, feedInput):
        self.output = self.inNodes[0].output * self.inNodes[1].output
