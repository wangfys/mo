import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...lib import getNumpyShape
from ...globalvar import *

class Sum(BaseLayer):
    """
    This is the reduce sum layer which can add the elements in the tensor.
    Here are the explanation of args:
        axis: the same meaning in numpy.sum()
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.axis = args["axis"] if "axis" in args else None
        self.outShape = getNumpyShape(np.sum(np.zeros(self.inShapes[0]), axis=self.axis))
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        rowNumber = self.outSize
        columnNumber = self.inSizes[0]
        if self.axis == None:
            thisInputGradient = np.ones((rowNumber, columnNumber), dtype=Dtype)
        else:
            thisInputGradient = np.zeros((rowNumber, columnNumber), dtype=Dtype)
            for i in range(columnNumber):
                tmp = np.zeros(self.inShapes[0], dtype=Dtype)
                tmp.ravel()[i] = 1
                tmp = np.sum(tmp, axis=self.axis).flatten()
                for j in np.argwhere(tmp!=0):
                    thisInputGradient[j, i] = tmp[j]
        inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        inputTensor = self.inNodes[0].output
        self.output = np.sum(inputTensor, axis=self.axis).reshape(self.outShape)
