import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class CrossEntropy(BaseLayer):
    """
    This is the cross entropy layer.
    Here are the explanations of args:
        isTrain: a boolean, False by default
        epsilon: very small value to be added in case of log(0), 1e-10 by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=2)
        self.epsilon = args["epsilon"] if "epsilon" in args else 1e-10
        if (self.inShapes[0] != self.inShapes[1]).any():
            raise Exception("the shape of the two inputs of '%s' are not equivalent" % self.name)
        self.outShape = np.array([1])
        self.outSize = 1
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        if self.inNodes[0] == self.inNodes[1]:
            thisInputGradient = -(np.log(self.inNodes[0].output + self.epsilon) + 1).reshape((self.outSize, self.inSizes[0]))
            inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
            self.inputGradients[self.inNodes[0].name] = inputGradient
        else:
            thisInputGradients = [-np.log(self.inNodes[1].output + self.epsilon).reshape((self.outSize, self.inSizes[0])), (-self.inNodes[0].output / (self.inNodes[1].output + self.epsilon)).reshape((self.outSize, self.inSizes[1]))]
            inputGradients = []
            for i in range(2):
                inputGradients.append(np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradients[i]))
                self.inputGradients[self.inNodes[i].name] = inputGradients[i]

    def forward(self, feedInput):
        self.output = np.sum(-self.inNodes[0].output * np.log(self.inNodes[1].output + self.epsilon)).reshape(self.outShape)
