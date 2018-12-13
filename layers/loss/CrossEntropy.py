import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...lib import Config

class CrossEntropy(BaseLayer):
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=2)
        if (self.inShapes[0] != self.inShapes[1]).any():
            raise Exception("the shape of the two inputs of '%s' are not equivalent" % self.name)
        self.outShape = np.array([1])
        self.outSize = 1
        if Config["imperative"]:
            self.forward({})
    
    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        self.output = np.sum(-self.inNodes[0].output * np.log(self.inNodes[1].output + 1e-10)).reshape(self.outShape)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        if self.inNodes[0] == self.inNodes[1]:
            thisInputGradient = -(np.log(self.inNodes[0].output + 1e-10) + 1).reshape((self.outSize, self.inSizes[0]))
            inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
            self.inputGradients[self.inNodes[0].name] = inputGradient
        else:
            thisInputGradients = [-np.log(self.inNodes[1].output + 1e-10).reshape((self.outSize, self.inSizes[0])), (-self.inNodes[0].output / (self.inNodes[1].output + 1e-10)).reshape((self.outSize, self.inSizes[1]))]
            inputGradients = []
            for i in range(2):
                inputGradients.append(reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradients[i]) for outNode in self.outNodes]))
                self.inputGradients[self.inNodes[i].name] = inputGradients[i]
        BaseLayer.backward(self, applyGradient)