import numpy as np
from functools import reduce
from .Base import BaseLayer
from ..lib import Config

class Flatten(BaseLayer):
    """
    This is the flatten layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.outShape = np.array((self.inShapes[0][0], np.prod(self.inShapes[0][1:])))
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})
    
    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        self.output = inputTensor.reshape(self.outShape)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        inputVector = np.ones([columnNumber])
        thisInputGradient = np.diag(inputVector)
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)