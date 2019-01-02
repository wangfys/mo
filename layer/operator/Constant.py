import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...lib import getNumpyShape
from ...globalvar import *

class Constant(BaseLayer):
    """
    This is the constant layer.
    """
    def __init__(self, data):
        BaseLayer.__init__(self, {"input": []}, inputNum=0)
        self.input = np.array(data, dtype=Dtype)
        self.outShape = getNumpyShape(self.input)
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        thisInputGradient = np.diag(np.ones((self.outSize,), dtype=Dtype))
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        self.inputGradients[""] = inputGradient

    def forward(self, feedInput):
        self.output = self.input.reshape(self.outShape)
