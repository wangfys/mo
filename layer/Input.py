import numpy as np
from functools import reduce
from .Base import BaseLayer
from ..globalvar import *

class Input(BaseLayer):
    """
    This is the input layer dealing with input data.
    Here are the explanation of args:
        name: the name of this layer, should be unique
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=0)
        self.outShape = np.array(args["shape"])
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.input = np.array(args["data"], dtype=Dtype)
            if (self.input.shape != self.outShape).any():
                raise Exception("data size error in '%s'" % self.name)
            self.forward({})

    def calcGradient(self):
        inputGradient = reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes])
        self.inputGradients[""] = inputGradient

    def forward(self, feedInput):
        if not Config["imperative"] or feedInput != {}:
            if self.name in feedInput:
                self.input = np.array(feedInput[self.name], dtype=Dtype)
                if (self.input.shape != self.outShape).any():
                    raise Exception("data size error in '%s'" % self.name)
            else:
                raise Exception("can not find input data for '%s'" % self.name)
        self.output = self.input
