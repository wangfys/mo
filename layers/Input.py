import numpy as np
from .Base import BaseLayer

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
    
    def forward(self, feedInput):
        if self.name in feedInput:
            self.input = np.array(feedInput[self.name])
        else:
            raise Exception("can not find input data for '%s'" % self.name)
        self.output = self.input

    def backward(self, applyGradient):
        pass