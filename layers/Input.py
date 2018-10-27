import numpy as np
from .Base import BaseLayer
from .Base import layerStatus

class Input(BaseLayer):
    """
    This is the input layer dealing with input data.
    Here are the explanation of args:
        name: the name of this layer, should be unique
    """
    def __init__(self, name, shape):
        self.name = name
        self.status = layerStatus.uninitialized
        self.inNodes = []
        self.outNodes = []
        self.params = []
        self.outShape = np.array(shape)
    
    def init(self, jsonParam=None):
        self.status = layerStatus.ready
    
    def forward(self, feedInput):
        try:
            self.input = np.array(feedInput[self.name])
        except:
            raise Exception("can not find input data for '%s'" % self.name)
        self.output = self.input

    def backward(self):
        pass