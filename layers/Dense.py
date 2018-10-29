import numpy as np
from .Base import BaseLayer

class Dense(BaseLayer):
    """
    This is the fully connected layer. The input should be proccessed via a flatten layer to become 1 dimensional.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
        unitNum: the number of units in this fully connected layer.
        K_init: the initializer of K, np.zeros if not set
        b_init: the initializer of b, np.zeros if not set
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args)
        self.outShape = np.array((self.inShapes[0][0], args["unitNum"]))
        self.K_init = args["K_init"] if "K_init" in args else np.zeros
        self.b_init = args["b_init"] if "b_init" in args else np.zeros
        self.params = ["K", "b"]

    def init(self, jsonParam=None):
        if jsonParam == None:
            self.K = self.K_init((self.outShape[1], self.inShapes[0][1]))
            self.b = self.b_init(self.outShape[1])
        BaseLayer.init(self, jsonParam)
    
    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        outputTensor = np.zeros(self.outShape)
        for i in range(self.inShapes[0][0]):
            outputTensor[i] = np.dot(self.K, inputTensor[i]) + self.b
        self.output = outputTensor.reshape(self.outShape)
    
    def backward(self):
        pass