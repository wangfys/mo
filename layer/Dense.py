import numpy as np
from functools import reduce
from .Base import BaseLayer
from .. import initializer
from ..globalvar import *

class Dense(BaseLayer):
    """
    This is the fully connected layer. The input should be proccessed via a flatten layer to become 1 dimensional.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
        unitNum: the number of units in this fully connected layer.
        K_init: the initializer of K, mo.initializer.Constant(0) by default
        b_init: the initializer of b, mo.initializer.Constant(0) by default
        fix: whether to fix the parameters during training, False by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.K_init = args["K_init"] if "K_init" in args else initializer.Constant(0)
        self.b_init = args["b_init"] if "b_init" in args else initializer.Constant(0)
        self.params = ["K", "b"]
        self.outShape = np.array((self.inShapes[0][0], args["unitNum"]))
        self.outSize = np.prod(self.outShape)
        self.K = np.zeros((self.outShape[1], self.inShapes[0][1]),dtype=Config["Dtype"])
        self.b = np.zeros((self.outShape[1],), dtype=Config["Dtype"])
        if Config["imperative"]:
            if "thisParam" in args:
                self.init(thisParam=args["thisParam"])
            else:
                self.init()
            self.forward({})

    def applyGradientDescent(self, applyFunc):
        if not self.fix:
            self.K.ravel()[:] = applyFunc(self.K.flatten(), self, "K")
            self.b.ravel()[:] = applyFunc(self.b.flatten(), self, "b")

    def calcGradient(self):
        outputGradient = reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]).reshape(self.outShape)
        self.inputGradients[self.inNodes[0].name] = np.dot(outputGradient, self.K)
        self.paramGradients["K"] += np.dot(self.inNodes[0].output.T, outputGradient).T.flatten()
        self.paramGradients["b"] += np.sum(outputGradient, axis=0).flatten()

    def forward(self, feedInput):
        self.output = np.dot(self.inNodes[0].output, self.K.T) + self.b

    def init(self, jsonParam=None, thisParam=None):
        if jsonParam == None:
            self.K = self.K_init.initialize((self.outShape[1], self.inShapes[0][1]))
            self.b = self.b_init.initialize(self.outShape[1])
        else:
            self.K = np.array(jsonParam[self.name]["K"], dtype=Config["Dtype"])
            self.b = np.array(jsonParam[self.name]["b"], dtype=Config["Dtype"])
        if Config["imperative"] and thisParam != None:
            self.K = np.array(thisParam["K"], dtype=Config["Dtype"])
            self.b = np.array(thisParam["b"], dtype=Config["Dtype"])
