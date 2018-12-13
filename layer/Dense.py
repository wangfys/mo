import numpy as np
from functools import reduce
from .Base import BaseLayer
from .. import initializer
from ..lib import Config

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
        if Config["imperative"]:
            if "thisParam" in args:
                self.init(thisParam=args["thisParam"])
            else:
                self.init()
            self.forward({})

    def init(self, jsonParam=None, thisParam=None):
        if jsonParam == None:
            self.K = self.K_init.initialize((self.outShape[1], self.inShapes[0][1]))
            self.b = self.b_init.initialize(self.outShape[1])
        if Config["imperative"] and thisParam != None:
            self.K = np.array(thisParam["K"])
            self.b = np.array(thisParam["b"])
        if not Config["imperative"]:
            BaseLayer.init(self, jsonParam)
    
    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        outputTensor = np.zeros(self.outShape)
        for i in range(self.inShapes[0][0]):
            outputTensor[i] = np.dot(self.K, inputTensor[i]) + self.b
        self.output = outputTensor.reshape(self.outShape)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        thisInputGradient = np.zeros((self.outSize, self.inSizes[0]))
        thisKGradient = np.zeros((self.outSize, self.K.size))
        thisBGradient = np.zeros((self.outSize, self.b.size))
        for i in range(self.inShapes[0][0]):
            thisInputGradient[i*self.K.shape[0]:(i+1)*self.K.shape[0], i*self.K.shape[1]:(i+1)*self.K.shape[1]] = self.K
            thisBGradient[i*self.outShape[1]:(i+1)*self.outShape[1]] = np.diag(np.ones((self.b.size)))
            for j in range(self.K.shape[0]):
                thisKGradient[i*self.outShape[1]+j, j*self.K.shape[1]:(j+1)*self.K.shape[1]] = self.inNodes[0].output[j]
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        KGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisKGradient) for outNode in self.outNodes])
        bGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisBGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient
        if not self.fix:
            self.K.ravel()[:] = applyGradient(self.K.flatten(), KGradient.flatten())
            self.b.ravel()[:] = applyGradient(self.b.flatten(), bGradient.flatten())
            self.paramGradients["K"] = KGradient
            self.paramGradients["b"] = bGradient
        BaseLayer.backward(self, applyGradient)