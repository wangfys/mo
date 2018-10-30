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
        self.K_init = args["K_init"] if "K_init" in args else np.zeros
        self.b_init = args["b_init"] if "b_init" in args else np.zeros
        self.params = ["K", "b"]
        self.outShape = np.array((self.inShapes[0][0], args["unitNum"]))
        self.outSize = np.prod(self.outShape)

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
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        thisGradient = np.zeros((self.outSize, self.inSizes[0]))
        thisKGradient = np.zeros((self.outSize, self.K.size))
        thisBGradient = np.zeros((self.outSize, self.b.size))
        for i in range(self.inShapes[0][0]):
            thisGradient[i*self.K.shape[0]:(i+1)*self.K.shape[0], i*self.K.shape[1]:(i+1)*self.K.shape[1]] = self.K
            thisBGradient[i*self.outShape[1]:(i+1)*self.outShape[1]] = np.diag(np.ones((self.b.size)))
            for j in range(self.K.shape[0]):
                thisKGradient[i*self.outShape[1]+j, j*self.K.shape[1]:(j+1)*self.K.shape[1]] = self.inNodes[0].output[j*self.K.shape[1]:(j+1)*self.K.shape[1]]
        inputGradient = np.zeros((1, columnNumber))
        KGradient = np.zeros((1, self.K.size))
        bGradient = np.zeros((1, self.b.size))
        for outNode in self.outNodes:
            inputGradient = inputGradient + np.dot(outNode.inputGradients[self.name], thisGradient)
            KGradient = KGradient + np.dot(outNode.inputGradients[self.name], thisKGradient)
            bGradient = bGradient + np.dot(outNode.inputGradients[self.name], thisBGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        self.K.ravel()[:] = applyGradient(self.K.flatten(), KGradient.flatten())
        self.b.ravel()[:] = applyGradient(self.b.flatten(), bGradient.flatten())
        self.paramGradients["K"] = KGradient
        self.paramGradients["b"] = bGradient
        BaseLayer.backward(self, applyGradient)