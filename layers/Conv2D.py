import numpy as np
from .Base import BaseLayer
from .. import initializers

class Conv2D(BaseLayer):
    """
    This is the 2D convolution layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
        kernel: the shape of kernel, like (output_channels, output_height, output_width), notice that output_height should be the same as output_width
        K_init: the initializer of K, mo.initializers.Constant(0) by default
        b_init: the initializer of b, mo.initializers.Constant(0) by default
        fix: whether to fix the parameters during training, False by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.kernelSize = args["kernel"]
        self.K_init = args["K_init"] if "K_init" in args else initializers.Constant(0)
        self.b_init = args["b_init"] if "b_init" in args else initializers.Constant(0)
        self.params = ["K", "b"]
        self.outShape = np.array((self.inShapes[0][0], self.kernelSize[0], self.inShapes[0][2]-self.kernelSize[1]+1, self.inShapes[0][3]-self.kernelSize[2]+1))
        self.outSize = np.prod(self.outShape)
    
    def init(self, jsonParam=None):
        if jsonParam == None:
            self.K = self.K_init.initialize((self.kernelSize[0], self.inShapes[0][0], self.kernelSize[1], self.kernelSize[2]))
            self.b = self.b_init.initialize(self.kernelSize[0])
        BaseLayer.init(self, jsonParam)
    
    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        inputTensor = np.array(self.inNodes[0].output)
        outputTensor = np.zeros(self.outShape)
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for h in range(self.outShape[2]):
                    for w in range(self.outShape[3]):
                        outputTensor[n, c, h, w] = np.sum(inputTensor[n, :, h:h+self.kernelSize[1], w:w+self.kernelSize[2]] * self.K[c, :]) + self.b[c]
        self.output = outputTensor
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        thisInputGradient = np.zeros((self.outSize, self.inSizes[0]))
        thisKGradient = np.zeros((self.outSize, self.K.size))
        thisBGradient = np.zeros((self.outSize, self.b.size))
        index = 0
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for h in range(self.outShape[2]):
                    for w in range(self.outShape[3]):
                        tmpInputGradient = np.zeros(self.inShapes[0])
                        tmpKGradient = np.zeros((self.kernelSize[0], self.inShapes[0][0], self.kernelSize[1], self.kernelSize[2]))
                        tmpBGradient = np.zeros(self.kernelSize[0])
                        tmpInputGradient[n, :, h:h+self.kernelSize[1], w:w+self.kernelSize[2]] = self.K[c, :]
                        tmpKGradient[c, :] = self.inNodes[0].output[n, :, h:h+self.kernelSize[1], w:w+self.kernelSize[2]]
                        tmpBGradient[c] = 1
                        thisInputGradient[index] = tmpInputGradient.flatten()
                        thisKGradient[index] = tmpKGradient.flatten()
                        thisBGradient[index] = tmpBGradient.flatten()
                        index += 1
        inputGradient = np.zeros((1, columnNumber))
        KGradient = np.zeros((1, self.K.size))
        bGradient = np.zeros((1, self.b.size))
        for outNode in self.outNodes:
            inputGradient += np.dot(outNode.inputGradients[self.name], thisInputGradient)
            KGradient += np.dot(outNode.inputGradients[self.name], thisKGradient)
            bGradient += np.dot(outNode.inputGradients[self.name], thisBGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        if not self.fix:
            self.K.ravel()[:] = applyGradient(self.K.flatten(), KGradient.flatten())
            self.b.ravel()[:] = applyGradient(self.b.flatten(), bGradient.flatten())
            self.paramGradients["K"] = KGradient
            self.paramGradients["b"] = bGradient
        BaseLayer.backward(self, applyGradient)