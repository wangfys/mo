import numpy as np
from functools import reduce
from .Base import BaseLayer
from .. import initializer
from ..globalvar import *

def img2row(inputTensor, layer):
    rows = np.zeros((layer.outShape[0]*layer.outShape[2]*layer.outShape[3], layer.inShapes[0][1]*layer.kernelSize[1]*layer.kernelSize[2]), dtype=Dtype)
    count = -1
    for n in range(layer.outShape[0]):
        for h in range(layer.outShape[2]):
            for w in range(layer.outShape[3]):
                count += 1
                rows[count] = inputTensor[n, :, h:h+layer.kernelSize[1], w:w+layer.kernelSize[2]].flatten()
    return rows

def row2img(rowTensor, layer, calcB=False):
    result = rowTensor.reshape((layer.outShape[0], layer.outShape[2], layer.outShape[3], layer.outShape[1]))
    if calcB:
        for n in range(layer.outShape[0]):
            result[n] += layer.b
    return result.transpose((0, 3, 1, 2))

class Conv2D(BaseLayer):
    """
    This is the 2D convolution layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
        kernel: the shape of kernel, like (output_channels, output_height, output_width), notice that output_height should be the same as output_width
        K_init: the initializer of K, mo.initializer.Constant(0) by default
        b_init: the initializer of b, mo.initializer.Constant(0) by default
        fix: whether to fix the parameters during training, False by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.kernelSize = args["kernel"]
        self.K_init = args["K_init"] if "K_init" in args else initializer.Constant(0)
        self.b_init = args["b_init"] if "b_init" in args else initializer.Constant(0)
        self.params = ["K", "b"]
        self.outShape = np.array((self.inShapes[0][0], self.kernelSize[0], self.inShapes[0][2]-self.kernelSize[1]+1, self.inShapes[0][3]-self.kernelSize[2]+1))
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            if "thisParam" in args:
                self.init(thisParam=args["thisParam"])
            else:
                self.init()
            self.forward({})

    def applyGradientDescent(self, applyFunc):
        if not self.fix:
            self.K.ravel()[:] = applyFunc(self.K.flatten(), self.paramGradients["K"].flatten())
            self.b.ravel()[:] = applyFunc(self.b.flatten(), self.paramGradients["b"].flatten())

    def calcGradient(self):
        thisInputGradient = np.zeros((self.outSize, self.inSizes[0]), dtype=Dtype)
        thisKGradient = np.zeros((self.outSize, self.K.size), dtype=Dtype)
        thisBGradient = np.zeros((self.outSize, self.b.size), dtype=Dtype)
        index = 0
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for h in range(self.outShape[2]):
                    for w in range(self.outShape[3]):
                        tmpInputGradient = np.zeros(self.inShapes[0], dtype=Dtype)
                        tmpKGradient = np.zeros((self.kernelSize[0], self.inShapes[0][1], self.kernelSize[1], self.kernelSize[2]), dtype=Dtype)
                        tmpBGradient = np.zeros(self.kernelSize[0], dtype=Dtype)
                        tmpInputGradient[n, :, h:h+self.kernelSize[1], w:w+self.kernelSize[2]] = self.K[c, :]
                        tmpKGradient[c, :] = self.inNodes[0].output[n, :, h:h+self.kernelSize[1], w:w+self.kernelSize[2]]
                        tmpBGradient[c] = 1
                        thisInputGradient[index] = tmpInputGradient.flatten()
                        thisKGradient[index] = tmpKGradient.flatten()
                        thisBGradient[index] = tmpBGradient.flatten()
                        index += 1
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        KGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisKGradient) for outNode in self.outNodes])
        bGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisBGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient
        self.paramGradients["K"] = KGradient
        self.paramGradients["b"] = bGradient

    def forward(self, feedInput):
        self.rowTensor = img2row(self.inNodes[0].output, self)
        self.colKernel = self.K.reshape((self.outShape[1], -1)).T
        outputTensor = np.dot(self.rowTensor, self.colKernel)
        self.output = row2img(outputTensor, self, True)

    def init(self, jsonParam=None, thisParam=None):
        if jsonParam == None:
            self.K = self.K_init.initialize((self.kernelSize[0], self.inShapes[0][1], self.kernelSize[1], self.kernelSize[2]))
            self.b = self.b_init.initialize(self.kernelSize[0])
        else:
            self.K = np.array(jsonParam[self.name]["K"], dtype=Dtype)
            self.b = np.array(jsonParam[self.name]["b"], dtype=Dtype)
        if Config["imperative"] and thisParam != None:
            self.K = np.array(thisParam["K"], dtype=Dtype)
            self.b = np.array(thisParam["b"], dtype=Dtype)
