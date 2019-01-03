import numpy as np
from functools import reduce
from .Base import BaseLayer
from .. import initializer
from ..globalvar import *

def img2row(inputTensor, tensorShape, kernelShape):
    rows = np.zeros((tensorShape[0]*tensorShape[2]*tensorShape[3], kernelShape[1]*kernelShape[2]*kernelShape[3]), dtype=Config["Dtype"])
    count = -1
    for n in range(tensorShape[0]):
        for h in range(tensorShape[2]):
            for w in range(tensorShape[3]):
                count += 1
                rows[count] = inputTensor[n, :, h:h+kernelShape[2], w:w+kernelShape[3]].flatten()
    return rows

def row2img(rowTensor, tensorShape, calcB=[]):
    result = rowTensor.reshape((tensorShape[0], tensorShape[2], tensorShape[3], tensorShape[1]))
    if len(calcB) > 0:
        for n in range(tensorShape[0]):
            result[n] += calcB[0]
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
        inputGradient = reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]).reshape(self.outShape)
        inputGradient = np.pad(inputGradient, ((0, 0), (0, 0), (self.kernelSize[1] - 1, self.kernelSize[1] - 1), (self.kernelSize[2] - 1, self.kernelSize[2] - 1)), "constant")
        flippedK = self.K[:,:,::-1,::-1].swapaxes(0, 1)
        inputGradient = row2img(np.dot(img2row(inputGradient, self.inShapes[0], flippedK.shape), flippedK.reshape((flippedK.shape[0], -1)).T), self.inShapes[0]).flatten()
        KGradient = np.zeros((self.K.size,), dtype=Config["Dtype"])
        colTensor = self.rowTensor.T
        colTensor = np.concatenate([colTensor for _ in range(self.outShape[1])], axis=1)
        for i in range(np.prod(self.K.shape[1:])):
            tmp = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), colTensor[i])
            KGradient[i::np.prod(self.K.shape[1:])] = tmp / self.outShape[1]
        bGradient = np.zeros((self.b.size,), dtype=Config["Dtype"])
        for i in range(self.b.size):
            tmp = np.zeros(self.outShape, dtype=Config["Dtype"])
            tmp[:,i,:,:] = 1
            bGradient[i] = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), tmp.flatten())
        self.inputGradients[self.inNodes[0].name] = inputGradient.reshape((1, -1))
        self.paramGradients["K"] = KGradient
        self.paramGradients["b"] = bGradient

    def forward(self, feedInput):
        tensorShape = self.outShape
        kernelShape = self.K.shape
        self.rowTensor = img2row(self.inNodes[0].output, tensorShape, kernelShape)
        colKernel = self.K.reshape((self.outShape[1], -1)).T
        outputTensor = np.dot(self.rowTensor, colKernel)
        self.output = row2img(outputTensor, tensorShape, [self.b])

    def init(self, jsonParam=None, thisParam=None):
        if jsonParam == None:
            self.K = self.K_init.initialize((self.kernelSize[0], self.inShapes[0][1], self.kernelSize[1], self.kernelSize[2]))
            self.b = self.b_init.initialize(self.kernelSize[0])
        else:
            self.K = np.array(jsonParam[self.name]["K"], dtype=Config["Dtype"])
            self.b = np.array(jsonParam[self.name]["b"], dtype=Config["Dtype"])
        if Config["imperative"] and thisParam != None:
            self.K = np.array(thisParam["K"], dtype=Config["Dtype"])
            self.b = np.array(thisParam["b"], dtype=Config["Dtype"])
