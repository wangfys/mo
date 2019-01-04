import numpy as np
from functools import reduce
from .Base import BaseLayer
from .. import initializer
from ..globalvar import *
from ..lib import im2col_indices, col2im_indices

class Conv2D(BaseLayer):
    """
    This is the 2D convolution layer.
    Here are the explanation of args:
        name: the name of this layer, should be unique
        input: the input of this layer
        kernel: the shape of kernel, like (output_channels, output_height, output_width), notice that output_height should be the same as output_width
        stride: the step size of kernel, 1 by default
        K_init: the initializer of K, mo.initializer.Constant(0) by default
        b_init: the initializer of b, mo.initializer.Constant(0) by default
        fix: whether to fix the parameters during training, False by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.kernelSize = args["kernel"]
        self.K_init = args["K_init"] if "K_init" in args else initializer.Constant(0)
        self.b_init = args["b_init"] if "b_init" in args else initializer.Constant(0)
        self.stride = args["stride"] if "stride" in args else 1
        self.params = ["K", "b"]
        self.outShape = np.array((self.inShapes[0][0], self.kernelSize[0], (self.inShapes[0][2]-self.kernelSize[1])//self.stride+1, (self.inShapes[0][3]-self.kernelSize[2])//self.stride+1))
        self.outSize = np.prod(self.outShape)
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
        self.paramGradients["b"] = np.sum(outputGradient, axis=(0, 2, 3))

        outputGradient = outputGradient.transpose((1, 2, 3, 0)).reshape((self.outShape[1], -1))
        self.paramGradients["K"] = np.dot(outputGradient, self.X_col.T)

        inputGradient_col = np.dot(self.K.reshape((self.outShape[1], -1)).T, outputGradient)
        self.inputGradients[self.inNodes[0].name] = col2im_indices(inputGradient_col, self.inShapes[0], self.kernelSize[1], self.kernelSize[2], padding=0, stride=self.stride).flatten()

    def forward(self, feedInput):
        self.X_col = im2col_indices(self.inNodes[0].output, self.kernelSize[1], self.kernelSize[2], padding=0, stride=self.stride)
        K_col = self.K.reshape((self.kernelSize[0], -1))
        outputTensor = np.dot(K_col, self.X_col)
        for c in range(self.kernelSize[0]):
            outputTensor[c] += self.b[c]
        outputTensor = outputTensor.reshape((self.outShape[1], self.outShape[2], self.outShape[3], self.outShape[0]))
        self.output = outputTensor.transpose((3, 0, 1, 2))

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
