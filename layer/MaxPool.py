import numpy as np
from functools import reduce
from .Base import BaseLayer
from ..globalvar import *
from ..lib import im2col_indices, col2im_indices

class MaxPool(BaseLayer):
    """
    This is the max pooling layer.
    Here are the explanation of args:
        size: pool size
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.ksize = args["ksize"]
        self.padding = args["padding"] if "padding" in args else 0
        self.outShape = np.array((self.inShapes[0][0], self.inShapes[0][1], (self.inShapes[0][2] + 2 * self.padding) // self.ksize, (self.inShapes[0][3] + 2 * self.padding) // self.ksize))
        self.pad_top = (self.outShape[2] * self.ksize - self.inShapes[0][2]) // 2
        self.pad_bottom = self.outShape[2] * self.ksize - self.inShapes[0][2] - self.pad_top
        self.pad_left = (self.outShape[3] * self.ksize - self.inShapes[0][3]) // 2
        self.pad_right = self.outShape[3] * self.ksize - self.inShapes[0][3] - self.pad_left
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        inputGradientCol = np.zeros(self.colShape, dtype=Config["Dtype"])
        inputGradient = reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]).reshape(self.outShape).transpose((2, 3, 0, 1)).ravel()
        inputGradientCol[self.maxIndex, range(self.maxIndex.size)] = inputGradient
        inputGradient = col2im_indices(inputGradientCol, (self.inShapes[0][0]*self.inShapes[0][1], 1, self.inShapes[0][2], self.inShapes[0][3]), self.ksize, self.ksize, padding=self.padding, stride=self.ksize)
        self.inputGradients[self.inNodes[0].name] = inputGradient.flatten()

    def forward(self, feedInput):
        inputTensor = self.inNodes[0].output.reshape((self.inShapes[0][0]*self.inShapes[0][1], 1, self.inShapes[0][2], self.inShapes[0][3]))
        inputCol = im2col_indices(inputTensor, self.ksize, self.ksize, padding=self.padding, stride=self.ksize)
        self.colShape = inputCol.shape
        self.maxIndex = np.argmax(inputCol, axis=0)
        outputTensor = inputCol[self.maxIndex, range(self.maxIndex.size)]
        outputTensor = outputTensor.reshape((self.outShape[2], self.outShape[3], self.outShape[0], self.outShape[1]))
        self.output = outputTensor.transpose((2, 3, 0, 1))
