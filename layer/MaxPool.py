import numpy as np
from functools import reduce
from .Base import BaseLayer
from ..globalvar import *

class MaxPool(BaseLayer):
    """
    This is the max pooling layer.
    Here are the explanation of args:
        size: pool size
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.ksize = args["ksize"]
        self.outShape = np.array((self.inShapes[0][0], self.inShapes[0][1], self.inShapes[0][2] // self.ksize + 1, self.inShapes[0][3] // self.ksize + 1))
        self.pad_top = (self.outShape[2] * self.ksize - self.inShapes[0][2]) // 2
        self.pad_bottom = self.outShape[2] * self.ksize - self.inShapes[0][2] - self.pad_top
        self.pad_left = (self.outShape[3] * self.ksize - self.inShapes[0][3]) // 2
        self.pad_right = self.outShape[3] * self.ksize - self.inShapes[0][3] - self.pad_left
        self.outSize = np.prod(self.outShape)
        if Config["imperative"]:
            self.forward({})

    def calcGradient(self):
        inputGradient = np.zeros((self.inSizes[0],), dtype=Config["Dtype"])
        inputTensorIndex = np.arange(self.inNodes[0].outSize).reshape(self.inShapes[0])
        inputTensor = np.pad(self.inNodes[0].output, ((0, 0), (0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)), "constant")
        inputTensorIndex = np.pad(inputTensorIndex, ((0, 0), (0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)), "constant", constant_values=-1)
        count = -1
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for i in range(self.outShape[2]):
                    for j in range(self.outShape[3]):
                        count += 1
                        maxIndex = np.argmax(inputTensor[n,c,i*self.ksize:(i+1)*self.ksize,j*self.ksize:(j+1)*self.ksize])
                        inputIndex = inputTensorIndex[n,c,i*self.ksize:(i+1)*self.ksize,j*self.ksize:(j+1)*self.ksize].flatten()[maxIndex]
                        if inputIndex != -1:
                            inputGradient[inputIndex] += reduce(np.add, [outNode.inputGradients[self.name][0][count] for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        inputTensor = np.pad(self.inNodes[0].output, ((0, 0),(0, 0), (self.pad_top, self.pad_bottom),(self.pad_left, self.pad_right)), "constant")
        outputTensor = np.zeros(self.outShape, dtype=Config["Dtype"])
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for i in range(self.outShape[2]):
                    for j in range(self.outShape[3]):
                        outputTensor[n][c][i][j] = np.max(inputTensor[n][c][i*self.ksize:(i+1)*self.ksize, j*self.ksize:(j+1)*self.ksize])
        self.output = outputTensor
