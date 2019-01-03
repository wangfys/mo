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
        columnNumber = self.inSizes[0]
        rowNumber = self.outSize
        thisInputGradient = np.zeros((rowNumber, columnNumber), dtype=Dtype)
        inputTensorIndex = np.arange(self.inNodes[0].outSize).reshape(self.inShapes[0])
        inputTensor = np.pad(self.inNodes[0].output, ((0, 0), (0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)), "constant")
        inputTensorIndex = np.pad(inputTensorIndex, ((0, 0), (0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)), "constant", constant_values=-1)
        thisInputGradientIndex = -1
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for i in range(self.outShape[2]):
                    for j in range(self.outShape[3]):
                        thisInputGradientIndex += 1
                        for x in range(self.ksize):
                            for y in range(self.ksize):
                                if self.output[n][c][i][j] == inputTensor[n][c][i*self.ksize+x][j*self.ksize+y]:
                                    if inputTensorIndex[n][c][i*self.ksize+x][j*self.ksize+y] != -1:
                                        thisInputGradient[thisInputGradientIndex][inputTensorIndex[n][c][i*self.ksize+x][j*self.ksize+y]] = 1
        inputGradient = np.dot(reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]), thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient

    def forward(self, feedInput):
        inputTensor = np.pad(self.inNodes[0].output, ((0, 0),(0, 0), (self.pad_top, self.pad_bottom),(self.pad_left, self.pad_right)), "constant")
        outputTensor = np.zeros(self.outShape, dtype=Dtype)
        for n in range(self.outShape[0]):
            for c in range(self.outShape[1]):
                for i in range(self.outShape[2]):
                    for j in range(self.outShape[3]):
                        outputTensor[n][c][i][j] = np.max(inputTensor[n][c][i*self.ksize:(i+1)*self.ksize, j*self.ksize:(j+1)*self.ksize])
        self.output = outputTensor
