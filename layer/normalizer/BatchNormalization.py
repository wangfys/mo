import numpy as np
from functools import reduce
from ..Base import BaseLayer
from ...globalvar import *

class BatchNormalization(BaseLayer):
    """
    This is the batch normalization layer.
    Here are the explanations of args:
        isTrain: a boolean, False by default
        epsilon: very small value to be added in case of divded by 0, 1e-10 by default
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.epsilon = args["epsilon"] if "epsilon" in args else 1e-10
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        self.params = ["scale", "shift", "batchSeen", "meanSeen", "varianceSeen"]
        self.scale = np.array([1])
        self.shift = np.array([0])
        self.batchSeen = np.array([0])
        self.meanSeen = np.array([0])
        self.varianceSeen = np.array([0])
        if Config["imperative"]:
            self.forward({})

    def applyGradientDescent(self, applyFunc):
        if not self.fix:
            self.scale.ravel()[:] = applyFunc(self.scale.flatten(), self.paramGradients["scale"].flatten())
            self.shift.ravel()[:] = applyFunc(self.shift.flatten(), self.paramGradients["shift"].flatten())

    def calcGradient(self):
        rowNumber = self.outSize
        columnNumber = self.outSize
        inputVector = self.inNodes[0].output.flatten()
        mean = np.mean(inputVector)
        variance = np.var(inputVector) + self.epsilon
        sigma = np.sqrt(variance)
        varGradient = np.zeros((self.outSize,))
        thisInputGradient = np.zeros((rowNumber, columnNumber))
        for i in range(rowNumber):
            for j in range(columnNumber):
                if i == j:
                    varGradient[i] += 2 * (inputVector[j] - mean) * (self.outSize - 1) / (self.outSize ** 2)
                else:
                    varGradient[i] += -2 * (inputVector[j] - mean) / (self.outSize ** 2)
        for i in range(rowNumber):
            for j in range(columnNumber):
                if i == j:
                    thisInputGradient[i][j] = (self.outSize - 1) * sigma / self.outSize - (inputVector[i] - mean) * varGradient[j] / (2 * sigma)
                else:
                    thisInputGradient[i][j] = -sigma / self.outSize -(inputVector[i] - mean) * varGradient[j] / (2 * sigma)
        thisInputGradient *= self.scale / variance
        thisScaleGradient = (inputVector - mean) / sigma
        thisShiftGradient = np.ones((self.outSize,))
        inputGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisInputGradient) for outNode in self.outNodes])
        scaleGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisScaleGradient) for outNode in self.outNodes])
        shiftGradient = reduce(np.add, [np.dot(outNode.inputGradients[self.name], thisShiftGradient) for outNode in self.outNodes])
        self.inputGradients[self.inNodes[0].name] = inputGradient
        self.paramGradients["scale"] = scaleGradient
        self.paramGradients["shift"] = shiftGradient

    def forward(self, feedInput):
        inputTensor = np.array(self.inNodes[0].output)
        if self.isTrain:
            mean = np.mean(self.inNodes[0].output)
            variance = np.var(self.inNodes[0].output)
            self.meanSeen = self.meanSeen * self.batchSeen + mean
            self.varianceSeen = self.varianceSeen * self.batchSeen + variance
            self.batchSeen += 1
            self.meanSeen /= self.batchSeen
            self.varianceSeen /= self.batchSeen
            inputTensor = (inputTensor - mean) / np.sqrt(variance + self.epsilon)
            self.output = self.scale * inputTensor + self.shift
        else:
            mean = self.meanSeen
            variance = self.varianceSeen * (self.inShapes[0][0] - 1) / self.inShapes[0][0]
            inputTensor = (inputTensor - mean) / np.sqrt(variance + self.epsilon)
            self.output = self.scale * inputTensor + self.shift
