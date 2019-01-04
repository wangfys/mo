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
        self.epsilon = args["epsilon"] if "epsilon" in args else 1e-5
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)
        self.params = ["scale", "shift", "batchSeen", "meanSeen", "varianceSeen"]
        self.scale = np.array([1], dtype=Config["Dtype"])
        self.shift = np.array([0], dtype=Config["Dtype"])
        self.batchSeen = np.array([0])
        self.meanSeen = np.array([0], dtype=Config["Dtype"])
        self.varianceSeen = np.array([0], dtype=Config["Dtype"])
        if Config["imperative"]:
            self.forward({})

    def applyGradientDescent(self, applyFunc):
        if not self.fix:
            self.scale.ravel()[:] = applyFunc(self.scale.flatten(), self, "scale")
            self.shift.ravel()[:] = applyFunc(self.shift.flatten(), self, "shift")

    def calcGradient(self):
        tmpTensor = self.inNodes[0].output - self.mean
        std_inv = 1. / np.sqrt(self.variance + self.epsilon)
        outGradient = reduce(np.add, [outNode.inputGradients[self.name] for outNode in self.outNodes]).reshape(self.outShape)
        normGradient = outGradient * self.scale
        varianceGradient = np.sum(normGradient * tmpTensor, axis=0) * -0.5 * std_inv ** 3
        meanGradient = np.sum(normGradient * -std_inv, axis=0) + varianceGradient * np.mean(-2. * tmpTensor, axis=0)
        inputGradient = (normGradient * std_inv) + (varianceGradient * 2 * tmpTensor / self.outShape[0]) + (meanGradient / self.outShape[0])
        scaleGradient = np.sum(outGradient * self.norm)
        shiftGradient = np.sum(outGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient.flatten()
        self.paramGradients["scale"] = scaleGradient
        self.paramGradients["shift"] = shiftGradient

    def forward(self, feedInput):
        inputTensor = self.inNodes[0].output
        if self.isTrain:
            self.mean = np.mean(self.inNodes[0].output)
            self.variance = np.var(self.inNodes[0].output)
            self.meanSeen = self.meanSeen * self.batchSeen + self.mean
            self.varianceSeen = self.varianceSeen * self.batchSeen + self.variance
            self.batchSeen += 1
            self.meanSeen /= self.batchSeen
            self.varianceSeen /= self.batchSeen
            self.norm = (inputTensor - self.mean) / np.sqrt(self.variance + self.epsilon)
            self.output = self.scale * self.norm + self.shift
        else:
            mean = self.meanSeen
            variance = self.varianceSeen * (self.inShapes[0][0] - 1) / self.inShapes[0][0]
            inputTensor = (inputTensor - mean) / np.sqrt(variance + self.epsilon)
            self.output = self.scale * inputTensor + self.shift
