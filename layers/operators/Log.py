import numpy as np
from ..Base import BaseLayer

class Log(BaseLayer):
    """
    This is the log layer.
    Here are the explanations of args:
        epsilon: very small value to be added in case of log(0)
    """
    def __init__(self, **args):
        BaseLayer.__init__(self, args, inputNum=1)
        self.epsilon = args["epsilon"] if "epsilon" in args else 0
        self.outShape = np.array(self.inShapes[0])
        self.outSize = np.prod(self.outShape)

    def forward(self, feedInput):
        if BaseLayer.forward(self, feedInput):
            return None
        outputTensor = np.array(self.inNodes[0].output)
        self.output = np.log(outputTensor + self.epsilon)
    
    def backward(self, applyGradient):
        if BaseLayer.preBackward(self):
            return None
        columnNumber = self.inSizes[0]
        inputVector = (1 / np.array(self.inNodes[0].output + self.epsilon)).flatten()
        thisInputGradient = np.diag(inputVector)
        inputGradient = np.zeros((1, columnNumber))
        for outNode in self.outNodes:
            inputGradient += np.dot(outNode.inputGradients[self.name], thisInputGradient)
        self.inputGradients[self.inNodes[0].name] = inputGradient
        BaseLayer.backward(self, applyGradient)