from enum import Enum
import numpy as np

forwardStatus = Enum("forwardStatus", ("uninitialized", "initialized", "computed"))
backwardStatus = Enum("backwardStatus", ("unforwarded", "forwarded", "computed"))

def getNumpyShape(data):
    if data.shape == ():
        return np.array([1])
    else:
        return np.array(data.shape)

Config = {"imperative": False}