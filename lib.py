from enum import Enum
import numpy as np

forwardStatus = Enum("forwardStatus", ("uninitialized", "initialized", "computed"))
backwardStatus = Enum("backwardStatus", ("unforwarded", "forwarded", "computed"))

def getNumpyShape(data):
    if data.shape == ():
        return np.array([1])
    else:
        return np.array(data.shape)

def mergeComputeSequence(a, b):
    bNodes = set(b)
    meregedSequence = []
    i = 0
    j = 0
    while i < len(a):
        if not a[i] in bNodes:
            meregedSequence.append(a[i])
        else:
            while j < len(b):
                if b[j] != a[i]:
                    meregedSequence.append(b[j])
                    j += 1
                else:
                    j += 1
                    break
            meregedSequence.append(a[i])
        i += 1
    while j < len(b):
        meregedSequence.append(b[j])
        j += 1
    return meregedSequence