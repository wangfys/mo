from enum import Enum

forwardStatus = Enum("forwardStatus", ("uninitialized", "initialized", "computed"))
backwardStatus = Enum("backwardStatus", ("unforwarded", "forwarded", "computed"))