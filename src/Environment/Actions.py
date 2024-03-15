from enum import Enum


class Actions(Enum):
    FORWARD = 0
    BACKWARD = 1
    ROTATE = 2

    #WAIT = 4


class Events(Enum):
    BLOCKED = 0
    NEW = 1
    REPEATED = 2
    FINISHED = 3
    TIMEOUT = 4
    WAITED = 5

