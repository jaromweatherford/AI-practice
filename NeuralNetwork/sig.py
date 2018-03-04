import math


def sigmoid(n):
    return 1 / (1 + math.e ** (0 - n))


print sigmoid(-.152)