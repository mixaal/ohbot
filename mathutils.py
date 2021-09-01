from collections import deque


class MovingAverage(object):
    def __init__(self, N):
        self.N = N
        self.deque = deque()

    def _put(self, element):
        self.deque.append(element)
        while len(self.deque) > self.N:
            self.deque.popleft()

    def compute_mae(self, element):
        self._put(element)
        s = 0
        for x in self.deque:
            s += x
        return s / len(self.deque)


class ShapeContour(object):
    def __init__(self, shape):
        self.shape = shape

    def min(self):
        min_x = None
        min_y = None
        for (x, y) in self.shape:
            if min_x is None:
                min_x = x
            if min_y is None:
                min_y = y
            if min_y > y:
                min_y = y
            if min_x > x:
                min_x = x

        return min_x, min_y

    def max(self):
        max_x = None
        max_y = None
        for (x, y) in self.shape:
            if max_x is None:
                max_x = x
            if max_y is None:
                max_y = y
            if max_y < y:
                max_y = y
            if max_x < x:
                max_x = x

        return max_x, max_y
