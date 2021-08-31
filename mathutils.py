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