import time


class RecordingStart(object):
    def __init__(self):
        pass

    def __str__(self):
        return "{}, RecordingStart, 0, 0, 0, 0".format(time.time())


class EyeAspectRatio(object):
    def __init__(self, current_aspect):
        self.current_aspect = current_aspect

    def __str__(self):
        return "{}, EyeAspectRatio, {}, 0, 0, 0".format(time.time(), self.current_aspect)


class BlinkEvent(object):
    def __init__(self, current_aspect):
        self.current_aspect = current_aspect

    def __str__(self):
        return "{}, BlinkEvent, {}, 0, 0, 0".format(time.time(), self.current_aspect)


class LeftEyePupil(object):
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def __str__(self):
        return "{}, LeftEyePupil, {}, {}, 0, 0".format(time.time(), self.cx, self.cy)


class RightEyePupil(object):
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def __str__(self):
        return "{}, RightEyePupil, {}, {}, 0, 0".format(time.time(), self.cx, self.cy)


class FaceBorder(object):
    def __init__(self, lx, ly, rx, ry):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry

    def __str__(self):
        return "{}, FaceBorder, {}, {}, {}, {}".format(time.time(), self.lx, self.ly, self.rx, self.ry)


class LeftEyeBorder(object):
    def __init__(self, lx, ly, rx, ry):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry

    def __str__(self):
        return "{}, LeftEyeBorder, {}, {}, {}, {}".format(time.time(), self.lx, self.ly, self.rx, self.ry)


class RightEyeBorder(object):
    def __init__(self, lx, ly, rx, ry):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry

    def __str__(self):
        return "{}, RightEyeBorder, {}, {}, {}, {}".format(time.time(), self.lx, self.ly, self.rx, self.ry)
