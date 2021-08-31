class EyeAspectRatio(object):
    def __init__(self, current_aspect):
        self.current_aspect = current_aspect


class BlinkEvent(object):
    def __init__(self, current_aspect):
        self.current_aspect = current_aspect

class LeftEyePupil(object):
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy


class RightEyePupil(object):
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy


class FaceBorder(object):
    def __init__(self, lx, ly, rx, ry):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry


class LeftEyeBorder(object):
    def __init__(self, lx, ly, rx, ry):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry


class RightEyeBorder(object):
    def __init__(self, lx, ly, rx, ry):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
