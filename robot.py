from threading import Thread
import time

from events import LeftEyePupil, RightEyePupil, EyeAspectRatio, BlinkEvent, LeftEyeBorder, RightEyeBorder, FaceBorder, \
    RecordingStart
from ohbot import ohbot
from mathutils import MovingAverage


class RobotHandler(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.blink_aspect_ratio = None
        self.max_aspect_ratio = 0
        self.queue = queue
        ohbot.reset()
        self.last_lid_update = time.time()
        self.last_eye_update = time.time()
        self.last_head_update = time.time()
        self.lid_mae = MovingAverage(20)
        self.left_eye_pos = None
        self.right_eye_pos = None
        self.left_eye_pupil = None
        self.right_eye_pupil = None
        self.face_border = None

    @staticmethod
    def append(item):
        file_object = open('recording.csv', 'a')
        file_object.write(str(item)+"\n")
        file_object.close()

    def run(self):
        self.append(RecordingStart())
        while True:
            item = self.queue.get()
            print(item)
            self.append(item)
            if isinstance(item, FaceBorder):
                self.face_border = item
            if isinstance(item, LeftEyeBorder):
                self.left_eye_pos = item
            if isinstance(item, RightEyeBorder):
                self.right_eye_pos = item
            if isinstance(item, LeftEyePupil):
                #print("LeftEyePupil cx={} cy={}".format(item.cx, item.cy))
                self.left_eye_pupil = item
            if isinstance(item, RightEyePupil):
                print("RightEyePupil cx={} cy={}".format(item.cx, item.cy))
                self.right_eye_pupil = item
            if isinstance(item, EyeAspectRatio):
                print("EyeAspectRatio max={} ear={}".format(self.max_aspect_ratio, item.current_aspect))
                # if self.max_aspect_ratio < item.current_aspect:
                #    self.max_aspect_ratio = item.current_aspect
                self.max_aspect_ratio = self.lid_mae.compute_mae(item.current_aspect)
                if self.blink_aspect_ratio is not None:
                    difference = (item.current_aspect - self.blink_aspect_ratio) / (self.max_aspect_ratio - self.blink_aspect_ratio)
                    print("difference={}".format(difference))
                    self.ohbot_lid_blink(difference)
            if isinstance(item, BlinkEvent):
                print("BlinkEvent ear={}".format(item.current_aspect))
                self.blink_aspect_ratio = item.current_aspect
            self.queue.task_done()
            self.ohbot_eye_move()
            self.ohbot_head_turn()

    def ohbot_head_turn(self):
        if self.face_border is None:
            return
        now = time.time()
        if now - self.last_head_update > 0.2:
            dx1 = 0
            dx2 = 0
            if self.right_eye_pos is not None:
                dx1 = self.right_eye_pos.lx - self.face_border.lx
                if dx1 < 0:
                    dx1 = 0
            if self.left_eye_pos is not None:
                dx2 = self.face_border.rx - self.left_eye_pos.rx
                if dx2 < 0:
                    dx2 = 0
            turn = 0
            if dx1 < dx2:
                if dx2 != 0:
                    turn = -1 + dx1 / dx2
            else:
                if dx1 != 0:
                    turn = dx2 / dx1
            head_pos = int(5.0 + 1.5 * turn)
            print("head_pos={} t={} dx1={} dx2={}".format(head_pos, turn, dx1, dx2))
            #head_pos = 5
            ohbot.move(ohbot.HEADTURN, head_pos)
            self.last_head_update = time.time()

    def ohbot_lid_blink(self, difference):
        now = time.time()
        if now - self.last_lid_update > 0.2:
            difference *= 2.5
            lid_position = int(2 + difference * 6)
            if lid_position > 8:
                lid_position = 8
            if lid_position < 2:
                lid_position = 2
            ohbot.move(ohbot.LIDBLINK, lid_position)
            self.last_lid_update = time.time()
            
    def ohbot_eye_move(self):
        now = time.time()
        if now - self.last_lid_update > 0.2:
            l_dx = l_dy = r_dx = r_dy = 0.5
            if self.left_eye_pos is not None and self.left_eye_pupil is not None:
                (l_dx, l_dy) = RobotHandler.get_eye_move(self.left_eye_pos, self.left_eye_pupil)
            if self.right_eye_pos is not None and self.right_eye_pupil is not None:
                (r_dx, r_dy) = RobotHandler.get_eye_move(self.right_eye_pos, self.right_eye_pupil)

            x = (l_dx + r_dx) * 0.5
            y = (l_dy + r_dy) * 0.5
            x = 1.8 * l_dx
            y = 5.5 * l_dy
            print("x={} y={}".format(x,y))

            xx = 2 + 6*x
            yy = 8 - 6*y
            if xx < 2:
                xx = 2
            if xx > 8:
                xx = 8
            if yy < 2:
                yy = 2
            if yy > 8:
                yy = 8
                
            print("xx={} yy={}".format(xx, yy))

            ohbot.move(ohbot.EYETURN, xx)
            ohbot.move(ohbot.EYETILT, yy)
            self.last_eye_update = time.time()

    @staticmethod
    def get_eye_move(eye_pos, pupil):
        lx = eye_pos.lx
        ly = eye_pos.ly
        rx = eye_pos.rx
        ry = eye_pos.ry
        print("lx={} rx={}".format(lx, rx))
        dx = (pupil.cx - lx) / (rx - lx)
        dy = (pupil.cy - ly) / (ry - ly)
        if dy < 0.0:
            dy = 0.0
        if dy > 1.0:
            dy = 1.0
        if dx < 0.0:
            dx = 0.0
        if dx > 1.0:
            dx = 1.0
        print("dx={} dy={}".format(dx, dy))
        return dx, dy
