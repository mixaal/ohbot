from threading import Thread
import time

from events import LeftEyePupil, RightEyePupil, EyeAspectRatio, BlinkEvent
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
        self.lid_mae = MovingAverage(20)

    def run(self):
        while True:
            item = self.queue.get()
            print(item)
            if isinstance(item, LeftEyePupil):
                print("LeftEyePupil cx={} cy={}".format(item.cx, item.cy))
            if isinstance(item, RightEyePupil):
                print("RightEyePupil cx={} cy={}".format(item.cx, item.cy))
            if isinstance(item, EyeAspectRatio):
                print("EyeAspectRatio max={} ear={}".format(self.max_aspect_ratio, item.current_aspect))
                # if self.max_aspect_ratio < item.current_aspect:
                #    self.max_aspect_ratio = item.current_aspect
                self.max_aspect_ratio = self.lid_mae.compute_mae(item.current_aspect)
                if self.blink_aspect_ratio is not None:
                    difference = (item.current_aspect - self.blink_aspect_ratio) / (self.max_aspect_ratio - self.blink_aspect_ratio)
                    print("difference={}".format(difference))
                    now = time.time()
                    if now - self.last_lid_update > 0.2:
                        ohbot_lid_position = int(2+difference*6)
                        if ohbot_lid_position > 8:
                            ohbot_lid_position = 8
                        if ohbot_lid_position < 2:
                            ohbot_lid_position = 2
                        ohbot.move(ohbot.LIDBLINK, ohbot_lid_position)
                        self.last_lid_update = time.time()
            if isinstance(item, BlinkEvent):
                print("BlinkEvent ear={}".format(item.current_aspect))
                self.blink_aspect_ratio = item.current_aspect
            self.queue.task_done()
