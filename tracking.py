import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from events import LeftEyePupil, RightEyePupil, LeftEyeBorder, RightEyeBorder, EyeAspectRatio, BlinkEvent
from mathutils import ShapeContour


# Eye tracking: https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
# Eye blink detection: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# Face detection using DNN: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/


class FaceTracker(object):
    def __init__(self, queue):
        self.queue = queue
        self.model = 'models/res10_300x300_ssd_iter_140000.caffemodel'
        self.proto = 'models/deploy.prototxt.txt'
        self.shape_predictor = 'models/shape_predictor_68_face_landmarks.dat'
        self.net = cv2.dnn.readNetFromCaffe(self.proto, self.model)
        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.left_eye_idx = [x for x in range(self.lStart, self.lEnd)]
        self.right_eye_idx = [x for x in range(self.rStart, self.rEnd)]

    def contouring(self, thresh, mid, img, left=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            matrix = cv2.moments(cnt)
            cx = int(matrix['m10']/matrix['m00'])
            cy = int(matrix['m01']/matrix['m00'])
            if left:
                cx += mid
                self.queue.put(LeftEyePupil(cx, cy))
            else:
                self.queue.put(RightEyePupil(cx, cy))
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        except:
            pass

    @staticmethod
    def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        c = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (a + b) / (2.0 * c)
        # return the eye aspect ratio
        return ear

    @staticmethod
    def eye_on_mask(mask, shape, side):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    def facial_detection(self, image):
        eyes_gray = None
        # frame = imutils.resize(image, width=450)
        frame = image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 1 - means upsampling
        rects = self.detector(gray, 0)
        # print(rects)
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            left_eye = shape[self.lStart:self.lEnd]
            right_eye = shape[self.rStart:self.rEnd]

            left_eye_shape = ShapeContour(left_eye)
            right_eye_shape = ShapeContour(right_eye)

            (left_eye_min_x, left_eye_min_y) = left_eye_shape.min()
            (left_eye_max_x, left_eye_max_y) = left_eye_shape.max()

            (right_eye_min_x, right_eye_min_y) = right_eye_shape.min()
            (right_eye_max_x, right_eye_max_y) = right_eye_shape.max()

            cv2.rectangle(image, (left_eye_min_x, left_eye_min_y), (left_eye_max_x, left_eye_max_y), (255, 0, 0), 2)
            cv2.rectangle(image, (right_eye_min_x, right_eye_min_y), (right_eye_max_x, right_eye_max_y), (255, 0, 0), 2)
            self.queue.put(LeftEyeBorder(left_eye_min_x, left_eye_min_y, left_eye_max_x, left_eye_max_y))
            self.queue.put(RightEyeBorder(right_eye_min_x, right_eye_min_y, right_eye_max_x, right_eye_max_y))

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask = FaceTracker.eye_on_mask(mask, shape, self.left_eye_idx)
            mask = FaceTracker.eye_on_mask(mask, shape, self.right_eye_idx)
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(frame, frame, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

            threshold = cv2.getTrackbarPos('threshold', 'frame')
            if threshold <= 0:
                threshold = 85
            # print(threshold)
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.medianBlur(thresh, 3)
            thresh = cv2.bitwise_not(thresh)
            mid = (shape[42][0] + shape[39][0]) // 2
            self.contouring(thresh[:, 0:mid], mid, frame)
            self.contouring(thresh[:, mid:], mid, frame, True)

            left_ear = FaceTracker.eye_aspect_ratio(left_eye)
            right_ear = FaceTracker.eye_aspect_ratio(right_eye)
            # average the eye aspect ratio together for both eyes
            ear = (left_ear + right_ear) / 2.0
            # print("ear={}".format(ear))
            self.queue.put(EyeAspectRatio(ear))

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                self.queue.put(BlinkEvent(ear))
                print("BLINK, counter={}".format(self.COUNTER))
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
            # reset the eye frame counter
                self.COUNTER = 0

        if eyes_gray is None:
            return frame, None
        else:
            return frame, thresh

    def detect_face(self, image):
        # image = cv2.imread(file_name)
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
 
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2.45, (0, 0, 255), 2)
        # cv2.imwrite('/tmp/out.jpg', image)
        return image
