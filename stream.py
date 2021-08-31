import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Eye tracking: https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
# Eye blink detection: https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# Face detection using DNN: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

model='models/res10_300x300_ssd_iter_140000.caffemodel'
proto='models/deploy.prototxt.txt'
shape_predictor='models/shape_predictor_68_face_landmarks.dat'

net = cv2.dnn.readNetFromCaffe(proto, model)

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

left_eye_idx = [x for x in range(lStart, lEnd)]
right_eye_idx = [x for x in range(rStart, rEnd)]
#print(left_eye_idx)

def nothing(x):
    pass

cv2.namedWindow('frame')
cv2.createTrackbar('threshold', 'frame', 0, 255, nothing)

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        print("cx={} cy={}".format(cx, cy))
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def eye_on_mask(mask, shape, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def facial_detection(image):
    global COUNTER
    global TOTAL
    
    eyes_gray = None
    #frame = imutils.resize(image, width=450)
    frame = image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 1 - means upsampling
    rects = detector(gray, 0)
    #print(rects)
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, shape, left_eye_idx)
        mask = eye_on_mask(mask, shape, right_eye_idx)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

        threshold = cv2.getTrackbarPos('threshold', 'frame')
        if threshold<=0:
            threshold = 85
        #print(threshold)
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)
        mid = (shape[42][0] + shape[39][0]) // 2
        contouring(thresh[:, 0:mid], mid, frame)
        contouring(thresh[:, mid:], mid, frame, True)

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        print("ear={}".format(ear))

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            print("BLINK, counter={}".format(COUNTER))
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0

    if eyes_gray is None:
        return (frame, None)
    else:
        return (frame, thresh)

def detect_face(image):
    #image = cv2.imread(file_name)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
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
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 2.45, (0, 0, 255), 2)
    #cv2.imwrite('/tmp/out.jpg', image)
    return image


# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(0)

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    frame = cv2.flip(frame,1)
    #image = detect_face(frame)
    (image, eyes) = facial_detection(frame)

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    #print(frame)
#    frame *= 2
#    frame = np.clip(frame, 0, 255)
    cv2.imshow('frame',image)
    if eyes is not None:
       cv2.imshow('eyes',eyes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
