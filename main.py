import cv2
from queue import Queue
from robot import RobotHandler
from tracking import FaceTracker


def nothing(x):
    pass


cv2.namedWindow('frame')
cv2.createTrackbar('threshold', 'frame', 0, 255, nothing)

queue = Queue()
robot_handler = RobotHandler(queue)
robot_handler.start()
face_tracker = FaceTracker(queue)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(0)

currentFrame = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    frame = cv2.flip(frame, 1)
    # image = detect_face(frame)
    (image, eyes) = face_tracker.facial_detection(frame)

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    # print(frame)
#    frame *= 2
#    frame = np.clip(frame, 0, 255)
    cv2.imshow('frame', image)
    if eyes is not None:
        cv2.imshow('eyes', eyes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
