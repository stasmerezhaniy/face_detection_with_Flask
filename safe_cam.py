import cv2
import dlib
import math
import pickle
import os
import face_recognition

all_videos = [
    'Highway - 56310.mp4',
    'faces2.mp4',
    'faces.mp4',
    'faces - 1900.mp4',  # <<----
    'Car - 2165.mp4',  # нет
    'Traffic - 27260.mp4',
    'vb.mp4',  # нет
]
video = cv2.VideoCapture(0)
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())

WIDTH = 1280
HEIGHT = 720

RAW_IMAGE_DIR = "../Unknown"


def find_plate(plate2=None, new_cv=None):

    gray2 = new_cv.cvtColor(plate2, new_cv.COLOR_BGR2GRAY)
    detections = faceCascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=1)

    for (x2, y2, w2, h2) in detections:
        number_plate = gray2[y2:y2 + h2, x2:x2 + w2]
        # result = new_cv.imshow("Number plate", number_plate)
        # print(f"Number plate : {result}")

    return


def estimatespeed2(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 18
    speed2 = d_meters * fps * 3.6
    return speed2


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentFaceID = 0

    faceTracker = {}
    faceLocation1 = {}
    faceLocation2 = {}
    faceName = {}

    matchfaceID2 = None

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        resultImage = image.copy()
        frameCounter = frameCounter + 1
        faceIDtoDelete = []
        matchfaceID2 = None

        rgb = cv2.cvtColor(resultImage, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        names = []

        for faceID in faceTracker.keys():
            trackingQuality = faceTracker[faceID].update(image)

            if trackingQuality < 10:
                faceIDtoDelete.append(faceID)

        for faceID in faceIDtoDelete:
            # print('Removing faceID ' + str(faceID) + ' from list of trackers.')
            # print('Removing faceID ' + str(faceID) + ' previous location.')
            # print('Removing faceID ' + str(faceID) + ' current location.')
            faceTracker.pop(faceID, None)
            faceLocation1.pop(faceID, None)
            faceLocation2.pop(faceID, None)
            faceName.pop(faceID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 19, 18, (50, 50))

            for ((_x, _y, _w, _h), encoding) in zip(faces, encodings):
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.6 * w
                y_bar = y + 0.6 * h

                matchfaceID = None

                for faceID in faceTracker.keys():
                    trackedPosition = faceTracker[faceID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.6 * t_w
                    t_y_bar = t_y + 0.6 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchfaceID = faceID

                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                # print(matches)
                # set name =inknown if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    # Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        # Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                        # increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                    # set name which has highest count
                    name = max(counts, key=counts.get)
                # print(name)
                if matchfaceID is None:

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    faceTracker[currentFaceID] = tracker
                    faceLocation1[currentFaceID] = [x, y, w, h]
                    faceName[currentFaceID] = name
                    matchfaceID2 = currentFaceID
                    currentFaceID = currentFaceID + 1
                    image_list = os.listdir(RAW_IMAGE_DIR)
                    if name == "Unknown":
                        img_name = f'Unknown{len(image_list) + 1}.jpg'
                        cv2.imwrite(img_name, image[y:y+h, x:x+w])
                else:
                    matchfaceID2 = None

        # cv2.line(resultImage, (0, 480), (1280, 480), (255, 0, 0), 5)
        for faceID in faceTracker.keys():
            trackedPosition = faceTracker[faceID].get_position()
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            name = faceName[faceID]

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            cv2.putText(resultImage, str(name), (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        cv2.imshow('result', resultImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Streaming finished')
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
