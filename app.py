import imghdr
import time

from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, Response
from werkzeug.utils import secure_filename
from named import named_peopel
from test import VideoCamera
import cv2
import dlib
import math
import pickle
import os
import face_recognition

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.jpeg']
app.config['UPLOAD_PATH'] = 'Images'
app.config['UNKNOWN_FOLDER'] = 'Unknown'


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cascPathface = os.path.dirname(
            cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        # load the harcaascade in the cascade classifier
        self.faceCascade = cv2.CascadeClassifier(self.cascPathface)
        self.data = pickle.loads(open('face_enc', "rb").read())

        self.WIDTH = 1280
        self.HEIGHT = 720

        self.RAW_IMAGE_DIR = "./Unknown"

    def __del__(self):
        self.cap.release()

    def find_plate(self, plate2=None, new_cv=None):

        gray2 = new_cv.cvtColor(plate2, new_cv.COLOR_BGR2GRAY)
        detections = self.faceCascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=1)

        for (x2, y2, w2, h2) in detections:
            number_plate = gray2[y2:y2 + h2, x2:x2 + w2]
            # result = new_cv.imshow("Number plate", number_plate)
            # print(f"Number plate : {result}")

        return

    def estimatespeed2(self, location1, location2):
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        ppm = 8.8
        d_meters = d_pixels / ppm
        # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 18
        speed2 = d_meters * fps * 3.6
        return speed2

    def get_frame(self):
        rectangleColor = (0, 255, 0)
        frameCounter = 0
        currentFaceID = 0

        faceTracker = {}
        faceLocation1 = {}
        faceLocation2 = {}
        faceName = {}

        matchfaceID2 = None

        rc, image = self.cap.read()
        # resultImage = image.copy()
        frameCounter = frameCounter + 1
        faceIDtoDelete = []
        matchfaceID2 = None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        if frameCounter % 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.1, 19, 18, (50, 50))

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

                matches = face_recognition.compare_faces(self.data["encodings"],
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
                        name = self.data["names"][i]
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
                    image_list = os.listdir(self.RAW_IMAGE_DIR)
                    if name == "Unknown":
                        img_name = f'Unknown/{len(image_list) + 1}.jpg'
                        cv2.imwrite(img_name, image[y:y + h, x:x + w])
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

            cv2.rectangle(image, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            cv2.putText(image, str(name), (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    named_peopel()
    return render_template('index.html', files=files)


@app.route('/', methods=['POST'])
def upload_files():
    folder = request.form['folder']
    if folder:
        os.mkdir(app.config['UPLOAD_PATH'] + "/" + folder)
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'] + "/" + folder, filename))
    named_peopel()
    # time.sleep(2)
    return 'Done', 204


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/video')
def video():
    named_peopel()
    # time.sleep(2)

    return render_template('video.html')


def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)