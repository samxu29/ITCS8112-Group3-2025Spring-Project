import os
import cv2
import network
from network import model
import time
import pyttsx3
import threading
from greenlet import getcurrent as get_ident


class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    event = CameraEvent()

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # start background frame thread
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame."""
        BaseCamera.last_access = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event.wait()
        BaseCamera.event.clear()

        return BaseCamera.frame
        
    @staticmethod
    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses.')

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last minute then stop the thread
            if time.time() - BaseCamera.last_access > 60:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        BaseCamera.thread = None

class Speech():
    def __init__(self):
        # self.running = False
        self.engine = pyttsx3.init()
        self.engine.say("")
        self.engine.runAndWait()
        time.sleep(3)

    def update(self, text):
        self.engine.endLoop()
        self.engine.say(text)
        self.engine.runAndWait()

class Camera(BaseCamera):
    video_source = 0
    letter_pred = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z",
        26: "del",
        27: "nothing",
        28: "space"
    }

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        print(f"Initializing camera with source: {Camera.video_source}")
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source
        print(f"Setting camera source to: {source}")

    @staticmethod
    def frames():
        print(f"Attempting to open camera with source: {Camera.video_source}")
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            print(f"Failed to open camera with source: {Camera.video_source}")
            raise RuntimeError('Could not start camera.')
        print("Camera successfully opened")

        # width = camera.get(3) #get width and height of img
        # height = camera.get(4)
        width = 1120
        height = 630
        
        crop_image_width = 224 #input size for pytorch pre-trained models
        crop_image_height = 224
        
        startW = int ((width-crop_image_width)/2)
        startH = int ((height - crop_image_height)/2)
        endH = startH + crop_image_height
        endW = startW + crop_image_width

        start_pt = (int(startW),int(startH))
        end_pt = (int(endW), int(endH) )
        color = (255,0,0)
        thickness = 1

        draw_H = startH - 20
        draw_W = int ((startW +endW)/2 )
        draw_pt = (int (draw_W), int(draw_H))
        font = cv2.FONT_HERSHEY_PLAIN

        tts = Speech()
        pred_old = ''
        now = time.time()
        print('OK')
        text = ''

        while True:
            # read current frame
            ret, img = camera.read()
            if not ret or img is None:
                print("Failed to grab frame from camera")
                time.sleep(0.1)  # Wait a bit before trying again
                continue

            try:
                # DO ERROR CHECK IF WEBCAM CAMERA NOT DETECTED
                img = cv2.resize(img, (16*60, 9*60))
                img = cv2.flip (img, 1)

                img = cv2.rectangle(img, start_pt, end_pt, color, thickness=thickness) #draw input rectangle

                cropped_img = img[startH:endH,startW:endW] #crop img to rectangle
                
                index = network.predict(model,cropped_img)

                prediction = Camera.letter_pred[index]
                cur = time.time()

                if prediction != pred_old and cur-now > 2:
                    print(prediction)
                    tts.update(prediction.lower())
                    now = cur
                    pred_old = prediction
                # print(cur-now)
                # prediction = str(index)
                text += prediction

                img  = cv2.putText(img, prediction, draw_pt, font, 2, color,thickness)

                # encode as a jpeg image and return it
                yield cv2.imencode('.jpg', img)[1].tobytes()
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                time.sleep(0.1)  # Wait a bit before trying again
                continue
