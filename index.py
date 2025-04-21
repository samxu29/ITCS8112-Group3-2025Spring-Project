import os
from importlib import import_module
from flask import Flask, render_template, Response

# Import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module(f'camera_{os.environ["CAMERA"]}').Camera
else:
    from camera import Camera

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for img tag's src attribute."""
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', threaded=True)
