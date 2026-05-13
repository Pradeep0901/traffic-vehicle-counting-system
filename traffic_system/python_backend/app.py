import os
import cv2
import numpy as np
import sqlite3
import json
import time
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from tracker import TrafficTracker
from database import init_db, DB_PATH

app = Flask(__name__)
CORS(app)
tracker = TrafficTracker()

# Video source: sample video or webcam fallback
VIDEO_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'sample.mp4')
if not os.path.exists(VIDEO_SOURCE):
    VIDEO_SOURCE = 0

camera = None
start_time = time.time()

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(VIDEO_SOURCE)
        if not camera.isOpened():
            print("Warning: Could not open video source. Using blank frames.")

    while True:
        if camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success:
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue
        else:
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(frame, "No Video Source", (250, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        processed_frame = tracker.process_frame(frame)
        if processed_frame is None:
            processed_frame = frame

        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    elapsed = time.time() - start_time
    density_ratio = getattr(tracker, '_last_density_ratio', 0)
    density = "Low"
    if density_ratio > 0.3:
        density = "High"
    elif density_ratio > 0.1:
        density = "Medium"

    return jsonify({
        "total": tracker.total_count,
        "car": tracker.count['car'],
        "bike": tracker.count['bike'],
        "bus": tracker.count['bus'],
        "truck": tracker.count['truck'],
        "density": density,
        "elapsed_seconds": round(elapsed, 1),
        "mode": "YOLO" if tracker.use_yolo else "SVM/MOG2"
    })


@app.route('/history')
def history():
    """Get historical stats from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM traffic_stats ORDER BY id DESC LIMIT 100')
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "id": row[0], "total": row[1], "car": row[2],
            "bike": row[3], "bus": row[4], "truck": row[5],
            "density": row[6], "timestamp": row[7]
        })
    return jsonify(result)


@app.route('/vehicles')
def vehicles():
    """Get logged vehicles from DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM vehicles ORDER BY id DESC LIMIT 200')
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "id": row[0], "vehicle_id": row[1], "type": row[2],
            "speed": row[3], "lane": row[4], "timestamp": row[5]
        })
    return jsonify(result)


@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    mode = data.get("mode", "svm")
    if mode == "yolo":
        tracker.enable_yolo(True)
    else:
        tracker.enable_yolo(False)
    return jsonify({"status": "success", "mode": mode})


@app.route('/set_video', methods=['POST'])
def set_video():
    """Allow uploading / switching video source."""
    global camera
    if 'video' in request.files:
        f = request.files['video']
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'dataset', 'uploaded_video.mp4')
        f.save(save_path)
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(save_path)
        return jsonify({"status": "success", "source": save_path})
    return jsonify({"status": "error", "message": "No file provided"}), 400


@app.route('/reset', methods=['POST'])
def reset():
    """Reset all counters."""
    global start_time
    tracker.tracked_vehicles.clear()
    tracker.total_count = 0
    tracker.count = {'car': 0, 'bike': 0, 'bus': 0, 'truck': 0}
    tracker.next_vehicle_id = 0
    start_time = time.time()
    return jsonify({"status": "reset"})


@app.route('/')
def index():
    return jsonify({
        "message": "Traffic Vehicle Counting & Classification API",
        "endpoints": [
            "/video_feed - MJPEG video stream",
            "/stats - Live statistics",
            "/history - Historical stats",
            "/vehicles - Logged vehicles",
            "/set_mode - POST {mode: 'svm'|'yolo'}",
            "/set_video - POST multipart video file",
            "/reset - POST reset counters"
        ]
    })


if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("  Traffic Vehicle Counting & Classification System")
    print("  Backend running at http://localhost:5000")
    print("  Video feed at http://localhost:5000/video_feed")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
