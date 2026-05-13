"""
Traffic Vehicle Tracker
========================
Core vehicle detection, tracking, counting, and classification engine.

Algorithms used:
- MOG2 Background Subtraction for isolating moving vehicles
- KLT (Kanade-Lucas-Tomasi) optical flow for motion tracking
- HOG + SVM for vehicle classification
- Optional YOLOv8 for deep learning comparison
"""

import cv2
import numpy as np
import os
import pickle
from database import log_vehicle, log_stats

class TrafficTracker:
    def __init__(self):
        # ----- MOG2 Background Subtractor -----
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # ----- SVM Model -----
        self.svm_model = None
        self.scaler = None
        self.use_yolo = False
        self.yolo_model = None
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'svm_model.pkl')
        scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'scaler.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
        # ----- HOG Descriptor -----
        self.hog = cv2.HOGDescriptor(
            (64, 128),   # winSize
            (16, 16),    # blockSize
            (8, 8),      # blockStride
            (8, 8),      # cellSize
            9            # nbins
        )

        # ----- Tracking State -----
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        self.count = {'car': 0, 'bike': 0, 'bus': 0, 'truck': 0}
        self.total_count = 0
        self.counting_line_y = 350   # Virtual counting line Y position
        self.frame_width = 800
        self.frame_height = 600
        
        # ----- KLT / Optical Flow Parameters -----
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.prev_gray = None
        self.prev_points = None

        # ----- Density -----
        self._last_density_ratio = 0.0
        self._last_density = "Low"
        
        # ----- Frame counter -----
        self.frame_idx = 0
        
        # ----- Stale track timeout -----
        self.max_missing_frames = 15

    def enable_yolo(self, use_yolo=True):
        """Switch between SVM/MOG2 and YOLO detection mode."""
        self.use_yolo = use_yolo
        if self.use_yolo and self.yolo_model is None:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO("yolov8n.pt")
                print("[INFO] YOLOv8 model loaded successfully.")
            except Exception as e:
                print("[ERROR] Failed to load YOLO:", e)
                self.use_yolo = False

    def classify_blob(self, frame, x, y, w, h):
        """
        Classify a detected blob (vehicle ROI) into car/bike/bus/truck.
        Uses SVM + HOG if model is loaded, otherwise falls back to heuristic.
        """
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return "car"
            
        # ---- SVM + HOG Classification ----
        if self.svm_model is not None:
            try:
                resized = cv2.resize(roi, (64, 128))
                gray_roi = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
                hog_features = self.hog.compute(gray_roi).flatten()
                
                # Geometric features
                area = w * h
                aspect_ratio = w / float(h) if h > 0 else 0
                geo = np.array([area, aspect_ratio, 0, 0, 0])
                combined = np.concatenate([hog_features, geo])
                
                if self.scaler:
                    combined = self.scaler.transform([combined])
                else:
                    combined = [combined]
                    
                prediction = self.svm_model.predict(combined)
                labels = ["car", "bike", "bus", "truck"]
                return labels[int(prediction[0])]
            except Exception:
                pass
                
        # ---- Fallback: Heuristic based on bounding box size ----
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 1
        
        if area > 25000:
            return "truck" if aspect_ratio > 1.3 else "bus"
        elif area > 12000:
            return "car"
        elif area < 4000:
            return "bike"
        else:
            return "car"

    def _compute_optical_flow(self, gray):
        """
        Compute sparse optical flow using KLT (Lucas-Kanade) tracker.
        Returns tracked points and their displacement vectors.
        """
        flow_vectors = []
        if self.prev_gray is not None and self.prev_points is not None and len(self.prev_points) > 0:
            new_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            if new_points is not None:
                good_old = self.prev_points[status.flatten() == 1]
                good_new = new_points[status.flatten() == 1]
                for old, new in zip(good_old, good_new):
                    dx = new[0] - old[0]
                    dy = new[1] - old[1]
                    flow_vectors.append((new[0], new[1], dx, dy))
        
        # Detect new feature points for next frame
        new_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        self.prev_points = new_pts
        
        return flow_vectors

    def process_frame(self, frame):
        """Process a single video frame: detect, track, count, classify vehicles."""
        if frame is None:
            return None
        
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ===== Background Subtraction (MOG2) =====
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        
        # ===== Density Calculation =====
        self._last_density_ratio = cv2.countNonZero(fg_mask) / float(self.frame_width * self.frame_height)
        if self._last_density_ratio > 0.3:
            self._last_density = "High"
        elif self._last_density_ratio > 0.1:
            self._last_density = "Medium"
        else:
            self._last_density = "Low"

        # ===== Optical Flow (KLT) =====
        flow_vectors = self._compute_optical_flow(gray)
        
        # Draw optical flow arrows (visualization)
        for (fx, fy, dx, dy) in flow_vectors:
            mag = np.sqrt(dx*dx + dy*dy)
            if mag > 2:  # Only show significant motion
                cv2.arrowedLine(frame, 
                    (int(fx - dx), int(fy - dy)), 
                    (int(fx), int(fy)),
                    (0, 255, 255), 1, tipLength=0.3)

        # ===== Mark all existing tracks as 'not seen this frame' =====
        for vid in self.tracked_vehicles:
            self.tracked_vehicles[vid]['seen'] = False

        # ===== Detection =====
        if self.use_yolo and self.yolo_model:
            # ----- YOLO Detection -----
            results = self.yolo_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                    v_type = None
                    if cls == 2: v_type = "car"
                    elif cls == 3: v_type = "bike"
                    elif cls == 5: v_type = "bus"
                    elif cls == 7: v_type = "truck"
                    
                    if v_type is None or conf < 0.3:
                        continue
                    
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    self._update_tracking(frame, x, y, w, h, v_type, gray)
        else:
            # ----- Classical: Contours from MOG2 -----
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Filter out very thin or very wide contours (noise)
                    aspect = w / float(h) if h > 0 else 0
                    if 0.2 < aspect < 5.0:
                        v_type = self.classify_blob(frame, x, y, w, h)
                        self._update_tracking(frame, x, y, w, h, v_type, gray)

        # ===== Cleanup stale tracks =====
        stale_ids = []
        for vid, vdata in self.tracked_vehicles.items():
            if not vdata.get('seen', False):
                vdata['missing'] = vdata.get('missing', 0) + 1
                if vdata['missing'] > self.max_missing_frames:
                    stale_ids.append(vid)
            else:
                vdata['missing'] = 0
        for sid in stale_ids:
            del self.tracked_vehicles[sid]

        # ===== Draw Counting Line =====
        cv2.line(frame, (0, self.counting_line_y), (self.frame_width, self.counting_line_y), (0, 0, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (self.frame_width - 200, self.counting_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # ===== Draw Lane Lines =====
        for lane_x in [200, 400, 600]:
            cv2.line(frame, (lane_x, 0), (lane_x, self.frame_height), (50, 50, 80), 1, cv2.LINE_AA)

        # ===== Overlay HUD =====
        # Semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Density color
        d_color = (0, 255, 0)
        if self._last_density == "High": d_color = (0, 0, 255)
        elif self._last_density == "Medium": d_color = (0, 200, 255)
        
        cv2.putText(frame, f"Density: {self._last_density}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, d_color, 2)
        cv2.putText(frame, f"Total Vehicles: {self.total_count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Cars: {self.count['car']}  Bikes: {self.count['bike']}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Buses: {self.count['bus']}  Trucks: {self.count['truck']}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        mode_text = "YOLO" if self.use_yolo else "SVM/MOG2"
        cv2.putText(frame, f"Mode: {mode_text}", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 200), 1)

        # ===== Periodic DB Logging =====
        if self.frame_idx % 30 == 0:
            log_stats(self.total_count, self.count['car'], self.count['bike'],
                      self.count['bus'], self.count['truck'], self._last_density)

        self.prev_gray = gray.copy()
        self.frame_idx += 1
        return frame

    def _update_tracking(self, frame, x, y, w, h, v_type, gray):
        """
        Match detections to existing tracks using centroid distance.
        Implements counting when a vehicle crosses the virtual line.
        """
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        
        matched_id = None
        min_dist = float('inf')
        
        # ---- Centroid-based matching with KLT refinement ----
        for vid, vdata in list(self.tracked_vehicles.items()):
            prev_cx, prev_cy = vdata['pos']
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if dist < 60 and dist < min_dist:
                min_dist = dist
                matched_id = vid
        
        # ---- New vehicle ----
        if matched_id is None:
            matched_id = self.next_vehicle_id
            self.next_vehicle_id += 1
            self.tracked_vehicles[matched_id] = {
                'pos': (cx, cy),
                'type': v_type,
                'counted': False,
                'frames': 0,
                'seen': True,
                'missing': 0,
                'speeds': []
            }
            
        vdata = self.tracked_vehicles[matched_id]
        prev_cy = vdata['pos'][1]
        
        # ---- Speed Estimation (pixel displacement per frame) ----
        speed = min_dist if min_dist != float('inf') else 0
        vdata['speeds'].append(speed)
        if len(vdata['speeds']) > 10:
            vdata['speeds'] = vdata['speeds'][-10:]
        avg_speed = np.mean(vdata['speeds']) if vdata['speeds'] else 0
        
        vdata['pos'] = (cx, cy)
        vdata['frames'] += 1
        vdata['seen'] = True
        
        # ---- Lane Detection ----
        lane_num = min(cx // 200 + 1, 4)
        lane = f"Lane {lane_num}"

        # ---- Counting Logic (crossing the virtual line) ----
        if prev_cy < self.counting_line_y <= cy and not vdata['counted']:
            vdata['counted'] = True
            self.total_count += 1
            self.count[v_type] += 1
            log_vehicle(matched_id, v_type, avg_speed, lane)
            
        # ---- Draw Bounding Box + Label ----
        if vdata['counted']:
            color = (0, 255, 0)   # Green if already counted
        else:
            color = (255, 180, 0)  # Cyan-ish if tracking
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        label = f"ID:{matched_id} {v_type}"
        speed_label = f"{avg_speed:.1f}px/f"
        
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y - 20), (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(frame, speed_label, (x, y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
