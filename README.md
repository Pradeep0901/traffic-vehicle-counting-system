# AI Traffic Vehicle Counting & Classification System

A real-time AI-based Traffic Vehicle Counting and Classification System using Image Processing techniques.

## 🚀 Features

- **Real-time Vehicle Detection** — Uses MOG2 background subtraction and contour analysis
- **KLT Optical Flow Tracking** — Sparse optical flow for robust vehicle tracking
- **SVM Classification** — Classifies vehicles into Car, Bike, Bus, and Truck categories
- **Virtual Line Counting** — Counts vehicles crossing a configurable virtual line
- **Live Dashboard** — Web-based dashboard with real-time charts and statistics
- **Java GUI** — Desktop application for live video monitoring and analytics

## 📁 Project Structure

```
traffic_system/
├── python_backend/    # Flask API, vehicle tracker, SVM trainer
│   ├── app.py         # Flask server & video feed
│   ├── tracker.py     # KLT tracker + MOG2 detection
│   ├── train_svm.py   # SVM model training script
│   └── requirements.txt
├── dashboard/         # Web-based analytics dashboard
│   ├── index.html
│   ├── style.css
│   └── script.js
├── java_gui/          # Java Swing desktop GUI
│   ├── pom.xml
│   └── src/
├── models/            # Trained SVM model files
├── dataset/           # Training dataset (not tracked)
└── docs/              # Documentation
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask, OpenCV |
| Detection | MOG2 Background Subtraction |
| Tracking | KLT Sparse Optical Flow |
| Classification | SVM (scikit-learn) |
| Dashboard | HTML, CSS, JavaScript, Chart.js |
| Desktop GUI | Java Swing |
| Database | SQLite |

## ⚡ Quick Start

### Python Backend
```bash
cd traffic_system/python_backend
pip install -r requirements.txt
python app.py
```
The backend runs at `http://localhost:5000`.

### Web Dashboard
Open `traffic_system/dashboard/index.html` in a browser, or serve it via the Flask backend.

### Java GUI
```bash
cd traffic_system/java_gui
mvn compile exec:java
```

## 📄 License

This project is for educational purposes.
