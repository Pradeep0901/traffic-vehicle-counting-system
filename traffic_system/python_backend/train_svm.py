"""
SVM Model Training Script for Vehicle Classification
=====================================================
This script trains a Support Vector Machine classifier to classify vehicles
into 4 categories: car, bike, bus, truck.

It uses HOG (Histogram of Oriented Gradients) features extracted from 
vehicle ROI images.

Usage:
    1. Create a training dataset folder structure:
       dataset/train/car/    - images of cars
       dataset/train/bike/   - images of bikes
       dataset/train/bus/    - images of buses
       dataset/train/truck/  - images of trucks

    2. Run: python train_svm.py

    3. The trained model is saved as models/svm_model.pkl
"""

import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset', 'train')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Vehicle classes
CLASSES = ['car', 'bike', 'bus', 'truck']

# HOG parameters
HOG_WIN_SIZE = (64, 128)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9


def create_hog_descriptor():
    """Create and return an HOG descriptor with specified parameters."""
    hog = cv2.HOGDescriptor(
        HOG_WIN_SIZE,
        HOG_BLOCK_SIZE,
        HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE,
        HOG_NBINS
    )
    return hog


def extract_hog_features(image, hog):
    """Extract HOG features from a single image."""
    resized = cv2.resize(image, HOG_WIN_SIZE)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = hog.compute(resized)
    return features.flatten()


def extract_geometric_features(image):
    """
    Extract geometric/shape features from the vehicle image.
    These features supplement HOG for better classification.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    area = h * w
    aspect_ratio = w / float(h) if h > 0 else 0

    # Contour-based features
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter > 0 else 0
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
    else:
        contour_area = 0
        circularity = 0
        solidity = 0

    return np.array([area, aspect_ratio, contour_area, circularity, solidity])


def load_dataset():
    """Load images from the dataset directory and extract features."""
    features = []
    labels = []
    hog = create_hog_descriptor()

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  [WARNING] Directory not found: {class_dir}")
            continue

        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  Found {len(image_files)} images for class '{class_name}'")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Extract HOG features
            hog_feat = extract_hog_features(image, hog)

            # Extract geometric features
            geo_feat = extract_geometric_features(image)

            # Combine feature vectors
            combined = np.concatenate([hog_feat, geo_feat])
            features.append(combined)
            labels.append(class_idx)

    return np.array(features), np.array(labels)


def generate_synthetic_data(n_per_class=200):
    """
    Generate synthetic training data when real images are not available.
    Creates random vehicle-like ROI images for each class.
    """
    print("\n  Generating synthetic training data...")
    features = []
    labels = []
    hog = create_hog_descriptor()

    # Characteristic sizes (w, h) for each vehicle type
    class_params = {
        0: {'w_range': (60, 120), 'h_range': (40, 80), 'name': 'car'},      # car
        1: {'w_range': (20, 50),  'h_range': (30, 70), 'name': 'bike'},     # bike
        2: {'w_range': (80, 160), 'h_range': (60, 140), 'name': 'bus'},     # bus
        3: {'w_range': (100, 180), 'h_range': (50, 100), 'name': 'truck'},  # truck
    }

    for class_idx, params in class_params.items():
        for _ in range(n_per_class):
            w = np.random.randint(*params['w_range'])
            h = np.random.randint(*params['h_range'])

            # Create a synthetic image with random patterns
            img = np.random.randint(50, 200, (max(h, 10), max(w, 10), 3), dtype=np.uint8)

            # Add class-specific patterns
            if class_idx == 0:  # car: small rectangle
                cv2.rectangle(img, (5, 5), (w-5, h-5), (100, 100, 200), -1)
            elif class_idx == 1:  # bike: thin vertical
                cv2.line(img, (w//2, 0), (w//2, h), (200, 100, 100), 3)
            elif class_idx == 2:  # bus: large filled rectangle
                cv2.rectangle(img, (2, 2), (w-2, h-2), (50, 150, 50), -1)
            elif class_idx == 3:  # truck: large with cabin
                cv2.rectangle(img, (2, 2), (w-2, h-2), (150, 150, 50), -1)
                cv2.rectangle(img, (w//2, 2), (w-2, h//2), (200, 200, 100), -1)

            # Add some noise
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

            hog_feat = extract_hog_features(img, hog)
            geo_feat = extract_geometric_features(img)
            combined = np.concatenate([hog_feat, geo_feat])
            features.append(combined)
            labels.append(class_idx)

    print(f"  Generated {len(features)} synthetic samples.")
    return np.array(features), np.array(labels)


def train_model(X, y):
    """Train an SVM classifier."""
    print("\n  Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  Training SVM classifier (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = svm.predict(X_test_scaled)
    print("\n" + "=" * 50)
    print("  CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    print("  CONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = np.mean(y_pred == y_test) * 100
    print(f"\n  Test Accuracy: {accuracy:.2f}%")

    return svm, scaler


def save_model(svm, scaler):
    """Save the trained model and scaler to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(svm, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Scaler saved to: {SCALER_PATH}")


def main():
    print("=" * 60)
    print("  SVM Vehicle Classification - Training Script")
    print("=" * 60)

    # Try loading real dataset first
    if os.path.exists(DATASET_DIR):
        print(f"\n  Loading dataset from: {DATASET_DIR}")
        X, y = load_dataset()
    else:
        X, y = np.array([]), np.array([])

    # If not enough real data, use synthetic
    if len(X) < 100:
        print("\n  Insufficient real training data.")
        X_syn, y_syn = generate_synthetic_data(n_per_class=300)
        if len(X) > 0:
            X = np.vstack([X, X_syn])
            y = np.concatenate([y, y_syn])
        else:
            X, y = X_syn, y_syn

    print(f"\n  Total samples: {len(X)}")
    print(f"  Feature vector size: {X.shape[1]}")
    print(f"  Classes: {CLASSES}")

    # Train
    svm, scaler = train_model(X, y)

    # Save
    save_model(svm, scaler)

    print("\n" + "=" * 60)
    print("  Training complete! You can now run the main application.")
    print("=" * 60)


if __name__ == '__main__':
    main()
