import sqlite3
import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'traffic_data.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            type TEXT,
            speed REAL,
            lane TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_count INTEGER,
            car_count INTEGER,
            bike_count INTEGER,
            bus_count INTEGER,
            truck_count INTEGER,
            density TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_vehicle(vehicle_id, v_type, speed, lane):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO vehicles (vehicle_id, type, speed, lane) 
        VALUES (?, ?, ?, ?)
    ''', (vehicle_id, v_type, speed, lane))
    conn.commit()
    conn.close()

def log_stats(total, cars, bikes, buses, trucks, density):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO traffic_stats (total_count, car_count, bike_count, bus_count, truck_count, density)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (total, cars, bikes, buses, trucks, density))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print(f"Database initialized at {DB_PATH}")
