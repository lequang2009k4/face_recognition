# src/attendance_logger.py

from datetime import datetime
import os

LOG_FILE = "attendance_log.csv"

# Tạo file CSV nếu chưa có
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,name,probability\n")

def log_attendance(name, probability):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{now},{name},{round(probability, 3)}\n")
