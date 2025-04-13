# src/attendance_logger.py

from datetime import datetime
import os

LOG_FILE = "attendance_log.csv"

# Tạo file CSV nếu chưa có
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,name,probability\n")

# attendance_logger.py

logged_names = set()

def log_attendance(name, probability):
    if name in logged_names:
        return False  # ❌ đã điểm danh rồi
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("attendance_log.csv", "a", encoding="utf-8") as f:
            f.write(f"{now},{name},{round(probability, 3)}\n")
        logged_names.add(name)
        return True  # ✅ điểm danh thành công
    except PermissionError:
        print("⚠ File bị khóa")
        return False

