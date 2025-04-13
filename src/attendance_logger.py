from datetime import datetime
import os

LOG_FILE = "attendance_log.csv"

# Tạo file CSV nếu chưa có
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("timestamp,name,probability,status\n")

logged_names = set()

def log_attendance(name, probability, status):
    key = f"{name}_{status}"
    if key in logged_names:
        return False  # Đã điểm danh trạng thái đó
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now},{name},{round(probability, 3)},{status}\n")
        logged_names.add(key)
        return True
    except PermissionError:
        print("⚠ File bị khóa")
        return False

