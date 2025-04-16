# 🧠 Face Attendance System using FaceNet + MTCNN + SVM

## 📌 Giới thiệu

Đây là dự án điểm danh sinh viên sử dụng công nghệ **nhận diện khuôn mặt**, được thực hiện bởi Nhóm 8 - Lớp 64HTTT1, Trường Đại học Thủy Lợi.  
Hệ thống sử dụng các kỹ thuật hiện đại bao gồm:

- Phát hiện khuôn mặt bằng **MTCNN**
- Căn chỉnh khuôn mặt bằng **landmark và Affine Transform**
- Trích xuất đặc trưng bằng **FaceNet**
- Phân loại bằng **SVM**
- Giao diện đồ họa đơn giản bằng `tkinter`

> 🎓 Đề tài được thực hiện trong khuôn khổ môn học *Khai phá dữ liệu - Tháng 4/2025*

---


## Cách chạy chương trình

# 1. Tạo môi trường ảo (khuyến khích)
python -m venv env
source env/bin/activate        # Trên Linux/macOS
env\Scripts\activate           # Trên Windows

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 3. Chạy giao diện người dùng (GUI)
python src/gui_attendance.py


