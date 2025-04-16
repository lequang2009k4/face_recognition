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

## 📂 Cấu trúc dự án

```bash
📁 dataset/                 # Dữ liệu ảnh thô
📁 aligned_data/           # Ảnh sau khi căn chỉnh
📁 embeddings/             # Vector đặc trưng từ FaceNet
📁 models/
    └── facemodel.pkl      # Mô hình SVM đã huấn luyện
📁 Models/
    └── 20180402-114759.pb # File model FaceNet pre-trained
📁 logs/                   # Log điểm danh
📁 gui/                    # Giao diện tkinter (main.py)
📄 align_dataset_mtcnn.py  # Script căn chỉnh khuôn mặt
📄 classifier.py           # Huấn luyện SVM
📄 detect_face.py          # Thư viện MTCNN
📄 facenet.py              # Load, tiền xử lý & nhúng
