
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import threading
from datetime import datetime
import os
import subprocess
import numpy as np
import pickle
import tensorflow as tf
import facenet
import align.detect_face
from attendance_logger import log_attendance

tf.compat.v1.disable_eager_execution()

# Cấu hình
FACENET_MODEL = "Models/20180402-114759.pb"
CLASSIFIER_PATH = "Models/facemodel.pkl"
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Điểm danh nhân viên - AI")

        self.label = tk.Label(root, text="Camera", font=("Arial", 16))
        self.label.pack()

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.info = tk.Label(root, text="", font=("Arial", 12), fg="green")
        self.info.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.btn_log = tk.Button(self.btn_frame, text="Xem Log", command=self.open_log)
        self.btn_log.grid(row=0, column=0, padx=5)

        self.btn_add = tk.Button(self.btn_frame, text="Thêm nhân viên", command=self.add_person)
        self.btn_add.grid(row=0, column=1, padx=5)

        self.btn_train = tk.Button(self.btn_frame, text="Huấn luyện lại", command=self.run_align_and_train)
        self.btn_train.grid(row=0, column=2, padx=5)

        self.btn_checkin = tk.Button(self.btn_frame, text="Điểm danh vào", command=lambda: self.set_attendance_status("in"))
        self.btn_checkin.grid(row=0, column=3, padx=5)

        self.btn_checkout = tk.Button(self.btn_frame, text="Điểm danh ra", command=lambda: self.set_attendance_status("out"))
        self.btn_checkout.grid(row=0, column=4, padx=5)

        self.attendance_status = "in"
        self.recognizing_person = None
        self.recognizing_start_time = None

        self.running = False
        self.cap = None

        self.load_model()

    def set_attendance_status(self, status):
        self.attendance_status = status
        self.info.config(text=f"Chế độ: Điểm danh {'vào' if status == 'in' else 'ra'}")
        if not self.running:
            self.running = True
            self.start_video()

    def load_model(self):
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        with self.sess.as_default():
            facenet.load_model(FACENET_MODEL)
            self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]
            self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, "src/align")

        with open(CLASSIFIER_PATH, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def start_video(self):
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.update_frame, daemon=True).start()

    def update_frame(self):
        self.recognizing_person = None
        self.recognizing_start_time = None

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                bounding_boxes, _ = align.detect_face.detect_face(
                    frame_rgb, MINSIZE, self.pnet, self.rnet, self.onet, THRESHOLD, FACTOR
                )

                if len(bounding_boxes) > 0:
                    for bb in bounding_boxes:
                        x1, y1, x2, y2 = map(int, bb[:4])
                        face = frame_rgb[y1:y2, x1:x2]

                        if face.shape[0] < 20 or face.shape[1] < 20:
                            continue

                        face_resized = cv2.resize(face, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
                        face_normalized = facenet.prewhiten(face_resized)

                        feed_dict = {
                            self.images_placeholder: face_normalized.reshape(1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3),
                            self.phase_train_placeholder: False
                        }

                        emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
                        predictions = self.model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[0, best_class_indices]
                        name = self.class_names[best_class_indices[0]]
                        prob = best_class_probabilities[0]

                        if prob > 0.8:
                            if self.recognizing_person == name:
                                elapsed_time = (datetime.now() - self.recognizing_start_time).total_seconds()
                                self.info.config(text=f"Đang xác thực {name}... {int(elapsed_time)}s")

                                if elapsed_time >= 2:
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    success = log_attendance(name, prob, self.attendance_status)

                                    if success:
                                        self.info.config(
                                            text=f"{name} đã điểm danh {self.attendance_status.upper()} lúc {timestamp}"
                                        )
                                        messagebox.showinfo(
                                            "Thành công",
                                            f"Điểm danh {self.attendance_status.upper()} thành công cho {name} lúc {timestamp}"
                                        )
                                    else:
                                        self.info.config(
                                            text=f"{name} đã điểm danh {self.attendance_status.upper()} trước đó"
                                        )

                                    if self.cap:
                                        self.running = False
                                        self.cap.release()
                                        return
                            else:
                                self.recognizing_person = name
                                self.recognizing_start_time = datetime.now()
                        else:
                            self.recognizing_person = None
                            self.recognizing_start_time = None
                            self.info.config(text="Không xác định")
                else:
                    self.recognizing_person = None
                    self.recognizing_start_time = None
                    self.info.config(text="Đang chờ khuôn mặt...")

            except Exception as e:
                print("⚠️", e)

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.imgtk = img
            self.canvas.configure(image=img)

    def open_log(self):
        if os.name == 'nt':
            os.startfile("attendance_log.csv")
        else:
            subprocess.call(['xdg-open', "attendance_log.csv"])

    def add_person(self):
        name = simpledialog.askstring("Thêm nhân viên", "Nhập tên nhân viên:")
        if not name:
            return
        save_dir = os.path.join("Dataset", "FaceData", "raw", name.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(0)
        count = 0
        while count < 10:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{name} - SPACE để chụp {count+1}/10", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Thêm nhân viên", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                img_path = os.path.join(save_dir, f"{count+1}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Hoàn tất", f"Đã chụp {count} ảnh cho {name}.\nNhấn nút 'Huấn luyện lại' để cập nhật model.")

    def run_align_and_train(self):
        try:
            if self.running:
                self.running = False
                self.cap.release()
            messagebox.showinfo("Đang xử lý", "Bắt đầu căn chỉnh và huấn luyện lại model. Vui lòng đợi...")
            subprocess.run(["python", "src/align_dataset_mtcnn.py", "Dataset/FaceData/raw", "Dataset/FaceData/processed", "--image_size", "160", "--margin", "32", "--random_order", "--gpu_memory_fraction", "0.25"], check=True)
            subprocess.run(["python", "src/classifier.py", "TRAIN", "Dataset/FaceData/processed", "Models/20180402-114759.pb", "Models/facemodel.pkl", "--batch_size", "1000"], check=True)
            messagebox.showinfo("Thành công", "Đã huấn luyện lại mô hình thành công!")
            self.load_model()
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Lỗi", f"Có lỗi khi huấn luyện: {e}")

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
