import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
# 專案資料夾
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "YOLO model")
IMAGE_DIR = os.path.join(PROJECT_DIR, "image files")

class CellCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("細胞計數 YOLO - Green")
        self.root.geometry("1920x1080")

        # 掃描 models/ 和 images/
        self.model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt") and "Green" in f]
        self.image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_G.tif")]

        # 左邊：圖片顯示
        self.frame_left = tk.Frame(root, width=1400, height=1080, bg="black")
        self.frame_left.pack(side="left", fill="both", expand=True)
        self.image_label = tk.Label(self.frame_left, bg="black")
        self.image_label.pack(expand=True)

        # 右邊：控制區
        self.frame_right = tk.Frame(root, width=520, height=1080)
        self.frame_right.pack(side="right", fill="y", padx=10, pady=10)

        # 模型選單
        tk.Label(self.frame_right, text="選擇模型:", font=("Arial", 16)).pack(pady=10)
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.frame_right, textvariable=self.model_var, values=self.model_files, state="readonly")
        self.model_menu.pack(pady=10)
        self.model_menu.bind("<<ComboboxSelected>>", self.update_image_list)

        # 圖片選單
        tk.Label(self.frame_right, text="選擇圖片:", font=("Arial", 16)).pack(pady=10)
        self.image_var = tk.StringVar()
        self.image_menu = ttk.Combobox(self.frame_right, textvariable=self.image_var, values=self.image_files, state="readonly")
        self.image_menu.pack(pady=10)

        # 單張分析按鈕
        self.run_button = tk.Button(self.frame_right, text="單張分析", font=("Arial", 16), bg="green", fg="white", command=self.run_analysis)
        self.run_button.pack(pady=20, fill="x")

        # 統計折線圖按鈕
        self.plot_button = tk.Button(self.frame_right, text="統計折線圖", font=("Arial", 16), bg="blue", fg="white", command=self.plot_time_series)
        self.plot_button.pack(pady=20, fill="x")

        # 結果標籤
        self.result_label = tk.Label(self.frame_right, text="", font=("Arial", 24), fg="red")
        self.result_label.pack(pady=20)

    def update_image_list(self, event=None):
        """只保留綠色圖片"""
        self.image_menu["values"] = self.image_files
        self.image_var.set("")

    def run_analysis(self):
        model_name = self.model_var.get()
        image_name = self.image_var.get()
        if not model_name or not image_name:
            messagebox.showwarning("警告", "請先選擇模型和圖片！")
            return

        model_path = os.path.join(MODEL_DIR, model_name)
        image_path = os.path.join(IMAGE_DIR, image_name)
        model = YOLO(model_path)
        image = cv2.imread(image_path)

        results = model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cell_count = len(boxes)

        annotated = image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        save_path = os.path.join(IMAGE_DIR, "result_" + os.path.splitext(image_name)[0] + ".jpg")
        cv2.imwrite(save_path, annotated)

        self.result_label.config(text=f"細胞數量: {cell_count}")

        # 顯示圖片
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_rgb)
        img_pil.thumbnail((1400, 1000))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def plot_time_series(self):
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("警告", "請先選擇模型！")
            return

        # 載入模型
        model_path = os.path.join(MODEL_DIR, model_name)
        model = YOLO(model_path)

        # 排序圖片
        image_files_sorted = sorted(self.image_files, key=lambda f: datetime.strptime(f.split("_")[0]+"_"+f.split("_")[1], "%Y%m%d_%H%M%S"))

        times = []
        counts = []
        for img_file in image_files_sorted:
            img_path = os.path.join(IMAGE_DIR, img_file)
            img = cv2.imread(img_path)
            results = model(img)
            cell_count = len(results[0].boxes)
            counts.append(cell_count)
            times.append(datetime.strptime(img_file.split("_")[0]+"_"+img_file.split("_")[1], "%Y%m%d_%H%M%S"))

        # 畫折線圖
        plt.figure(figsize=(12,6))
        plt.plot(times, counts, marker='o', linestyle='-', color='green')
        plt.xlabel("time")
        plt.ylabel("cell count")
        plt.title("Green cell count over time")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CellCounterApp(root)
    root.mainloop()
