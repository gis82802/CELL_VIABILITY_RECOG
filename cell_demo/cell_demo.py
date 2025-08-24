import os
import tkinter as tk
from tkinter import ttk, messagebox
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "YOLO model")
IMAGE_DIR = os.path.join(PROJECT_DIR, "image files")

class CellCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("細胞計數 YOLO")
        self.root.geometry("1920x1080")

        # grid 排版
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 左邊顯示圖片
        self.image_label = tk.Label(root, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # 右邊控制區
        control_frame = tk.Frame(root, padx=20, pady=20)
        control_frame.grid(row=0, column=1, sticky="nsew")

        # 掃描 models/ 和 images/
        self.model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        self.image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".tif")]

        # 模型選單
        tk.Label(control_frame, text="選擇模型:", font=("Arial", 16)).pack(pady=10)
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(control_frame, textvariable=self.model_var, values=self.model_files, state="readonly", font=("Arial", 14))
        self.model_menu.pack(pady=10, fill="x")
        self.model_menu.bind("<<ComboboxSelected>>", self.update_image_menu)  # 綁定事件

        # 圖片選單
        tk.Label(control_frame, text="選擇圖片:", font=("Arial", 16)).pack(pady=10)
        self.image_var = tk.StringVar()
        self.image_menu = ttk.Combobox(control_frame, textvariable=self.image_var, state="readonly", font=("Arial", 14))
        self.image_menu.pack(pady=10, fill="x")

        # 執行按鈕
        self.run_button = tk.Button(control_frame, text="開始分析", font=("Arial", 18, "bold"), bg="green", fg="white", command=self.run_analysis)
        self.run_button.pack(pady=20, fill="x")

        # 結果標籤
        self.result_label = tk.Label(control_frame, text="細胞數量: 0", font=("Arial", 28, "bold"), fg="blue")
        self.result_label.pack(pady=40)

    def update_image_menu(self, event=None):
        """根據模型名稱過濾圖片清單"""
        model_name = self.model_var.get()
        if "_Green" in model_name:
            color = "G"
        elif "_Red" in model_name:
            color = "R"
        else:
            color = None

        if color:
            filtered = [f for f in self.image_files if f.endswith(f"_{color}.tif")]
        else:
            filtered = self.image_files

        self.image_menu["values"] = filtered
        self.image_var.set("")  # 重置選單

    def run_analysis(self):
        model_name = self.model_var.get()
        image_name = self.image_var.get()

        if not model_name or not image_name:
            messagebox.showwarning("警告", "請先選擇模型和圖片！")
            return

        # 載入模型與圖片
        model_path = os.path.join(MODEL_DIR, model_name)
        image_path = os.path.join(IMAGE_DIR, image_name)
        model = YOLO(model_path)
        image = cv2.imread(image_path)

        # 推論
        results = model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cell_count = len(boxes)

        # 畫框
        annotated = image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        save_path = os.path.join(IMAGE_DIR, "result_" + os.path.splitext(image_name)[0] + ".jpg")
        cv2.imwrite(save_path, annotated)

        # 更新數字
        self.result_label.config(text=f"細胞數量: {cell_count}")

        # 顯示圖片
        max_w, max_h = 1280, 1080
        h, w = annotated.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        annotated_resized = cv2.resize(annotated, (new_w, new_h))

        annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = CellCounterApp(root)
    root.mainloop()
