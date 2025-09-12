import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
from openpyxl import Workbook
import torch
import numpy as np

# RealESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ====== 專案資料夾 ======
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
IMAGE_DIR = os.path.join(PROJECT_DIR, "images/image files")
RESULT_DIR = os.path.join(PROJECT_DIR, "images/results")
os.makedirs(RESULT_DIR, exist_ok=True)

ESRGAN_WEIGHTS = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")

# ====== 螢光亮度對比增強 ======
def enhance_contrast(image):
    # 1️⃣ CLAHE 自適應對比
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2️⃣ 可選：增強綠色 channel
    green = image[:,:,1]
    green = cv2.equalizeHist(green)
    image[:,:,1] = green

    return image

# ====== HSI 邊緣偵測 ======
def edge_from_hsi(image):
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    intensity = hsi[:,:,2].astype(np.uint8)
    edges = cv2.Canny(intensity, 50, 150)
    return edges

# ====== ESRGAN ======
class ESRGAN:
    def __init__(self, model_weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_weights = model_weights
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_weights,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=self.device
        )

    def enhance_single_image(self, input_path, output_path):
        img = cv2.imread(input_path)
        img = enhance_contrast(img)  # 亮度對比增強
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        sr_image, _ = self.upsampler.enhance(np.array(img_pil), outscale=4)
        sr_image = Image.fromarray(sr_image)
        sr_image.save(output_path)
        print(f"✅ 單張超解析完成: {output_path}")
        return cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)

# ====== GUI ======
class CellCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("細胞計數 YOLO + HSI I + ESRGAN")
        self.root.geometry("1920x1080")

        self.model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt") and "green" in f.lower()]
        self.image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_G.tif")]

        self.esrgan_model = ESRGAN(ESRGAN_WEIGHTS)

        # 左邊圖片區
        self.frame_left = tk.Frame(root, width=1400, height=1080, bg="black")
        self.frame_left.pack(side="left", fill="both", expand=True)
        self.image_label = tk.Label(self.frame_left, bg="black")
        self.image_label.pack(expand=True)

        # 右邊控制區
        self.frame_right = tk.Frame(root, width=520, height=1080)
        self.frame_right.pack(side="right", fill="y", padx=10, pady=10)

        tk.Label(self.frame_right, text="選擇模型:", font=("Arial", 16)).pack(pady=10)
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.frame_right, textvariable=self.model_var,
                                       values=self.model_files, state="readonly")
        self.model_menu.pack(pady=10)
        self.model_menu.bind("<<ComboboxSelected>>", self.update_image_list)

        tk.Label(self.frame_right, text="選擇圖片:", font=("Arial", 16)).pack(pady=10)
        self.image_var = tk.StringVar()
        self.image_menu = ttk.Combobox(self.frame_right, textvariable=self.image_var,
                                       values=self.image_files, state="readonly")
        self.image_menu.pack(pady=10)

        self.run_button = tk.Button(self.frame_right, text="單張分析", font=("Arial", 16),
                                    bg="green", fg="white", command=self.run_analysis)
        self.run_button.pack(pady=20, fill="x")

        self.plot_button = tk.Button(self.frame_right, text="統計輸出 Excel", font=("Arial", 16),
                                     bg="blue", fg="white", command=self.export_to_excel)
        self.plot_button.pack(pady=20, fill="x")

        self.result_label = tk.Label(self.frame_right, text="", font=("Arial", 24), fg="red")
        self.result_label.pack(pady=20)

    def update_image_list(self, event=None):
        self.image_menu["values"] = self.image_files
        self.image_var.set("")

    def enhance_with_esrgan(self, img_path, esrgan_file):
        if os.path.exists(esrgan_file):
            return cv2.imread(esrgan_file)
        else:
            return self.esrgan_model.enhance_single_image(img_path, esrgan_file)

    def run_analysis(self):
        model_name = self.model_var.get()
        image_name = self.image_var.get()
        if not model_name or not image_name:
            messagebox.showwarning("警告", "請先選擇模型和圖片！")
            return

        model = YOLO(os.path.join(MODEL_DIR, model_name))
        img_path = os.path.join(IMAGE_DIR, image_name)
        esrgan_file = os.path.join(RESULT_DIR, f"esrgan_{os.path.splitext(image_name)[0]}.jpg")
        result_file = os.path.join(RESULT_DIR, f"result_{os.path.splitext(image_name)[0]}.jpg")

        # 亮度對比增強 + ESRGAN
        enhanced_img = self.enhance_with_esrgan(img_path, esrgan_file)

        # HSI I 通道邊緣
        edges = edge_from_hsi(enhanced_img)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # YOLO 偵測
        results = model(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        cell_count = len(boxes)
        annotated = edges_bgr.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(result_file, annotated)

        self.result_label.config(text=f"細胞數量: {cell_count}")

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_rgb)
        img_pil.thumbnail((1400, 1000))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def export_to_excel(self):
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("警告", "請先選擇模型！")
            return

        model = YOLO(os.path.join(MODEL_DIR, model_name))
        image_files_sorted = sorted(
            self.image_files,
            key=lambda f: datetime.strptime(f.split("_")[0] + "_" + f.split("_")[1], "%Y%m%d_%H%M%S")
        )
        times, counts = [], []

        for img_file in image_files_sorted:
            esrgan_file = os.path.join(RESULT_DIR, f"esrgan_{os.path.splitext(img_file)[0]}.jpg")
            result_file = os.path.join(RESULT_DIR, f"result_{os.path.splitext(img_file)[0]}.jpg")
            enhanced_img = self.enhance_with_esrgan(os.path.join(IMAGE_DIR, img_file), esrgan_file)

            results = model(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cell_count = len(boxes)
            counts.append(cell_count)
            times.append(datetime.strptime(img_file.split("_")[0] + "_" + img_file.split("_")[1], "%Y%m%d_%H%M%S"))

        wb = Workbook()
        ws = wb.active
        ws.title = "Cell Count"
        ws.append(["Time", "Cell Count"])
        for t, c in zip(times, counts):
            ws.append([t.strftime("%Y-%m-%d %H:%M:%S"), c])
        save_path = os.path.join(PROJECT_DIR, "cell_counts.xlsx")
        wb.save(save_path)
        messagebox.showinfo("完成", f"已輸出 Excel 檔案：\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CellCounterApp(root)
    root.mainloop()
