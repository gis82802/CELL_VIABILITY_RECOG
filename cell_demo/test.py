import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
from datetime import datetime
from openpyxl import Workbook
from basicsr.archs.rrdbnet_arch import RRDBNet

# 專案資料夾
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
IMAGE_DIR = os.path.join(PROJECT_DIR, "images/image files")
RESULT_DIR = os.path.join(PROJECT_DIR, "images/results")  # 存匡過圖片的資料夾
os.makedirs(RESULT_DIR, exist_ok=True)
ESRGAN_WEIGHTS = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")  # ESRGAN 權重路徑

# ====== ESRGAN ======
class ESRGAN:
    def __init__(self, model_path):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=4)
        self.model.eval()
        self.load_weights(model_path)

    def load_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        else:
            state_dict = checkpoint
        model_dict = self.model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.model.load_state_dict(filtered_dict, strict=False)
        print("ESRGAN 權重載入完成！")

    def enhance(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0)/255.0
        with torch.no_grad():
            output = self.model(tensor)
        output_img = output.squeeze().permute(1,2,0).clamp(0,1).numpy()
        return (output_img*255).astype("uint8")

# ====== GUI ======
class CellCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("細胞計數 YOLO - Green")
        self.root.geometry("1920x1080")

        # 掃描 models/ 和 images/
        self.model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt") and "green" in f.lower()]
        self.image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_G.tif")]

        # 初始化 ESRGAN
        self.esrgan_model = ESRGAN(ESRGAN_WEIGHTS)

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

        # 統計輸出按鈕
        self.plot_button = tk.Button(self.frame_right, text="統計輸出 Excel", font=("Arial", 16), bg="blue", fg="white", command=self.export_to_excel)
        self.plot_button.pack(pady=20, fill="x")

        # 結果標籤
        self.result_label = tk.Label(self.frame_right, text="", font=("Arial", 24), fg="red")
        self.result_label.pack(pady=20)

    def update_image_list(self, event=None):
        self.image_menu["values"] = self.image_files
        self.image_var.set("")

    def enhance_with_esrgan(self, img, esrgan_file):
        if os.path.exists(esrgan_file):
            return cv2.imread(esrgan_file)
        else:
            enhanced = self.esrgan_model.enhance(img)
            cv2.imwrite(esrgan_file, enhanced)
            return enhanced

    def run_analysis(self):
        model_name = self.model_var.get()
        image_name = self.image_var.get()
        if not model_name or not image_name:
            messagebox.showwarning("警告", "請先選擇模型和圖片！")
            return

        model = YOLO(os.path.join(MODEL_DIR, model_name))
        img_path = os.path.join(IMAGE_DIR, image_name)

        # 檔名
        esrgan_file = os.path.join(RESULT_DIR, "esrgan_" + os.path.splitext(image_name)[0] + ".jpg")
        result_file = os.path.join(RESULT_DIR, "result_" + os.path.splitext(image_name)[0] + ".jpg")

        # ESRGAN 增強（如果已存在就讀取）
        image = cv2.imread(img_path)
        image = self.enhance_with_esrgan(image, esrgan_file)

        # YOLO 分析（如果已存在就讀取）
        if os.path.exists(result_file):
            annotated = cv2.imread(result_file)
            cell_count = len(model.predict(annotated)[0].boxes)
        else:
            results = model.predict(image)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cell_count = len(boxes)
            annotated = image.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(result_file, annotated)

        self.result_label.config(text=f"細胞數量: {cell_count}")

        # 顯示圖片
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
        image_files_sorted = sorted(self.image_files, key=lambda f: datetime.strptime(f.split("_")[0]+"_"+f.split("_")[1], "%Y%m%d_%H%M%S"))
        times = []
        counts = []

        for img_file in image_files_sorted:
            esrgan_file = os.path.join(RESULT_DIR, "esrgan_" + os.path.splitext(img_file)[0] + ".jpg")
            result_file = os.path.join(RESULT_DIR, "result_" + os.path.splitext(img_file)[0] + ".jpg")

            # 先處理 ESRGAN
            if os.path.exists(esrgan_file):
                enhanced = cv2.imread(esrgan_file)
            else:
                img_path = os.path.join(IMAGE_DIR, img_file)
                image = cv2.imread(img_path)
                enhanced = self.enhance_with_esrgan(image, esrgan_file)

            # 再處理 YOLO
            if os.path.exists(result_file):
                annotated = cv2.imread(result_file)
                cell_count = len(model.predict(annotated)[0].boxes)
            else:
                results = model.predict(enhanced)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                cell_count = len(boxes)
                annotated = enhanced.copy()
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(result_file, annotated)

            counts.append(cell_count)
            times.append(datetime.strptime(img_file.split("_")[0]+"_"+img_file.split("_")[1], "%Y%m%d_%H%M%S"))

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
