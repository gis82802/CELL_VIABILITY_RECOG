import os
import tkinter as tk
from tkinter import ttk, messagebox
from ultralytics import YOLO
from openpyxl import Workbook
from datetime import datetime
import cv2

# ====== 專案資料夾 ======
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
IMAGE_DIR = os.path.join(PROJECT_DIR, "testimages")
RESULT_DIR = os.path.join(PROJECT_DIR, "images/results/yolo")
os.makedirs(RESULT_DIR, exist_ok=True)

# ====== ESRGAN (已註解掉) ======
"""
class ESRGAN:
    def __init__(self, model_weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_weights = model_weights
        # 初始化模型...

    def enhance_single_image(self, input_path, output_path):
        # ESRGAN 增強程式碼...
        ...
"""

# ====== YOLO 偵測 ======
def run_yolo(model_file, image_file, save_annotated=True):
    model_path = os.path.join(MODEL_DIR, model_file)
    img_path = os.path.join(IMAGE_DIR, image_file)
    
    model = YOLO(model_path)
    results = model(img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    print(f"✅ {image_file} 偵測完成，共 {len(boxes)} 個物件")

    # 標註圖片並存檔
    if save_annotated:
        img = cv2.imread(img_path)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_path = os.path.join(RESULT_DIR, f"result_{image_file}")
        cv2.imwrite(save_path, img)
        print(f"✅ 已存標註圖片: {save_path}")

    return len(boxes)

# ====== Excel 匯出 ======
def export_counts_to_excel(model_file):
    image_files_sorted = sorted(
        [f for f in os.listdir(IMAGE_DIR) if f.endswith("_G.tif")],
        key=lambda f: datetime.strptime(f.split("_")[0] + "_" + f.split("_")[1], "%Y%m%d_%H%M%S")
    )

    counts, times = [], []
    for img_file in image_files_sorted:
        cell_count = run_yolo(model_file, img_file)
        counts.append(cell_count)
        times.append(datetime.strptime(img_file.split("_")[0] + "_" + img_file.split("_")[1], "%Y%m%d_%H%M%S"))

    # 匯出 Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Cell Count"
    ws.append(["Time", "Cell Count"])
    for t, c in zip(times, counts):
        ws.append([t.strftime("%Y-%m-%d %H:%M:%S"), c])
    save_path = os.path.join(PROJECT_DIR, "cell_counts.xlsx")
    wb.save(save_path)
    messagebox.showinfo("完成", f"✅ 已輸出 Excel 檔案：\n{save_path}")

# ====== GUI ======
class CellCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("細胞計數 YOLO")
        self.root.geometry("600x400")

        self.model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]

        tk.Label(root, text="選擇模型:", font=("Arial", 14)).pack(pady=10)
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(root, textvariable=self.model_var,
                                       values=self.model_files, state="readonly")
        self.model_menu.pack(pady=10)

        self.run_button = tk.Button(root, text="統計輸出 Excel", font=("Arial", 14),
                                    bg="blue", fg="white", command=self.run_export)
        self.run_button.pack(pady=20, fill="x")

    def run_export(self):
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("警告", "請先選擇模型！")
            return
        export_counts_to_excel(model_name)

if __name__ == "__main__":
    root = tk.Tk()
    app = CellCounterApp(root)
    root.mainloop()
