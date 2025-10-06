import os
import cv2
import numpy as np
from openpyxl import Workbook

# ====== 設定資料夾路徑 ======
INPUT_DIR = "testimages"         # 換成你的資料夾路徑
OUTPUT_DIR = "images/results/DoG"   # 輸出 DoG 圖片資料夾
OUTPUT_EXCEL = "images/results/DoG_results.xlsx"  # 輸出 Excel 檔案

# 建立輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 建立 Excel 工作簿 ======
wb = Workbook()
ws = wb.active
ws.title = "DoG_Result"
ws.append(["檔名", "灰階平均值"])  # 標題列

# ====== 處理影像 ======
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        filepath = os.path.join(INPUT_DIR, filename)

        # 讀取影像 (灰階)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"無法讀取: {filename}")
            continue

        # ====== DoG (Difference of Gaussians) ======
        blur1 = cv2.GaussianBlur(img, (5, 5), 1)   # sigma=1
        blur2 = cv2.GaussianBlur(img, (5, 5), 2)   # sigma=2
        dog = cv2.subtract(blur1, blur2)

        # ====== 計算平均灰階 ======
        mean_gray = np.mean(dog)

        # 寫入 Excel
        ws.append([filename, round(float(mean_gray), 2)])

        # ====== 輸出 DoG 圖片 ======
        save_path = os.path.join(OUTPUT_DIR, f"DoG_{filename}")
        cv2.imwrite(save_path, dog)

        print(f"{filename} → 平均灰階值: {mean_gray:.2f} → 已存 {save_path}")

# ====== 存 Excel ======
wb.save(OUTPUT_EXCEL)
print(f"\n處理完成！結果已存到 {OUTPUT_EXCEL}，圖片存到 {OUTPUT_DIR}")
