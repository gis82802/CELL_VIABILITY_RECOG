import cv2
import os

# 輸入資料夾與輸出資料夾
input_folder = "testimages"
output_folder = "images/results/bilateral_filtered"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 讀取資料夾內所有圖片
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"讀取失敗: {filename}")
            continue

        # 雙邊濾波 (Bilateral Filter)
        # d: 鄰域直徑 (越大越平滑)，通常 5~15
        # sigmaColor: 顏色空間的標準差，越大越模糊 (建議 50~150)
        # sigmaSpace: 座標空間的標準差，越大影響越遠 (建議 50~150)
        filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        # 儲存結果
        save_path = os.path.join(output_folder, f"bilateral_{filename}")
        cv2.imwrite(save_path, filtered)
        print(f"處理完成: {filename} -> {save_path}")

print("全部圖片處理完成 ✅")
