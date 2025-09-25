import cv2
import os

# 輸入資料夾與輸出資料夾
input_folder = "testimages"
output_folder = "images/results/edges"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 處理整個資料夾的圖片
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保持原始格式（灰階直接讀進來）

        if img is None:
            print(f"讀取失敗: {filename}")
            continue

        # Canny 邊緣偵測
        edges = cv2.Canny(img, threshold1=10, threshold2=50)

        # 儲存結果
        save_path = os.path.join(output_folder, f"edge_{filename}")
        cv2.imwrite(save_path, edges)
        print(f"完成: {filename} -> {save_path}")

print("全部圖片邊緣偵測完成 ✅")
