import cv2
import os

# 輸入資料夾與輸出資料夾
input_folder = "testimages"
output_folder = "images/results/binary"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 處理整個資料夾的圖片
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保持灰階

        if img is None:
            print(f"讀取失敗: {filename}")
            continue

        # --- 方法1：固定閾值 (Threshold) ---
        # 門檻值設 127，超過就是白(255)，否則就是黑(0)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # --- 方法2：自適應閾值 (Adaptive Threshold) ---
        # 對亮度不均的影像特別好用
        adaptive = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 儲存結果
        cv2.imwrite(os.path.join(output_folder, f"binary_{filename}"), binary)
        cv2.imwrite(os.path.join(output_folder, f"adaptive_{filename}"), adaptive)

        print(f"完成: {filename}")

print("全部圖片二值化完成 ✅")
