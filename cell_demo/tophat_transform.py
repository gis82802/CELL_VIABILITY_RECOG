import cv2 as cv
import numpy as np
import os

# --- CLAHE 參數 ---
clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(16, 16))  # 保留你剛剛的設定

# --- TopHat 與 Gamma 參數 ---
kernel = np.ones((3, 3), np.uint8)   # 3x3 kernel
gamma_value = 1.2                    # gamma 值，大於1變亮，小於1變暗

# 建立 Gamma 校正查表
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma_value) * 255.0, 0, 255)

def enhance_contrast(input_path, output_path): 
    image = cv.imread(input_path)  # 讀取彩色圖
    if image is None:
        print(f"❌ 無法讀取圖像：{input_path}")
        return
    
    # 轉 RGB 並取 G 通道
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_channel = img_rgb[:,:,1]

    # 步驟1：通道強度正規化
    img_channel = cv.normalize(img_channel, None, 0, 255, cv.NORM_MINMAX)

    # 步驟2：CLAHE 增強
    img_clahe = clahe.apply(img_channel)

    # 步驟3：TopHat
    tophat = cv.morphologyEx(img_clahe, cv.MORPH_TOPHAT, kernel)

    # 步驟4：融合 CLAHE 與 TopHat (1.5 : -0.5)
    fused = cv.addWeighted(img_clahe, 1.5, tophat, -0.5, 0)

    # 步驟5：Gamma 校正
    gamma_corrected = cv.LUT(fused, lookUpTable)

    # 步驟6：銳化 (Unsharp Mask)
    blurred = cv.GaussianBlur(gamma_corrected, (3, 3), 0)   # 模糊版本
    sharpened = cv.addWeighted(gamma_corrected, 1.5, blurred, -0.5, 0)

    # 步驟7：模糊 (Gaussian Blur)
    final_blur = cv.GaussianBlur(sharpened, (3, 3), 0)

    # 步驟8：合併回彩色圖 (只保留 G 通道)
    result = cv.merge([
        np.zeros_like(final_blur),  # R
        final_blur,                 # G
        np.zeros_like(final_blur)   # B
    ])

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, result)  # 儲存結果
    print(f"✅ 已處理完成並儲存：{output_path}")

def batch_enhance(input_folder, output_folder):
    exts = ('.jpg', '.png', '.bmp', '.tif')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    if not files:
        print("⚠️ 找不到任何圖像檔案")
        return

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        enhance_contrast(input_path, output_path)

if __name__ == "__main__":
    INPUT_FOLDER = "testimages"
    OUTPUT_FOLDER = "images/results/tophat"
    batch_enhance(INPUT_FOLDER, OUTPUT_FOLDER)
