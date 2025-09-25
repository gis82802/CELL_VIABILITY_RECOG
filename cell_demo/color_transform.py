import cv2 as cv
import numpy as np
import os

# 使用 CLAHE 方法增強圖像對比度
clahe = cv.createCLAHE(clipLimit=20.0, tileGridSize=(16, 16))# 創建 CLAHE 物件 tileGridSize=(8, 8)表示將圖像分成8x8的區塊進行對比度限制
def enhance_contrast(input_path, output_path): 
    image = cv.imread(input_path) #讀取圖像
    if image is None:
        print(f"❌ 無法讀取圖像：{input_path}")
        return
    
    #img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 將圖像轉為灰階 (測試先轉灰階在增強對比度 效果?)
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # 將圖像轉為 RGB

    img_channel = img_rgb[:,:,1]  # 取出 G 通道 (R =[:,:,0], G =[:,:,1], B =[:,:,2])
    #img_gray_channel = img_gray  # 取出灰階通道 (灰階圖只有一個通道)

    img_channel = cv.normalize(img_channel, None, 0, 255, cv.NORM_MINMAX)   #通道強度增強
    #img_gray_channel = cv.normalize(img_gray_channel, None, 0, 255, cv.NORM_MINMAX)   #通道強度增強
    # img_channel = cv.equalizeHist(img_channel)  # 使用直方圖均衡化增強對比度
    
    clahe.setClipLimit(5.0)  # 調整對比度限制

    img_clahe = clahe.apply(img_channel)  # 使用 CLAHE 增強對比度
    #img_clahe = clahe.apply(img_gray_channel)  # 使用 CLAHE 增強對比度

    ## 將增強後的通道合併回三通道圖像
    enchanted_img = cv.merge([
        np.zeros_like(img_clahe), #R
        img_clahe, #G
        np.zeros_like(img_clahe)]  #B
    )

    ## 將灰階圖轉回綠螢光色RGB圖像
    # enchanted_img = cv.merge([
    #     np.zeros_like(img_clahe),  # R
    #     img_clahe,  # G
    #     np.zeros_like(img_clahe)   # B
    #  ])

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, enchanted_img)  # 儲存增強後的圖像
    print(f"✅ 已增強對比度並儲存：{output_path}")

def batch_enhance(input_folder, output_folder):
    # 支援的圖像副檔名
    exts = ('.jpg', '.png', '.bmp', '.tif')
    # 取得所有圖像檔案
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    if not files:
        print("⚠️ 找不到任何圖像檔案")
        return

    # 逐一處理每個檔案
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        enhance_contrast(input_path, output_path)

if __name__ == "__main__":
    # 設定來源資料夾與輸出資料夾路徑
    INPUT_FOLDER = r"C:\Users\USER\Desktop\lab\cell\opencv_ench\origin_ph\\"    # 換成你的圖片資料夾路徑
    OUTPUT_FOLDER = r"C:\Users\USER\Desktop\lab\cell\opencv_ench\output_ph\\"     # 轉換後的輸出資料夾

    # 執行批次轉換
    batch_enhance(INPUT_FOLDER, OUTPUT_FOLDER)