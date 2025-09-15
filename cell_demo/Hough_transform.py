import cv2
import numpy as np
import os

# ====== 參數設定 ======
IMAGE_DIR = "testimages"  # 原圖資料夾
RESULT_DIR = "images/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 批次處理資料夾裡所有圖片
for filename in os.listdir(IMAGE_DIR):
    if not (filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".jpg")):
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 無法讀取: {filename}")
        continue

    # 霍夫圓檢測
    circles = cv2.HoughCircles(img,
                               cv2.HOUGH_GRADIENT,
                               dp=0.9,
                               minDist=20,
                               param1=15,  # Canny 高閾值
                               param2=15,   # 圓心累計閾值
                               minRadius=5,
                               maxRadius=50)

    # 畫圓
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output, center, 2, (0, 0, 255), 3)   # 圓心
            cv2.circle(output, center, radius, (0, 255, 0), 2)  # 圓周

    # 存檔
    result_path = os.path.join(RESULT_DIR, f"hough_{filename}")
    cv2.imwrite(result_path, output)
    print(f"✅ 已存: {result_path}")

print("🎉 所有圖片處理完成！")
