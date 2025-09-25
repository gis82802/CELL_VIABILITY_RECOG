import os
import cv2

# 單張轉換
def get_hsv_v(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:,:,2]   # V channel (灰階)

def get_lab_l(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab[:,:,0]   # L channel (灰階)

# 批次處理資料夾
def process_folder(input_dir, output_dir_v, output_dir_l):
    os.makedirs(output_dir_v, exist_ok=True)
    os.makedirs(output_dir_l, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ 無法讀取：{img_path}")
                continue

            # 取 V / L
            v_channel = get_hsv_v(img)
            l_channel = get_lab_l(img)

            # 存檔
            base, ext = os.path.splitext(filename)
            v_path = os.path.join(output_dir_v, f"{base}_HSV_V{ext}")
            l_path = os.path.join(output_dir_l, f"{base}_Lab_L{ext}")

            cv2.imwrite(v_path, v_channel)
            cv2.imwrite(l_path, l_channel)

            print(f"✅ 已處理: {filename}")

if __name__ == "__main__":
    input_dir = "testimages"   # 你的輸入資料夾
    output_dir_v = "images/results/HSV_V"
    output_dir_l = "images/results/Lab_L"

    process_folder(input_dir, output_dir_v, output_dir_l)
    print("🎉 全部完成！")
