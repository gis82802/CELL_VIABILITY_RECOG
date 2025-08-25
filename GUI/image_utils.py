# image_utils.py

import os
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from realesrgan import RealESRGANer
import torch 
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from ultralytics import YOLO
import timm
yolo_model = None
def split_image_to_nine(img_path, output_folder):
    """
    把指定圖片切成9等分，存到 output_folder
    回傳：切好的圖片路徑 list
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    h, w = img.shape[:2]
    split_h = h // 3
    split_w = w // 3

    os.makedirs(output_folder, exist_ok=True)
    split_paths = []

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for row in range(3):
        for col in range(3):
            y1 = row * split_h
            y2 = (row + 1) * split_h
            x1 = col * split_w
            x2 = (col + 1) * split_w
            sub_img = img[y1:y2, x1:x2]

            split_filename = f"{base_name}_{row*3+col+1}.jpg"
            split_path = os.path.join(output_folder, split_filename)
            cv2.imwrite(split_path, sub_img)
            split_paths.append(split_path)

    return split_paths
def enhance_single_image(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),  # 有GPU就開half
        device=device
    )

    img = Image.open(input_path).convert('RGB')
    sr_image, _ = upsampler.enhance(np.array(img), outscale=4)

    sr_image = Image.fromarray(sr_image)
    sr_image.save(output_path)
    print(f"✅ 單張超解析完成: {output_path}")



def adjust_single_image(input_path, output_path, contrast, brightness):
    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ 無法讀取圖片: {input_path}")
        return

    output = img * (contrast / 127 + 1) - contrast + brightness
    output = np.clip(output, 0, 255)
    output = np.uint8(output)

    cv2.imwrite(output_path, output)
    print(f"✅ 單張亮度對比調整完成: {output_path}")

def load_yolo_model(model_path):
    global yolo_model
    yolo_model = YOLO(model_path)
    print(f"✅ 成功載入 YOLOv8 模型: {model_path}")

def yolo_detect_and_draw_and_save_txt(input_path, output_image_path, output_txt_path):
    if yolo_model is None:
        raise Exception("❌ 請先呼叫 load_yolo_model 載入模型")

    # 🔥 加上你的設定：iou=0.5, conf=0.1
    results = yolo_model.predict(source=input_path, save=False, imgsz=640, conf=0.1, iou=0.1, verbose=False)

    img = cv2.imread(input_path)
    if img is None:
        raise Exception(f"❌ 無法讀取圖片 {input_path}")

    height, width = img.shape[:2]
    annotations = []

    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0])

            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            annotations.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")

            # 🔥 畫黃色框 (255, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    # 儲存畫好框線的圖片
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, img)
    print(f"✅ 已儲存畫好框線的圖片：{output_image_path}")

    # 儲存txt標註檔
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        for line in annotations:
            f.write(line + "\n")
    print(f"✅ 已儲存框線標註txt：{output_txt_path}")

def load_vit_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ 成功載入ViT模型")
    return model, device