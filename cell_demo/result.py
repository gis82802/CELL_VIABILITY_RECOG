import os
import cv2
from ultralytics import YOLO
from openpyxl import Workbook

IOU_THRESHOLD = 0.5

# ===== 資料夾路徑 =====
IMAGE_DIR = 'testimages'
GT_DIR = 'labels_gt'
PRED_DIR = 'labels_pred'
MODEL_PATH = 'models/YOLOv11_green_best1.pt'

os.makedirs(PRED_DIR, exist_ok=True)

# ===== 計算 IoU =====
def compute_iou(box1, box2):
    # box = [x_center, y_center, w, h] (像素)
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2

    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_area = max(0, inter_xmax-inter_xmin) * max(0, inter_ymax-inter_ymin)

    area1 = (x1_max-x1_min)*(y1_max-y1_min)
    area2 = (x2_max-x2_min)*(y2_max-y2_min)
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

# ===== 讀取 YOLO 標籤 =====
def read_yolo_txt(path, img_w, img_h):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            # 將歸一化座標轉成像素
            x *= img_w
            y *= img_h
            w *= img_w
            h *= img_h
            boxes.append([cls, x, y, w, h])
    return boxes

# ===== 主程式 =====
def main():
    model = YOLO(MODEL_PATH)
    wb = Workbook()
    ws = wb.active
    ws.append(["Image", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"])

    total_tp, total_fp, total_fn = 0,0,0

    for img_file in os.listdir(IMAGE_DIR):
        if not img_file.lower().endswith(('.jpg','.png','.jpeg','.tif','.tiff')):
            continue
        img_path = os.path.join(IMAGE_DIR, img_file)
        gt_path = os.path.join(GT_DIR, os.path.splitext(img_file)[0]+'.txt')
        pred_path = os.path.join(PRED_DIR, os.path.splitext(img_file)[0]+'.txt')

        # ===== 讀取圖片尺寸 =====
        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]

        # ===== YOLO 偵測 =====
        results = model.predict(img_path, save=False, verbose=False)
        pred_boxes = []
        # 寫入 pred txt
        with open(pred_path, 'w') as f:
            for r in results:
                for i in range(len(r.boxes)):
                    box = r.boxes.xywh[i].cpu().numpy()
                    cls = int(r.boxes.cls[i].cpu().numpy())
                    x, y, w, h = box
                    pred_boxes.append([cls, x, y, w, h])
                    # 也存成 YOLO 格式（歸一化）
                    f.write(f"{cls} {x/img_w:.6f} {y/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n")

        # ===== 若有手動標註，計算指標 =====
        if os.path.exists(gt_path):
            gt_boxes = read_yolo_txt(gt_path, img_w, img_h)

            matched_gt = set()
            tp = 0
            for pred in pred_boxes:
                pred_cls, px, py, pw, ph = pred
                best_iou = 0
                best_gt = -1
                for i, gt in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    gt_cls, gx, gy, gw, gh = gt
                    # 忽略 class
                    iou = compute_iou([px, py, pw, ph], [gx, gy, gw, gh])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = i
                if best_iou >= IOU_THRESHOLD:
                    tp += 1
                    matched_gt.add(best_gt)

            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp

            precision = tp / (tp + fp) if (tp+fp) >0 else 0
            recall = tp / (tp + fn) if (tp+fn) >0 else 0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) >0 else 0

            total_tp += tp
            total_fp += fp
            total_fn += fn

            ws.append([img_file, tp, fp, fn, precision, recall, f1])
        else:
            ws.append([img_file, "N/A","N/A","N/A","N/A","N/A","N/A"])

    overall_precision = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0
    overall_recall = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0
    overall_f1 = 2*overall_precision*overall_recall/(overall_precision+overall_recall) if (overall_precision+overall_recall)>0 else 0
    ws.append(["Total", total_tp, total_fp, total_fn, overall_precision, overall_recall, overall_f1])
    wb.save("YOLO_Batch_Evaluation.xlsx")
    print("✅ 批量處理完成，結果已輸出：YOLO_Batch_Evaluation.xlsx")

if __name__ == "__main__":
    main()
