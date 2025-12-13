# 導入必要的 Python 庫
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import os
import time
import cv2
import glob
import shutil
import torch
import timm
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
from config import IMAGE_FOLDER, SUMMARY_OUTPUT_PATH, ESRGAN_OUTPUT_PATH, CLUSTER_FOLDER, FONT_PATH, ROW_LABELS, COL_LABELS, CLASS_NAMES, CELL_SIZE, MAX_PER_ROW, MAX_ROWS, NUM_CLASSES, SCALING_FACTOR, ORIGIN_FOLDER
from image_utils import enhance_single_image, load_vit_model, adjust_single_image, load_yolo_model, yolo_detect_and_draw_and_save_txt, split_image_to_nine

# 主類：定義 GUI 界面和功能
class CellImageGUI:
    def __init__(self, root):
        # 初始化主窗口，設置標題為 "GUI"，大小為 1920x1080，背景色為白色
        self.root = root
        self.root.title("GUI")
        self.root.geometry("1920x1080")
        self.root.configure(bg="white")

        # 設置 YOLO 模型資料夾並載入可用模型
        yolo_folder = os.path.join("./weights", "YOLO")
        self.yolo_model_options = glob.glob(os.path.join(yolo_folder, "*.pt"))  # 獲取所有 .pt 檔案
        self.available_yolo_models = [os.path.basename(model) for model in self.yolo_model_options if os.path.isfile(model)]  # 過濾有效檔案
        if not self.available_yolo_models:
            print(f"❌ 無可用 YOLO 模型，請檢查 ./weights/YOLO/ 資料夾")
            self.available_yolo_models = ["YOLO_best.pt"]  # 預設模型
        self.current_yolo_model = self.available_yolo_models[0]  # 預設當前模型
        self.yolo_model = load_yolo_model(os.path.join(yolo_folder, self.current_yolo_model))  # 載入預設 YOLO 模型

        # 設置模型選項並載入可用模型
        self.model_options = ["Vision.pth", "best_mobilenet.pth", "swin_6class.pth"]  # 可用模型列表
        self.available_models = [model for model in self.model_options if os.path.exists(os.path.join("./weights", model))]  # 過濾存在檔案
        if not self.available_models:
            print("❌ 無可用模型，請檢查 ./weights/ 資料夾")
            self.available_models = ["Vision.pth"]  # 預設模型
        self.current_model_name = self.available_models[0]  # 預設當前模型

        # 定義 timm 模型映射和正規化參數
        self.timm_model_map = {
            "Vision.pth": "vit_base_patch16_224",
            "best_mobilenet.pth": "mobilenetv2_100",
            "swin_6class.pth": "swin_tiny_patch4_window7_224"
        }  # 映射 timm 模型名稱
        self.normalization_params = {
            "Vision.pth": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "swin_6class.pth": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            "best_mobilenet.pth": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }  # 正規化參數

        # 強制使用 CPU 避免 CUDA 問題
        self.device = torch.device("cpu")  # 設置設備為 CPU
        self.vit_model, _ = self.load_model(self.current_model_name)  # 載入預設模型

        # 載入原始圖片列表並初始化相關變量
        self.origin_files = sorted([f for f in os.listdir(ORIGIN_FOLDER) if f.lower().endswith((".jpg", ".png", ".bmp"))])  # 獲取並排序圖片
        self.image_files = []  # 儲存分割後的小圖列表
        self.image_index = 0  # 當前圖片索引
        self.summary_ready = False  # 總覽圖表是否準備好
        self.view_summary = True  # 預設啟用 view 模式
        self.current_origin_image = None  # 當前選擇的原始圖片

        # --- 左邊框 ---
        self.left_frame = tk.Frame(root, width=800, height=800, bg="white")  # 創建左側框架
        self.left_frame.pack(side="left", fill="y", padx=(19, 10), pady=10)  # 放置在左側

        # --- 標題列框架 ---
        self.title_frame = tk.Frame(self.left_frame, bg="white")  # 創建標題框架
        self.title_frame.pack(pady=(30, 10))  # 放置在左側框架中

        # 創建文字框架，放置標題和作者資訊
        text_frame = tk.Frame(self.title_frame, bg="white")  # 創建文字框架
        text_frame.pack(side="left", padx=(0, 10))  # 放置在左側
        tk.Label(text_frame, text="隊伍名稱：CellFies細胞自拍隊", font=("Arial", 24, "bold"), anchor="w", bg="white").pack(anchor="w")  
        tk.Label(text_frame, text="作品名稱：基於人工智慧之細胞活性分析", font=("Arial", 24, "bold"), anchor="w", bg="white").pack(anchor="w") 
        
        # 新增切片資訊標籤
        self.slice_info_label = tk.Label(text_frame, text="Current Slice: None", font=("Arial", 16), anchor="w", bg="white")  # 創建切片資訊標籤
        self.slice_info_label.pack(anchor="w", pady=5)  # 放置在左側

        # 嘗試載入並顯示 logo 圖片
        try:
            logo_image = Image.open("./asset/WhiteCell.png").resize((100, 100))  # 打開並調整 logo 大小
            self.logo_photo = ImageTk.PhotoImage(logo_image)  # 轉換為 Tkinter 格式
            tk.Label(self.title_frame, image=self.logo_photo, bg="white").pack(side="left")  # 顯示 logo
        except Exception as e:
            print(f"無法載入 logo 圖片: {e}")  # 打印錯誤訊息

        # --- 圖片顯示區域框架 ---
        self.img_placeholder = tk.Frame(self.left_frame, width=800, height=500, bg="white")  # 創建圖片顯示框架
        self.img_placeholder.pack_propagate(False)  # 禁止框架自動調整大小
        self.img_placeholder.pack(pady=(20, 10), padx=0)  # 放置在左側框架中

        # 初始化圖片標籤
        self.img_label = tk.Label(self.img_placeholder, text="尚未載入圖片", bg="white", anchor="center")  # 創建圖片標籤
        self.img_label.pack(fill="both", expand=True)  # 填充框架並擴展

        # --- 按鈕列框架 ---
        self.button_row = tk.Frame(self.left_frame, bg="white")  # 創建按鈕框架
        self.button_row.pack(pady=(10, 20), anchor="w")  # 放置在左側框架中

        # 上部分：創建按鈕框架
        self.button_frame = tk.Frame(self.button_row, bg="white")  # 創建按鈕子框架
        self.button_frame.pack(side="top", pady=(0, 10))  # 放置在頂部

        # 創建 "Previous" 按鈕
        self.prev_button = tk.Button(self.button_frame, text="⯇ Previous", command=self.prev_image,
                                     bg="white", fg="black", font=("Arial", 16, "bold"), relief="raised", width=16)  # 創建按鈕
        self.prev_button.pack(side="left", padx=(0, 10))  # 放置在左側

        # 創建 "Next" 按鈕
        self.next_button = tk.Button(self.button_frame, text="Next ⯈", command=self.next_image,
                                     bg="white", fg="black", font=("Arial", 16, "bold"), relief="raised", width=16)  # 創建按鈕
        self.next_button.pack(side="left", padx=20)  # 放置在左側

        self.selector_container = tk.Frame(self.button_row, bg="white")
        self.selector_container.pack(side="top", pady=(0, 0))  # 保持頂部放置

        # 標題行，使用 grid 佈局從左到右排列並左對齊
        self.title_row = tk.Frame(self.selector_container, bg="white")
        self.title_row.grid(row=0, column=0, sticky="w")  # 置於第0行第0列，左對齊
        tk.Label(self.title_row, text="image", font=("Arial", 20), bg="white").grid(row=0, column=0, padx=(0, 180), sticky="w")  # 左對齊
        tk.Label(self.title_row, text="YOLO", font=("Arial", 20), bg="white").grid(row=0, column=1, padx=(0, 240), sticky="w")  # 左對齊
        tk.Label(self.title_row, text="model", font=("Arial", 20), bg="white").grid(row=0, column=2, padx=(0, 20), sticky="w")  # 左對齊

        # 下拉選單行，使用 grid 佈局對齊標題
        self.selector_row = tk.Frame(self.selector_container, bg="white")
        self.selector_row.grid(row=1, column=0, sticky="w", pady=(5, 0))  # 置於第1行第0列，左對齊
        self.image_selector = ttk.Combobox(self.selector_row, values=self.origin_files, state="readonly", width=20, font=("Arial", 14))
        self.image_selector.grid(row=0, column=0, padx=(0, 20), sticky="w")  # 左對齊
        self.image_selector.bind("<<ComboboxSelected>>", self.on_image_select)
        if self.origin_files:
            self.image_selector.current(0)

        self.yolo_model_selector = ttk.Combobox(self.selector_row, values=self.available_yolo_models, state="readonly", width=25, font=("Arial", 14))
        self.yolo_model_selector.grid(row=0, column=1, padx=(0, 20), sticky="w")  # 左對齊
        self.yolo_model_selector.set(self.current_yolo_model)
        self.yolo_model_selector.bind("<<ComboboxSelected>>", self.on_yolo_select)

        self.model_selector = ttk.Combobox(self.selector_row, values=self.available_models, state="readonly", width=25, font=("Arial", 14))
        self.model_selector.grid(row=0, column=2, padx=(0, 20), sticky="w")  # 左對齊
        self.model_selector.set(self.current_model_name)
        self.model_selector.bind("<<ComboboxSelected>>", self.on_model_select)

        # --- 右邊框 ---
        self.right_frame = tk.Frame(root, width=1100, height=900, bg="white")  # 創建右側框架
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=100)  # 放置在右側

        # --- 畫布框架 ---
        self.canvas_frame = tk.Frame(self.right_frame, height=900, bg="white")  # 創建畫布框架
        self.canvas_frame.pack(side="top", fill="both", expand=True)  # 放置在右側框架頂部

        # 創建畫布
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", highlightthickness=0, width=1000, height=900)  # 創建畫布
        self.canvas.pack(side="left", fill="both", expand=True)  # 放置在畫布框架中

        # 分成兩個 Frame
        self.summary_frame = tk.Frame(self.canvas, bg="white", width=850, height=500)  # 創建總覽框架
        self.stats_frame = tk.Frame(self.canvas, bg="white", width=850, height=100)  # 創建統計框架

        self.canvas.create_window((50, 50), window=self.summary_frame, anchor="nw")  # 放置總覽框架
        self.canvas.create_window((50, 600), window=self.stats_frame, anchor="nw")  # 放置統計框架

        # 設置當前顯示框架
        self.image_frame = self.summary_frame

        # --- 進度條 ---
        self.esrgan_progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", maximum=100, length=100)  # 創建進度條
        self.esrgan_progress.place(x=20, y=860)  # 放置在窗口底部
        self.esrgan_progress.lower()  # 預設隱藏進度條

        # 如果有圖片，自動載入第一張
        if self.origin_files:
            self.on_image_select(None)  # 觸發圖片選擇事件

    def load_model(self, model_name):
        # 載入指定模型
        try:
            timm_model_name = self.timm_model_map.get(model_name, "vit_base_patch16_224")  # 獲取 timm 模型名稱
            model = timm.create_model(timm_model_name, pretrained=False, num_classes=NUM_CLASSES)  # 創建模型
            state_dict = torch.load(os.path.join("./weights", model_name), map_location=self.device)  # 載入權重
            model.load_state_dict(state_dict)  # 應用權重
            print(f"✅ 成功載入 {model_name} (timm 模型: {timm_model_name})")
            model.to(self.device)  # 移動到設備
            model.eval()  # 設置為評估模式
            return model, self.device
        except FileNotFoundError as e:
            print(f"❌ 載入模型 {model_name} 失敗: 檔案不存在 - {e}")
            return self.load_model("Vision.pth")  # 回退到預設模型
        except RuntimeError as e:
            print(f"❌ 載入模型 {model_name} 失敗: 權重不匹配 - {e}")
            return self.load_model("Vision.pth")  # 回退到預設模型
        except Exception as e:
            print(f"❌ 載入模型 {model_name} 失敗: 未知錯誤 - {e}")
            return self.load_model("Vision.pth")  # 回退到預設模型

    def on_model_select(self, event):
        # 處理模型選擇事件
        selected_model = self.model_selector.get()  # 獲取選中模型
        if selected_model != self.current_model_name:  # 檢查是否為新模型
            self.current_model_name = selected_model  # 更新當前模型
            self.vit_model, self.device = self.load_model(selected_model)  # 載入新模型
            for widget in self.image_frame.winfo_children():
                widget.destroy()  # 清空總覽框架
            for widget in self.stats_frame.winfo_children():
                widget.destroy()  # 清空統計框架
            self.stats_rows = []  # 重置統計行
            if self.image_files:
                self.load_image()  # 重新載入圖片

    def on_image_select(self, event):
        # 處理圖片選擇事件
        selected_image = self.image_selector.get()  # 獲取選中圖片
        if selected_image and selected_image != self.current_origin_image:  # 檢查是否為新圖片
            # 切換圖片時清理與前一張圖片相關的臨時檔案
            if self.current_origin_image:
                self.cleanup_temp_files_for_image(self.current_origin_image)
            
            self.current_origin_image = selected_image  # 更新當前圖片
            self.image_files = []  # 清空分割圖片列表
            self.image_index = 0  # 重置索引
            self.view_summary = True  # 啟用總覽模式
            self.on_camera_click(selected_image)  # 觸發圖片分割
            for widget in self.image_frame.winfo_children():
                widget.destroy()  # 清空總覽框架
            for widget in self.stats_frame.winfo_children():
                widget.destroy()  # 清空統計框架
            self.stats_rows = []  # 重置統計行
        self.load_image()  # 載入圖片
        self.update_slice_info()  # 更新切片資訊

    def cleanup_temp_files_for_image(self, image_name):
        # 清理與指定圖片相關的臨時檔案
        base_name = os.path.splitext(image_name)[0]  # 獲取檔案名（不含擴展名）
        temp_folders = [
            "./image/ESRGAN/",
            "./image/light&contrast/",
            "./image/YOLO/",
            "./image/label/",
            "./image/cropped/",
            IMAGE_FOLDER
        ]  # 定義臨時資料夾
        print(f"🔍 開始清理與 {image_name} 相關的臨時資料夾: {temp_folders}")
        for folder in temp_folders:
            if os.path.exists(folder):  # 檢查資料夾是否存在
                print(f"🔍 處理資料夾: {folder}")
                for item in os.listdir(folder):  # 遍歷資料夾內容
                    if item.startswith(base_name):  # 只刪除與前一張圖片相關的檔案
                        item_path = os.path.join(folder, item)
                        if os.path.isfile(item_path):  # 處理檔案
                            try:
                                os.remove(item_path)  # 刪除檔案
                                print(f"✅ 已刪除檔案: {item_path}")
                            except Exception as e:
                                print(f"❌ 刪除檔案 {item_path} 失敗: {e}")
                        elif os.path.isdir(item_path):  # 處理子資料夾
                            try:
                                shutil.rmtree(item_path)  # 刪除子資料夾
                                print(f"✅ 已刪除子資料夾: {item_path}")
                            except Exception as e:
                                print(f"❌ 刪除子資料夾 {item_path} 失敗: {e}")

    def update_slice_info(self):
        # 更新切片資訊標籤
        if self.image_files and 0 <= self.image_index < len(self.image_files):  # 檢查索引有效性
            slice_name = self.image_files[self.image_index]  # 獲取當前切片名稱
            self.slice_info_label.configure(text=f"Current Slice: {slice_name}")  # 更新標籤
        else:
            self.slice_info_label.configure(text="Current Slice: None")  # 預設無切片

    def run_full_pipeline(self, selected_img_name, selected_img_path, esrgan_output_path, light_contrast_output_path, final_with_boxes_path, final_txt_path):
        # 執行圖片處理管線
        try:
            threading.Thread(target=self.progress_to_target, args=(30, 0.045), daemon=True).start()  # 更新進度條到 30%
            enhance_single_image(selected_img_path, esrgan_output_path)  # 執行圖片增強
            self.root.after(0, lambda: self.update_left_image(esrgan_output_path))  # 更新左側圖片

            threading.Thread(target=self.progress_to_target, args=(50, 0.01), daemon=True).start()  # 更新進度條到 50%
            adjust_single_image(esrgan_output_path, light_contrast_output_path, contrast=0, brightness=0)  # 調整亮度
            self.root.after(0, lambda: self.update_left_image(light_contrast_output_path))  # 更新左側圖片

            threading.Thread(target=self.progress_to_target, args=(70, 0.01), daemon=True).start()  # 更新進度條到 70%
            yolo_detect_and_draw_and_save_txt(light_contrast_output_path, final_with_boxes_path, final_txt_path)  # 執行 YOLO 檢測
            self.root.after(0, lambda: self.update_left_image(final_with_boxes_path))  # 更新左側圖片

            threading.Thread(target=self.progress_to_target, args=(100, 0.15), daemon=True).start()  # 更新進度條到 100%
            self.crop_current_image_objects()  # 裁剪物件
            self.classify_and_move_cropped_cells()  # 分類並移動細胞
            self.root.after(0, lambda: self.esrgan_progress.lower())  # 處理完成後隱藏進度條
        except Exception as e:
            print(f"❌ 背景子執行緒錯誤: {e}")  # 打印異常訊息

    def progress_to_target(self, target_value, sleep_time):
        # 更新進度條到指定目標值
        while self.esrgan_progress['value'] < target_value:  # 檢查是否達到目標值
            self.esrgan_progress['value'] += 1  # 增加進度值
            self.root.update_idletasks()  # 更新界面
            time.sleep(sleep_time)  # 控制更新速度

    def update_left_image(self, image_path):
        # 更新左側圖片顯示區域
        try:
            img = Image.open(image_path).copy().resize((800, 500))  # 打開並調整圖片大小
            img_tk = ImageTk.PhotoImage(img)  # 轉換為 Tkinter 格式
            self.img_label.configure(image=img_tk, text="")  # 更新圖片標籤
            self.img_label.image = img_tk  # 保留引用防止垃圾回收
        except Exception as e:
            print(f"❌ 更新圖片失敗: {e}")  # 打印錯誤訊息
            self.img_label.configure(text="無法載入圖片", bg="white", anchor="center")  # 顯示錯誤訊息

    def generate_and_show_summary(self):
        # 觸發生成並顯示總覽圖表
        self.root.update_idletasks()  # 更新界面
        self.root.after(0, self.generate_summary_image)  # 延遲執行

    def crop_current_image_objects(self):
        # 根據 YOLO 檢測結果裁剪當前圖片中的目標物件
        image_name = self.image_files[self.image_index]  # 獲取當前圖片名稱
        base_name = os.path.splitext(image_name)[0]  # 獲取檔案名（不含擴展名）
        image_path = os.path.join("./image/light&contrast", image_name)  # 構建圖片路徑
        label_path = os.path.join("./image/label", f"{base_name}.txt")  # 構建標籤路徑
        output_folder = "./image/cropped"  # 定義裁剪輸出資料夾

        if not os.path.exists(output_folder):  # 檢查資料夾是否存在
            os.makedirs(output_folder)  # 創建資料夾

        img = cv2.imread(image_path)  # 讀取圖片
        if img is None:  # 檢查是否成功讀取
            print(f"⚠️ 無法讀取圖片: {image_path}")
            return

        height, width = img.shape[:2]  # 獲取圖片尺寸

        if not os.path.exists(label_path):  # 檢查標籤檔案是否存在
            print(f"⚠️ 找不到標籤檔案: {label_path}")
            return

        with open(label_path, 'r') as f:  # 打開標籤檔案
            lines = f.readlines()  # 讀取所有行

        for idx, line in enumerate(lines):  # 遍歷每行
            parts = line.strip().split()  # 分割行內容
            if len(parts) != 5:  # 檢查格式是否正確
                continue

            class_id, cx, cy, w, h = map(float, parts)  # 解析座標和尺寸
            cx *= width  # 轉換為像素值
            cy *= height
            w *= width
            h *= height

            xmin = int(cx - w / 2)  # 計算左上角 x 座標
            ymin = int(cy - h / 2)  # 計算左上角 y 座標
            xmax = int(cx + w / 2)  # 計算右下角 x 座標
            ymax = int(cy + h / 2)  # 計算右下角 y 座標

            xmin = max(0, xmin)  # 確保不超出邊界
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)

            cropped_img = img[ymin:ymax, xmin:xmax]  # 裁剪圖片
            output_path = os.path.join(output_folder, f"{base_name}_{idx}.jpg")  # 構建輸出路徑
            cv2.imwrite(output_path, cropped_img)  # 儲存裁剪結果
            print(f"✅ 裁切完成: {output_path}")  # 打印成功訊息

    def classify_and_move_cropped_cells(self):
        # 對裁剪後的細胞進行分類並移動到相應資料夾
        cropped_folder = "./image/cropped"  # 定義裁剪資料夾
        import shutil  # 導入 shutil 用於檔案操作

        for class_id in range(1, NUM_CLASSES + 1):  # 遍歷所有類別
            class_folder = os.path.join('./sorted', str(class_id))  # 構建類別資料夾路徑
            if os.path.exists(class_folder):  # 檢查資料夾是否存在
                shutil.rmtree(class_folder)  # 刪除舊資料夾
            os.makedirs(class_folder, exist_ok=True)  # 創建新資料夾

        base_name = os.path.splitext(self.image_files[self.image_index])[0]  # 獲取當前圖片名稱
        cropped_images = sorted(glob.glob(os.path.join(cropped_folder, f"{base_name}_*.jpg")))  # 獲取裁剪圖片

        if not cropped_images:  # 檢查是否有裁剪圖片
            print("⚠️ 沒有找到裁切的小圖")
            return

        mean, std = self.normalization_params.get(self.current_model_name, ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # 獲取正規化參數
        transform = transforms.Compose([  # 定義圖片轉換流程
            transforms.Resize((224, 224)),  # 調整大小
            transforms.ToTensor(),  # 轉為張量
            transforms.Normalize(mean=mean, std=std),  # 正規化
        ])
        count = 0  # 計數器
        for img_path in cropped_images:  # 遍歷裁剪圖片
            img_pil = Image.open(img_path).convert('RGB')  # 打開圖片
            input_tensor = transform(img_pil).unsqueeze(0).to(self.device)  # 轉換為張量

            with torch.no_grad():  # 關閉梯度計算
                logits = self.vit_model(input_tensor)  # 預測
                pred_class = logits.argmax(dim=1).item()  # 獲取預測類別

            class_folder = os.path.join('./sorted', str(pred_class + 1))  # 構建目標資料夾
            os.makedirs(class_folder, exist_ok=True)  # 創建資料夾

            filename = os.path.basename(img_path)  # 獲取檔案名稱
            filename_no_ext = os.path.splitext(filename)[0]  # 移除擴展名
            new_filename = f"{filename_no_ext}_c{pred_class+1}.jpg"  # 新檔案名稱
            new_path = os.path.join(class_folder, new_filename)  # 新路徑

            shutil.copy(img_path, new_path)  # 複製檔案
            count += 1  # 增加計數
            if count % 8 == 0:  # 每 8 張更新一次總覽
                self.root.after(0, self.generate_summary_image)
        self.root.after(0, self.generate_summary_image)  # 最後更新總覽

    def show_loading_gif(self):
        # 顯示載入動畫 GIF
        self.loading_gif = Image.open("./asset/loading.gif")  # 打開 GIF 檔案
        self.loading_frames = []  # 儲存 GIF 幀

        try:
            while True:  # 遍歷所有幀
                frame = self.loading_gif.copy()  # 複製當前幀
                frame = frame.resize((frame.width * 3, frame.height * 3), Image.LANCZOS)  # 調整大小
                frame = ImageTk.PhotoImage(frame)  # 轉換為 Tkinter 格式
                self.loading_frames.append(frame)  # 添加到幀列表
                self.loading_gif.seek(len(self.loading_frames))  # 切換到下一幀
        except EOFError:
            pass  # 處理 GIF 結束

        self.loading_index = 0  # 初始化索引
        self.loading_label = tk.Label(self.right_frame, bg="white")  # 創建載入標籤
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")  # 置中顯示
        self.root.update_idletasks()  # 更新界面
        self.animate_loading()  # 啟動動畫

    def animate_loading(self):
        # 動畫載入 GIF 幀
        if hasattr(self, 'loading_frames') and self.loading_frames:  # 檢查幀列表
            frame = self.loading_frames[self.loading_index]  # 獲取當前幀
            self.loading_label.configure(image=frame)  # 更新標籤
            self.loading_label.image = frame  # 保留引用
            self.loading_index = (self.loading_index + 1) % len(self.loading_frames)  # 更新索引
            self.root.after(30, self.animate_loading)  # 30ms 後刷新

    def generate_summary_image(self):
        # 生成並顯示總覽圖表
        import os
        import cv2
        import numpy as np
        from PIL import Image, ImageTk, ImageDraw, ImageFont

        def update_summary():
            if hasattr(self, 'loading_label') and self.loading_label.winfo_exists():  # 檢查載入標籤是否存在
                self.loading_label.destroy()  # 銷毀載入標籤
            if hasattr(self, 'loading_frames'):  # 清理幀列表
                del self.loading_frames
            if hasattr(self, 'loading_gif'):  # 清理 GIF 物件
                del self.loading_gif

            self.image_prefix = os.path.splitext(self.image_files[self.image_index])[0]  # 獲取圖片前綴
            self.cell_paths_by_class = {}  # 儲存每個類別的圖片路徑
            cell_grids = {}  # 儲存每個類別的網格
            cell_counts = {}  # 儲存每個類別的數量

            for class_id in range(1, NUM_CLASSES + 1):  # 遍歷所有類別
                class_path = os.path.join(CLUSTER_FOLDER, str(class_id))  # 構建類別資料夾路徑
                image_files = sorted(
                    [f for f in os.listdir(class_path) if f.startswith(self.image_prefix)],  # 過濾圖片
                    key=lambda x: os.path.getmtime(os.path.join(class_path, x))  # 按修改時間排序
                ) if os.path.exists(class_path) else []  # 檢查資料夾是否存在

                paths = [os.path.join(class_path, f) for f in image_files]  # 構建圖片路徑列表
                self.cell_paths_by_class[class_id] = paths.copy()  # 儲存路徑
                cell_counts[class_id] = len(image_files)  # 記錄數量

                cell_images = []  # 儲存圖片
                for f in image_files:  # 遍歷圖片
                    img = cv2.imread(os.path.join(class_path, f))  # 讀取圖片
                    if img is not None:  # 檢查是否成功讀取
                        img_resized = cv2.resize(img, (CELL_SIZE, CELL_SIZE))  # 調整大小
                        cell_images.append(img_resized)  # 添加到列表

                total_needed = MAX_PER_ROW * MAX_ROWS  # 計算所需總數
                while len(cell_images) < total_needed:  # 填充空白
                    cell_images.append(np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8))
                    self.cell_paths_by_class[class_id].append(None)

                cell_images = cell_images[:total_needed]  # 截斷到最大數量
                rows = [np.hstack(cell_images[i:i + MAX_PER_ROW]) for i in range(0, len(cell_images), MAX_PER_ROW)]  # 水平拼接
                cell_grids[class_id] = np.vstack(rows)  # 垂直拼接

            cell_height = MAX_ROWS * CELL_SIZE  # 計算網格高度
            cell_width = MAX_PER_ROW * CELL_SIZE  # 計算網格寬度
            summary_cv2 = np.full(((cell_height * 3) + 60, (cell_width * 2) + 80, 3), 255, dtype=np.uint8)  # 創建總覽畫布

            for r in range(3):  # 遍歷 3 行
                for c in range(2):  # 遍歷 2 列
                    idx = r * 2 + c + 1  # 計算類別索引
                    y, x = 60 + r * cell_height, 80 + c * cell_width  # 計算位置
                    summary_cv2[y:y + cell_height, x:x + cell_width] = cell_grids[idx]  # 填充網格
                    cv2.rectangle(summary_cv2, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 4)  # 繪製邊框

            summary_pil = Image.fromarray(cv2.cvtColor(summary_cv2, cv2.COLOR_BGR2RGB))  # 轉換為 PIL 圖像
            draw = ImageDraw.Draw(summary_pil)  # 創建繪圖物件
            font = ImageFont.truetype(FONT_PATH, 27)  # 設置字體

            for i, label in enumerate(COL_LABELS):  # 添加列標籤
                x = 80 + i * cell_width + cell_width // 2 - 30
                draw.text((x, 0), label, fill="black", font=font)

            for i, label in enumerate(ROW_LABELS):  # 添加行標籤
                y = 60 + i * cell_height + cell_height // 2 - 20
                draw.text((0, y), label, fill="black", font=font)

            if SCALING_FACTOR < 1.0:  # 檢查縮放因子
                w, h = summary_pil.size  # 獲取圖像尺寸
                try:
                    summary_pil = summary_pil.resize((int(w * SCALING_FACTOR), int(h * SCALING_FACTOR)), resample=Image.Resampling.LANCZOS)
                except AttributeError:
                    summary_pil = summary_pil.resize((int(w * SCALING_FACTOR), int(h * SCALING_FACTOR)), resample=Image.LANCZOS)

            for widget in self.image_frame.winfo_children():  # 清空舊 widget
                widget.destroy()

            summary_pil_tk = ImageTk.PhotoImage(summary_pil)  # 轉換為 Tkinter 格式
            summary_label = tk.Label(self.image_frame, image=summary_pil_tk, anchor="nw", bg="white")  # 創建標籤
            summary_label.image = summary_pil_tk  # 保留引用
            summary_label.pack()  # 顯示圖像
            summary_label.bind("<Button-1>", self.on_summary_click)  # 綁定點擊事件

            total_cells = sum(cell_counts.values())  # 計算總細胞數
            if not hasattr(self, 'stats_rows') or not self.stats_rows:  # 初始化統計行
                self.stats_rows = []
                for row in range(3):
                    for col in range(2):
                        stat_row = tk.Frame(self.stats_frame, bg="white")  # 創建統計行框架
                        label = tk.Label(stat_row, text="", font=("Arial", 14), bg="white", width=18, anchor="w")  # 創建標籤
                        label.pack(side="left")  # 放置在左側
                        blocks = []  # 儲存進度塊
                        for i in range(10):  # 創建 10 個進度塊
                            block = tk.Frame(stat_row, width=11, height=17, bg="black", bd=1, relief="solid")  # 創建塊
                            block.pack(side="left", padx=1)  # 放置在左側
                            blocks.append(block)
                        stat_row.grid(row=row, column=col, padx=4, pady=4, sticky="w")  # 佈局
                        self.stats_rows.append((label, blocks))

            for idx, (label_widget, blocks_widgets) in enumerate(self.stats_rows):  # 更新統計數據
                cid = idx + 1  # 類別 ID
                count = cell_counts.get(cid, 0)  # 獲取數量
                percent = (count / total_cells) * 100 if total_cells > 0 else 0  # 計算百分比
                lights_on = min(10, int((percent + 9.999) // 10)) if percent > 0 else 0  # 計算亮燈數

                label_text = f"{CLASS_NAMES[cid]}: {percent:.1f}%"  # 格式化文字
                label_widget.config(text=label_text)  # 更新標籤

                colors = ["#fdcb6e"] * 3 + ["#e17055"] * 3 + ["#d63031"] * 4  # 定義顏色
                for i in range(10):  # 更新進度塊顏色
                    color = colors[i] if i < lights_on else "black"
                    blocks_widgets[i].config(bg=color)

        self.root.after(0, update_summary)  # 延遲執行更新

    def on_summary_click(self, event):
        # 處理總覽圖表點擊事件，切換細胞類別
        scaling = getattr(self, 'scaling_factor', SCALING_FACTOR)  # 獲取縮放因子
        x = int(event.x / scaling)  # 轉換 x 座標
        y = int(event.y / scaling)  # 轉換 y 座標

        from config import CELL_SIZE, MAX_PER_ROW, MAX_ROWS  # 導入配置
        CELL_W = CELL_SIZE  # 單元格寬度
        CELL_H = CELL_SIZE  # 單元格高度
        PAD_X = 80  # 水平邊距
        PAD_Y = 60  # 垂直邊距
        GRID_W = MAX_PER_ROW * CELL_W  # 網格寬度
        GRID_H = MAX_ROWS * CELL_H  # 網格高度

        col = (x - PAD_X) // GRID_W  # 計算列
        row = (y - PAD_Y) // GRID_H  # 計算行
        if not (0 <= col <= 1 and 0 <= row <= 2):  # 檢查是否在範圍內
            print("點擊範圍超出六宮格")
            return

        class_id = row * 2 + col + 1  # 計算類別 ID
        local_x = (x - PAD_X - col * GRID_W)  # 計算局部 x
        local_y = (y - PAD_Y - row * GRID_H)  # 計算局部 y
        grid_c = local_x // CELL_W  # 計算列索引
        grid_r = local_y // CELL_H  # 計算行索引
        index = grid_r * MAX_PER_ROW + grid_c  # 計算索引

        paths = self.cell_paths_by_class.get(class_id, [])  # 獲取路徑列表
        if 0 <= index < len(paths):  # 檢查索引有效性
            path = paths[index]  # 獲取路徑
            if path is not None:  # 檢查是否為有效圖片
                print(f"✅ 點選 class_id={class_id}, index={index} → {os.path.basename(path)}")
                self.toggle_cell_class(path, class_id)  # 切換類別
            else:
                print("⚠️ 點到空格子，無圖片")
        else:
            print("⚠️ 無效 index")

    def toggle_cell_class(self, img_path, current_class_id):
        # 切換細胞類別
        import shutil  # 導入 shutil 用於檔案操作
        toggle_map = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5}  # 定義切換映射
        new_class_id = toggle_map[current_class_id]  # 獲取新類別
        filename = os.path.basename(img_path)  # 獲取檔案名稱
        new_dir = os.path.join("./sorted", str(new_class_id))  # 構建新資料夾路徑
        os.makedirs(new_dir, exist_ok=True)  # 創建資料夾

        try:
            dst_path = os.path.join(new_dir, filename)  # 構建目標路徑
            shutil.move(img_path, dst_path)  # 移動檔案
            os.utime(dst_path, None)  # 更新檔案時間
            print(f"✅ 已將 {filename} 從類別 {current_class_id} 移至 {new_class_id}")
            self.root.after(0, self.generate_summary_image)  # 更新總覽圖表
        except Exception as e:
            print(f"❌ 移動失敗: {e}")  # 打印錯誤訊息

    def on_yolo_select(self, event):
        # 處理 YOLO 模型選擇事件
        selected_yolo_model = self.yolo_model_selector.get()  # 獲取選中模型
        if selected_yolo_model != self.current_yolo_model:  # 檢查是否為新模型
            self.current_yolo_model = selected_yolo_model  # 更新當前模型
            yolo_folder = os.path.join("./weights", "YOLO")  # 構建 YOLO 資料夾路徑
            self.yolo_model = load_yolo_model(os.path.join(yolo_folder, self.current_yolo_model))  # 載入新模型
            print(f"✅ 已切換至 YOLO 模型: {self.current_yolo_model}")
            for widget in self.image_frame.winfo_children():
                widget.destroy()  # 清空總覽框架
            for widget in self.stats_frame.winfo_children():
                widget.destroy()  # 清空統計框架
            self.stats_rows = []  # 重置統計行
            if self.image_files:
                self.load_image()  # 重新載入圖片

    def prev_image(self):
        # 處理上一張圖片按鈕事件
        self.view_summary = True  # 啟用總覽模式
        self.image_index = (self.image_index - 1) % len(self.image_files)  # 更新索引

        for widget in self.image_frame.winfo_children():
            widget.destroy()  # 清空總覽框架
        for widget in self.stats_frame.winfo_children():
            widget.destroy()  # 清空統計框架
        self.stats_rows = []  # 重置統計行
        self.cleanup_temp_files()  # 清理臨時檔案
        self.load_image()  # 載入圖片
        self.update_slice_info()  # 更新切片資訊

    def next_image(self):
        # 處理下一張圖片按鈕事件
        self.view_summary = True  # 啟用總覽模式
        self.image_index = (self.image_index + 1) % len(self.image_files)  # 更新索引

        for widget in self.image_frame.winfo_children():
            widget.destroy()  # 清空總覽框架
        for widget in self.stats_frame.winfo_children():
            widget.destroy()  # 清空統計框架
        self.stats_rows = []  # 重置統計行
        self.cleanup_temp_files()  # 清理臨時檔案
        self.load_image()  # 載入圖片
        self.update_slice_info()  # 更新切片資訊

    def load_image(self):
        # 載入當前圖片
        if not self.image_files:  # 檢查是否有圖片
            self.img_label.configure(image=None, text="尚未載入圖片")  # 顯示提示
            self.update_slice_info()  # 更新切片資訊
            return

        file_path = os.path.join(IMAGE_FOLDER, self.image_files[self.image_index])  # 構建圖片路徑
        img = Image.open(file_path).convert("RGB").copy().resize((800, 500))  # 打開並調整大小
        img_tk = ImageTk.PhotoImage(img)  # 轉換為 Tkinter 格式
        self.img_label.configure(image=None)  # 清空舊圖片
        self.img_label.configure(image=img_tk, text="")  # 更新圖片標籤
        self.img_label.image = img_tk  # 保留引用

        # 自動運行 view 邏輯
        selected_img_name = self.image_files[self.image_index]  # 獲取當前圖片名稱
        selected_img_path = os.path.join(IMAGE_FOLDER, selected_img_name)  # 構建圖片路徑
        esrgan_output_path = "./image/ESRGAN/" + selected_img_name  # ESRGAN 輸出路徑
        light_contrast_output_path = "./image/light&contrast/" + selected_img_name  # 亮度調整輸出路徑
        final_with_boxes_path = "./image/YOLO/" + selected_img_name  # YOLO 檢測輸出路徑
        final_txt_path = "./image/label/" + selected_img_name.replace(".jpg", ".txt").replace(".png", ".txt")  # 標籤檔案路徑

        self.esrgan_progress['value'] = 0  # 重置進度條
        self.esrgan_progress.lift()  # 顯示進度條
        self.root.update_idletasks()  # 更新界面
        threading.Thread(
            target=self.run_full_pipeline,
            args=(selected_img_name, selected_img_path, esrgan_output_path, light_contrast_output_path, final_with_boxes_path, final_txt_path),
            daemon=True  # 設置為背景執行緒
        ).start()  # 啟動執行緒

        self.update_slice_info()  # 更新切片資訊

    def on_camera_click(self, selected_image=None):
        # 處理相機按鈕點擊事件，執行圖片分割
        origin_files = sorted([
            f for f in os.listdir(ORIGIN_FOLDER)
            if f.lower().endswith((".jpg", ".png", ".bmp"))
        ])  # 獲取並排序原始圖片

        if not origin_files:  # 檢查是否有圖片
            print("⚠️ ORIGIN_PATH裡沒有原圖！")
            return

        if selected_image is None:  # 若未指定圖片
            selected_image = self.image_selector.get()  # 從下拉選單獲取
            if not selected_image:
                print("⚠️ 請從下拉選單選擇一張圖片！")
                return

        if selected_image not in origin_files:  # 檢查圖片是否有效
            print(f"⚠️ 選擇的圖片 {selected_image} 不在原始圖片列表中！")
            return

        if os.path.exists(IMAGE_FOLDER):  # 檢查資料夾是否存在
            for file in os.listdir(IMAGE_FOLDER):  # 遍歷資料夾內容
                file_path = os.path.join(IMAGE_FOLDER, file)
                if os.path.isfile(file_path):  # 處理檔案
                    os.remove(file_path)  # 刪除檔案
        else:
            os.makedirs(IMAGE_FOLDER)  # 創建資料夾

        self.summary_ready = True  # 設置總覽準備狀態
        current_path = os.path.join(ORIGIN_FOLDER, selected_image)  # 構建圖片路徑

        try:
            split_paths = split_image_to_nine(current_path, IMAGE_FOLDER)  # 執行圖片分割
        except Exception as e:
            print(f"切割失敗: {e}")  # 打印錯誤訊息
            return

        base_name = os.path.splitext(selected_image)[0]  # 獲取檔案名（不含擴展名）
        self.image_files = sorted([
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith((".jpg", ".png", ".bmp")) and f.startswith(base_name)
        ])  # 更新分割圖片列表
        self.image_index = 0  # 重置索引

        if self.image_files:
            self.load_image()  # 載入第一張圖片

    def cleanup_temp_files(self):
        # 清理臨時檔案和資料夾
        temp_folders = ["./image/ESRGAN/", "./image/light&contrast/", "./image/YOLO/", "./image/label/", "./image/cropped/"]  # 定義臨時資料夾
        print(f"🔍 開始清理臨時資料夾: {temp_folders}")
        for folder in temp_folders:
            if os.path.exists(folder):  # 檢查資料夾是否存在
                print(f"🔍 處理資料夾: {folder}")
                try:
                    for item in os.listdir(folder):  # 遍歷資料夾內容
                        item_path = os.path.join(folder, item)
                        if os.path.isfile(item_path):  # 處理檔案
                            os.remove(item_path)  # 刪除檔案
                            print(f"✅ 已刪除檔案: {item_path}")
                        elif os.path.isdir(item_path):  # 處理子資料夾
                            shutil.rmtree(item_path)  # 刪除子資料夾
                            print(f"✅ 已刪除子資料夾: {item_path}")
                except Exception as e:
                    print(f"❌ 清理資料夾 {folder} 失敗: {e}")  # 打印清理失敗訊息
        if os.path.exists(SUMMARY_OUTPUT_PATH):  # 檢查總覽檔案是否存在
            try:
                os.unlink(SUMMARY_OUTPUT_PATH)  # 刪除檔案
                print(f"✅ 已自動刪除檔案: {SUMMARY_OUTPUT_PATH}")
            except Exception as e:
                print(f"❌ 刪除檔案 {SUMMARY_OUTPUT_PATH} 失敗: {e}")  # 打印刪除失敗訊息

if __name__ == "__main__":
    # 啟動主應用程式
    root = tk.Tk()  # 創建主窗口
    app = CellImageGUI(root)  # 實例化 GUI
    root.mainloop()  # 啟動主循環