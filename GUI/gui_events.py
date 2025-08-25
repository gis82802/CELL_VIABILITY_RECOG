import os
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import shutil
from config import IMAGE_FOLDER, ESRGAN_OUTPUT_PATH, SUMMARY_OUTPUT_PATH, CLUSTER_FOLDER, FONT_PATH, ROW_LABELS, COL_LABELS, CLASS_NAMES_CH, CELL_SIZE, MAX_PER_ROW, MAX_ROWS, NUM_CLASSES, SCALING_FACTOR, ORIGIN_FOLDER
from image_utils import load_yolo_model, yolo_detect_and_draw_and_save_txt, enhance_single_image, adjust_single_image

def prev_image(self):
    self.view_summary = False
    self.image_index = (self.image_index - 1) % len(self.image_files)

    for widget in self.image_frame.winfo_children():
        widget.destroy()

    for widget in self.stats_frame.winfo_children():
        widget.destroy()

    self.stats_rows = []
    load_image(self)
    self.update_slice_info()  # 更新切片資訊

def next_image(self):
    self.view_summary = False
    self.image_index = (self.image_index + 1) % len(self.image_files)

    for widget in self.image_frame.winfo_children():
        widget.destroy()

    for widget in self.stats_frame.winfo_children():
        widget.destroy()

    self.stats_rows = []
    load_image(self)
    self.update_slice_info()  # 更新切片資訊

def load_image(self):
    if not self.image_files:
        self.img_label.configure(image=None, text="尚未載入圖片")
        self.update_slice_info()  # 更新切片資訊
        return
    file_path = os.path.join(IMAGE_FOLDER, self.image_files[self.image_index])
    img = Image.open(file_path).convert("RGB").copy().resize((800, 500))
    img_tk = ImageTk.PhotoImage(img)
    self.img_label.configure(image=None)
    self.img_label.configure(image=img_tk, text="")
    self.img_label.image = img_tk
    self.update_slice_info()  # 更新切片資訊

def on_camera_click(self, selected_image=None):
    from image_utils import split_image_to_nine
    origin_files = sorted([
        f for f in os.listdir(ORIGIN_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".bmp"))
    ])

    if not origin_files:
        print("⚠️ ORIGIN_PATH裡沒有原圖！")
        return

    # 如果未提供 selected_image，使用下拉選單的當前選擇
    if selected_image is None:
        selected_image = self.image_selector.get()
        if not selected_image:
            print("⚠️ 請從下拉選單選擇一張圖片！")
            return

    # 確保選擇的圖片在 origin_files 中
    if selected_image not in origin_files:
        print(f"⚠️ 選擇的圖片 {selected_image} 不在原始圖片列表中！")
        return

    # 清空 IMAGE_FOLDER 中的舊切片
    if os.path.exists(IMAGE_FOLDER):
        for file in os.listdir(IMAGE_FOLDER):
            file_path = os.path.join(IMAGE_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(IMAGE_FOLDER)

    self.summary_ready = True
    current_path = os.path.join(ORIGIN_FOLDER, selected_image)

    try:
        split_paths = split_image_to_nine(current_path, IMAGE_FOLDER)
    except Exception as e:
        print(f"切割失敗: {e}")
        return

    # 只載入當前圖片的切片
    base_name = os.path.splitext(selected_image)[0]
    self.image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".bmp")) and f.startswith(base_name)
    ])
    self.image_index = 0

    if self.image_files:
        load_image(self)