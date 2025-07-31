# config.py

# === 資料夾設定 ===
IMAGE_FOLDER = "./image/split"
ORIGIN_FOLDER ="./image/origin"    #原圖的位置
CLUSTER_FOLDER = "./sorted/"
SUMMARY_OUTPUT_PATH = "./output_summary.png"
ESRGAN_OUTPUT_PATH = "./image/ESRGAN"

# === 字體設定 ===
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"

# === Summary小圖格設定 ===
CELL_SIZE = 64    # 單格尺寸 (px)
MAX_PER_ROW = 10    # 每一列最大幾個
MAX_ROWS = 6        # 總共幾列
NUM_CLASSES = 6     # 類別數量
SCALING_FACTOR = 0.5  # Summary圖片縮小比例

# === 類別名稱 ===
CLASS_NAMES = {
    1: "Strong/Activate",
    2: "Strong/Inactivate",
    3: "Medium/Activate",
    4: "Medium/Inactivate",
    5: "Weak/Activate",
    6: "Weak/Inactivate",
}

# === 中文版類別名稱 (統計表用) ===
CLASS_NAMES_CH = {
    1: "強 / 活化  ",
    2: "強 / 非活化",
    3: "中 / 活化  ",
    4: "中 / 非活化",
    5: "弱 / 活化  ",
    6: "弱 / 非活化",
}

# === 統計表 標籤 ===
ROW_LABELS = ["Strong", "Med", "Weak"]
COL_LABELS = ["Activate", "Inactivate"]
