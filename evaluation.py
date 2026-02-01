import os
import sys
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "zalo_traffic_signs_v2_saved/weights/best.pt")
INPUT_DIR = os.path.join(
    BASE_DIR, "zalo_dataset/traffic_public_test/traffic_public_test/images"
)
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "result_side_by_side")

# Đường dẫn Font chữ Tiếng Việt trên macOS
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/Library/Fonts/Arial.ttf"
# Nếu vẫn không thấy, bạn hãy copy file Arial.ttf vào cùng thư mục code và sửa thành:
# FONT_PATH = os.path.join(BASE_DIR, "Arial.ttf")

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# HÀM SINH MÀU TỰ ĐỘNG (Cho từng Class)
# ---------------------------------------------------------
def generate_class_colors(num_classes):
    """
    Sinh ra một danh sách các màu (BGR) khác nhau dựa trên số lượng class.
    Sử dụng không gian màu HSV để tạo màu rực rỡ và khác biệt.
    """
    colors = []
    for i in range(num_classes):
        # Hue chạy từ 0 đến 179 trong OpenCV
        hue = int(180 * i / num_classes)
        # Saturation và Value đặt cao để màu tươi sáng
        hsv_color = np.uint8([[[hue, 255, 255]]])
        # Chuyển từ HSV sang BGR (để dùng cho OpenCV)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # Lưu dưới dạng tuple số nguyên (B, G, R)
        colors.append(tuple(map(int, bgr_color)))
    return colors


# ---------------------------------------------------------
# HÀM VẼ CHỮ TIẾNG VIỆT (Dùng PIL)
# ---------------------------------------------------------
def put_text_vietnamese(
    img_cv, text, position, font_path, font_size, color_bgr, thickness=0
):
    # Chuyển BGR (OpenCV) -> RGB (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"⚠️ Không tìm thấy font {font_path}, dùng font mặc định.")
        font = ImageFont.load_default()

    x, y = position
    # Chuyển màu đầu vào từ BGR sang RGB cho PIL
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    if thickness > 0:
        # Vẽ viền chữ (màu tương phản: trắng hoặc đen)
        outline_color = (0, 0, 0) if sum(color_rgb) > 400 else (255, 255, 255)
        offset = thickness
        # Vẽ 4 hướng để tạo viền
        draw.text((x - offset, y), text, font=font, fill=outline_color)
        draw.text((x + offset, y), text, font=font, fill=outline_color)
        draw.text((x, y - offset), text, font=font, fill=outline_color)
        draw.text((x, y + offset), text, font=font, fill=outline_color)

    # Vẽ chữ chính
    draw.text(position, text, font=font, fill=color_rgb)

    # Chuyển lại RGB -> BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------
# HÀM VẼ NHÃN THÔNG MINH (CẬP NHẬT MÀU SẮC)
# ---------------------------------------------------------
# Thêm tham số class_colors vào hàm
def draw_smart_labels(image, boxes, class_names, class_colors):
    h, w, _ = image.shape
    sorted_indices = np.argsort(boxes.xyxy[:, 1].cpu().numpy())
    occupied_areas = []

    for idx in sorted_indices:
        box = boxes[idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())

        label_text = f"{class_names[cls_id]} {conf:.2f}"

        # --- LẤY MÀU RIÊNG CHO CLASS NÀY ---
        # Đảm bảo ID không vượt quá số lượng màu đã sinh
        color_bgr = class_colors[cls_id % len(class_colors)]

        # 1. Vẽ Box với màu riêng
        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 3)

        # 2. Tính toán vị trí nhãn
        font_size = 24
        text_h = 30  # Ước lượng chiều cao chữ

        label_y_bottom = y1 - 5
        label_y_top = label_y_bottom - text_h

        # 3. Thuật toán "né" nhãn
        is_overlapping = True
        while is_overlapping and label_y_top > 0:
            is_overlapping = False
            for occupied_y_top, occupied_y_bottom in occupied_areas:
                if not (
                    label_y_bottom < occupied_y_top or label_y_top > occupied_y_bottom
                ):
                    label_y_bottom = occupied_y_top - 5
                    label_y_top = label_y_bottom - text_h
                    is_overlapping = True
                    break
        occupied_areas.append((label_y_top, label_y_bottom))

        # 4. Vẽ Nhãn Tiếng Việt với MÀU RIÊNG
        label_x1 = max(x1, 0)
        label_y_top = max(label_y_top, 0)

        # Truyền color_bgr vào hàm vẽ chữ
        image = put_text_vietnamese(
            image,
            label_text,
            (label_x1, label_y_top),
            FONT_PATH,
            font_size,
            color_bgr,
            thickness=2,
        )

    return image


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------
print(f"📍 Base Dir: {BASE_DIR}")

if not os.path.exists(FONT_PATH):
    print(f"⚠️ CẢNH BÁO: Không tìm thấy font tại {FONT_PATH}. Vui lòng kiểm tra lại.")

if not os.path.exists(INPUT_DIR):
    print(f"❌ Lỗi: Không tìm thấy Input Dir: {INPUT_DIR}")
    sys.exit()

input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.[pPjJ][nNpP][gG]*")))
if not input_files:
    print("❌ Folder đầu vào trống!")
    sys.exit()

print("🚀 Đang load model...")
model = YOLO(MODEL_PATH)

# --- SINH BẢNG MÀU ---
num_classes = len(model.names)
CLASS_COLORS_BGR = generate_class_colors(num_classes)
print(f"🎨 Đã tạo {num_classes} màu sắc riêng biệt cho các lớp.")


print(f"\n--- BẮT ĐẦU XỬ LÝ {len(input_files)} ẢNH ---")
# Dùng tqdm để hiện thanh tiến trình
for i, img_path in enumerate(tqdm(input_files, desc="Processing")):
    filename = os.path.basename(img_path)
    output_path = os.path.join(FINAL_OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        continue

    img_origin = cv2.imread(img_path)
    if img_origin is None:
        continue

    results = model(img_origin, conf=0.4, verbose=False)

    img_predicted = img_origin.copy()
    # Truyền thêm bảng màu vào hàm vẽ
    img_predicted = draw_smart_labels(
        img_predicted, results[0].boxes, model.names, CLASS_COLORS_BGR
    )

    # Tiêu đề ảnh (Màu cố định: Đỏ cho gốc, Xanh cho dự đoán)
    img_origin = put_text_vietnamese(
        img_origin, "ẢNH GỐC", (30, 50), FONT_PATH, 40, (0, 0, 255), thickness=3
    )
    img_predicted = put_text_vietnamese(
        img_predicted, "DỰ ĐOÁN", (30, 50), FONT_PATH, 40, (0, 255, 0), thickness=3
    )

    composite_img = np.hstack((img_origin, img_predicted))
    cv2.imwrite(output_path, composite_img)

print(f"\n✅ Đã xử lý xong! Kiểm tra folder: {FINAL_OUTPUT_DIR}")

# ---------------------------------------------------------
# PHẦN 2: TRÌNH XEM ẢNH (VIEWER) - Xem lại các ảnh vừa lưu
# ---------------------------------------------------------
print("\n--- TRÌNH DUYỆT KẾT QUẢ SIDE-BY-SIDE ---")
print("👉 Phím D: Ảnh sau | A: Ảnh trước | Q: Thoát")

output_files = sorted(glob.glob(os.path.join(FINAL_OUTPUT_DIR, "*.[pPjJ][nNpP][gG]*")))

if not output_files:
    print("❌ Không có ảnh kết quả nào để hiển thị.")
    sys.exit()

current_index = 0
total = len(output_files)

while True:
    img_path = output_files[current_index]
    img = cv2.imread(img_path)

    if img is None:
        current_index += 1
        continue

    # Vẽ số thứ tự ảnh lên góc trên cùng giữa
    label_idx = f"{current_index + 1}/{total}"
    cv2.putText(
        img,
        label_idx,
        (img.shape[1] // 2 - 50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 0),
        4,
    )

    # Hiển thị (Cửa sổ này sẽ rất rộng vì là 2 ảnh ghép lại)
    # Dùng WINDOW_NORMAL để có thể resize cửa sổ nếu nó quá to so với màn hình
    cv2.namedWindow("Ket qua Side-by-Side", cv2.WINDOW_NORMAL)
    cv2.imshow("Ket qua Side-by-Side", img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("d") or key == 83:  # Next
        current_index += 1
    elif key == ord("a") or key == 81:  # Back
        current_index -= 1
    elif key == ord("q") or key == 27:  # Quit
        break

    if current_index >= total:
        current_index = 0
    elif current_index < 0:
        current_index = total - 1

cv2.destroyAllWindows()
cv2.waitKey(1)
