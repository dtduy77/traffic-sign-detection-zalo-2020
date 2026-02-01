import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "zalo_traffic_signs_v2_saved/weights/best.pt")
INPUT_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(INPUT_DIR, "result")

# Font for Vietnamese support
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/Library/Fonts/Arial.ttf"
if not os.path.exists(FONT_PATH):
    # Fallback or local path
    FONT_PATH = "Arial.ttf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS (Borrowed from evaluation.py)
# ---------------------------------------------------------
def generate_class_colors(num_classes):
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color)))
    return colors

def put_text_vietnamese(img_cv, text, position, font_path, font_size, color_bgr, thickness=0):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    x, y = position
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    if thickness > 0:
        outline_color = (0, 0, 0) if sum(color_rgb) > 400 else (255, 255, 255)
        offset = thickness
        draw.text((x - offset, y), text, font=font, fill=outline_color)
        draw.text((x + offset, y), text, font=font, fill=outline_color)
        draw.text((x, y - offset), text, font=font, fill=outline_color)
        draw.text((x, y + offset), text, font=font, fill=outline_color)

    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_smart_labels(image, boxes, class_names, class_colors):
    if boxes is None:
        return image

    sorted_indices = np.argsort(boxes.xyxy[:, 1].cpu().numpy())
    occupied_areas = []

    for idx in sorted_indices:
        box = boxes[idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())

        label_text = f"{class_names[cls_id]} {conf:.2f}"
        color_bgr = class_colors[cls_id % len(class_colors)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 3)

        font_size = 20
        text_h = 25
        label_y_bottom = y1 - 5
        label_y_top = label_y_bottom - text_h

        is_overlapping = True
        while is_overlapping and label_y_top > 0:
            is_overlapping = False
            for occupied_y_top, occupied_y_bottom in occupied_areas:
                if not (label_y_bottom < occupied_y_top or label_y_top > occupied_y_bottom):
                    label_y_bottom = occupied_y_top - 5
                    label_y_top = label_y_bottom - text_h
                    is_overlapping = True
                    break
        occupied_areas.append((label_y_top, label_y_bottom))

        label_x1 = max(x1, 0)
        label_y_top = max(label_y_top, 0)

        image = put_text_vietnamese(
            image, label_text, (label_x1, label_y_top),
            FONT_PATH, font_size, color_bgr, thickness=2
        )
    return image

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
import argparse

def main():
    parser = argparse.ArgumentParser(description="Detect traffic signs in a video and save as GIF.")
    parser.add_argument("video_path", type=str, help="Path to the input video file (e.g., videos/test.mp4)")
    args = parser.parse_args()

    video_path = args.video_path

    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    print("🚀 Loading model...")
    model = YOLO(MODEL_PATH)
    num_classes = len(model.names)
    class_colors = generate_class_colors(num_classes)

    # Output directory: created inside the same folder as the video, named 'result'
    # Or strict 'videos/result' if preferred. Let's use the parent folder + 'result'
    parent_dir = os.path.dirname(os.path.abspath(video_path))
    output_dir = os.path.join(parent_dir, "result")
    os.makedirs(output_dir, exist_ok=True)

    print(f"📹 Processing: {video_path}")

    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    output_gif_path = os.path.join(output_dir, f"{name_no_ext}_result.gif")

    cap = cv2.VideoCapture(video_path)
    frames = []

    # Limit frames for GIF size
    frame_count = 0
    skip_frames = 2 # Process every Nth frame to keep GIF smaller/faster

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Predict
        results = model(frame, conf=0.4, verbose=False)

        # Draw
        annotated_frame = frame.copy()
        annotated_frame = draw_smart_labels(annotated_frame, results[0].boxes, model.names, class_colors)

        # Convert to RGB for PIL GIF
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

        # Limit max frames to prevent memory crash on huge videos
        if len(frames) > 300:
            print("      ⚠️ Video too long, truncating GIF to 300 frames.")
            break

    cap.release()

    if frames:
        print(f"      💾 Saving GIF ({len(frames)} frames)...")
        # Duration is time per frame in ms. 1000ms / 15fps ~ 66ms
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=66,
            loop=0
        )
        print(f"      ✅ Saved: {output_gif_path}")
    else:
        print("      ⚠️ No frames read.")

    print(f"\n✅ Done! Result saved at: {output_gif_path}")

if __name__ == "__main__":
    main()
