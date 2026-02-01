"""
Complete Data Pre-processing Pipeline for Zalo Traffic Sign Dataset
Based on TrafficSignDetection project's quality control steps:
1. Remove boxes with area < 40 pixel square
2. Suppress duplicate/overlapping boxes (IoU > 0.7, same class)
3. Apply data augmentation with class balancing
4. Generate high-quality training dataset

This ensures clean data BEFORE augmentation.
"""

import json
import cv2
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from collections import Counter
import shutil

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
dataset_path = "zalo_dataset"
source_images = os.path.join(dataset_path, "traffic_train/traffic_train/images")
json_file = os.path.join(
    dataset_path, "traffic_train/traffic_train/train_traffic_sign_dataset.json"
)
output_path = "zalo_clean_augmented_dataset"

print("🧹 High-Quality Data Processing Pipeline")
print("=" * 60)

# Load COCO annotations
print("\n1. Loading annotations...")
with open(json_file, "r") as f:
    coco_data = json.load(f)

annotations = coco_data["annotations"]
images = {img["id"]: img for img in coco_data["images"]}
categories = {cat["id"]: cat for cat in coco_data["categories"]}

print(f"   Original: {len(annotations)} annotations, {len(images)} images")

# =====================================================
# STEP 1: Remove small boxes (area < 40 pixel square)
# =====================================================
print("\n2. Removing small boxes (area < 40)...")
filtered_annotations = []
removed_small = 0

for ann in annotations:
    bbox = ann["bbox"]  # [x, y, w, h]
    area = bbox[2] * bbox[3]

    if area >= 40:
        filtered_annotations.append(ann)
    else:
        removed_small += 1

annotations = filtered_annotations
print(f"   Removed {removed_small} small boxes")
print(f"   Remaining: {len(annotations)} annotations")

# =====================================================
# STEP 2: Suppress duplicate/overlapping boxes
# =====================================================
print("\n3. Suppressing duplicate overlapping boxes (IoU > 0.7, same class)...")


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] format
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# Group annotations by image
image_annotations = {}
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

# Suppress overlapping boxes
cleaned_annotations = []
removed_duplicates = 0

for img_id, img_anns in image_annotations.items():
    # Group by category
    category_groups = {}
    for ann in img_anns:
        cat_id = ann["category_id"]
        if cat_id not in category_groups:
            category_groups[cat_id] = []
        category_groups[cat_id].append(ann)

    # For each category, remove overlapping boxes
    for cat_id, anns in category_groups.items():
        # Sort by area (keep larger boxes)
        anns_sorted = sorted(
            anns, key=lambda x: x["bbox"][2] * x["bbox"][3], reverse=True
        )

        kept_anns = []
        for i, ann1 in enumerate(anns_sorted):
            should_keep = True

            for ann2 in kept_anns:
                iou = calculate_iou(ann1["bbox"], ann2["bbox"])
                if iou > 0.7:  # Threshold from TrafficSignDetection project
                    should_keep = False
                    removed_duplicates += 1
                    break

            if should_keep:
                kept_anns.append(ann1)

        cleaned_annotations.extend(kept_anns)

annotations = cleaned_annotations
print(f"   Removed {removed_duplicates} duplicate boxes")
print(f"   Remaining: {len(annotations)} annotations")

# Rebuild image_annotations dict
image_annotations = {}
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

# =====================================================
# STEP 3: Analyze class distribution for balancing
# =====================================================
print("\n4. Analyzing class distribution for balancing...")
category_counts = Counter([ann["category_id"] for ann in annotations])
max_count = max(category_counts.values())
target_count = int(max_count * 0.8)  # Target 80% of max class

print(f"   Target samples per class: {target_count}")

# Calculate augmentation factor for each class
aug_factor = {}
for cat_id, count in category_counts.items():
    if count < target_count:
        aug_factor[cat_id] = int(np.ceil(target_count / count))
    else:
        aug_factor[cat_id] = 1
    cat_name = categories[cat_id]["name"]
    print(
        f"   {cat_name}: {count} → ~{count * aug_factor[cat_id]} (×{aug_factor[cat_id]})"
    )

# =====================================================
# STEP 4: Define augmentation pipeline
# =====================================================
print("\n5. Creating augmentation pipeline...")
transform = A.Compose(
    [
        # Random brightness & contrast (p=0.5)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # Random horizontal flip (p=0.5)
        A.HorizontalFlip(p=0.5),
        # Random blur (p=0.5)
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
            ],
            p=0.5,
        ),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_area=40,  # Ensure no small boxes after augmentation
        min_visibility=0.3,
    ),
)

# Crop augmentation (p=0.5)
crop_transform = A.Compose(
    [
        A.RandomSizedBBoxSafeCrop(height=500, width=1200, erosion_rate=0.2, p=1.0),
    ],
    bbox_params=A.BboxParams(
        format="coco", label_fields=["category_ids"], min_area=40, min_visibility=0.3
    ),
)

# =====================================================
# STEP 5: Apply augmentation
# =====================================================
print("\n6. Augmenting images with class balancing...")
Path(output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_path, "images")).mkdir(exist_ok=True)

new_annotations = []
new_images = []
ann_id = 0
img_id = 0

for orig_img_id, img_info in tqdm(images.items(), desc="Processing"):
    if orig_img_id not in image_annotations:
        continue

    img_anns = image_annotations[orig_img_id]

    # Determine augmentation count
    class_ids = [ann["category_id"] for ann in img_anns]
    max_aug = max([aug_factor[cid] for cid in class_ids])

    # Load image
    img_path = os.path.join(source_images, img_info["file_name"])
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare boxes
    bboxes = [ann["bbox"] for ann in img_anns]
    category_ids = [ann["category_id"] for ann in img_anns]

    # Save original
    orig_filename = f"{img_id:06d}.png"
    cv2.imwrite(
        os.path.join(output_path, "images", orig_filename),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )

    new_images.append(
        {
            "id": img_id,
            "file_name": orig_filename,
            "height": image.shape[0],
            "width": image.shape[1],
        }
    )

    for bbox, cat_id in zip(bboxes, category_ids):
        new_annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        ann_id += 1

    img_id += 1

    # Apply augmentations
    for aug_idx in range(max_aug - 1):
        try:
            # Apply crop with 50% probability
            if random.random() < 0.5 and len(bboxes) > 0:
                transformed = crop_transform(
                    image=image, bboxes=bboxes, category_ids=category_ids
                )
            else:
                transformed = {
                    "image": image,
                    "bboxes": bboxes,
                    "category_ids": category_ids,
                }

            # Apply other augmentations
            transformed = transform(
                image=transformed["image"],
                bboxes=transformed["bboxes"],
                category_ids=transformed["category_ids"],
            )

            if len(transformed["bboxes"]) == 0:
                continue

            # Save augmented image
            aug_filename = f"{img_id:06d}.png"
            cv2.imwrite(
                os.path.join(output_path, "images", aug_filename),
                cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR),
            )

            new_images.append(
                {
                    "id": img_id,
                    "file_name": aug_filename,
                    "height": transformed["image"].shape[0],
                    "width": transformed["image"].shape[1],
                }
            )

            for bbox, cat_id in zip(transformed["bboxes"], transformed["category_ids"]):
                new_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1

            img_id += 1

        except Exception as e:
            continue

# =====================================================
# STEP 6: Create COCO JSON
# =====================================================
print("\n7. Creating COCO annotation file...")
new_coco_data = {
    "info": {
        "description": "Zalo Traffic Sign Dataset - Cleaned & Augmented",
        "version": "2.0",
        "year": 2024,
    },
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco_data["categories"],
}

output_json = os.path.join(output_path, "annotations.json")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(new_coco_data, f, ensure_ascii=False, indent=2)

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n8. Final Summary:")
print(f"   Data Quality Steps:")
print(f"     • Removed {removed_small} small boxes (< 40px²)")
print(f"     • Removed {removed_duplicates} duplicate boxes (IoU > 0.7)")
print(f"   ")
print(f"   Dataset Statistics:")
print(
    f"     • Original: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations"
)
print(f"     • After cleaning: {len(images)} images, {len(annotations)} annotations")
print(
    f"     • After augmentation: {len(new_images)} images, {len(new_annotations)} annotations"
)
print(
    f"     • Increase: +{len(new_images) - len(images)} images ({(len(new_images) / len(images) - 1) * 100:.1f}%)"
)

# New class distribution
new_category_counts = Counter([ann["category_id"] for ann in new_annotations])
print(f"\n   Final class distribution:")
for cat_id in sorted(new_category_counts.keys()):
    cat_name = categories[cat_id]["name"]
    old_count = category_counts[cat_id]
    new_count = new_category_counts[cat_id]
    print(f"     {cat_name}: {old_count} → {new_count} (+{new_count - old_count})")

print(f"\n✅ High-quality dataset saved to: {output_path}")
print("=" * 60)
print("\n📦 Ready for Kaggle upload!")
