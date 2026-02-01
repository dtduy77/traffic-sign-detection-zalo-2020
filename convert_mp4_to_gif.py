import cv2
import glob
import os
from PIL import Image

def main():
    # Directories
    input_dir = "videos"
    output_dir = "videos/result"
    os.makedirs(output_dir, exist_ok=True)

    # Config for optimization
    skip_frames = 2   # Keep 1 out of every N frames (higher = smaller file / faster playback)
    resize_factor = 0.5 # Resize to 50% of original
    duration_ms = 66  # ~15 FPS

    print(f"🚀 Converting MP4s in '{input_dir}' to GIFs in '{output_dir}'...")

    mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))

    if not mp4_files:
        print("❌ No MP4 files found.")
        return

    for video_path in mp4_files:
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name_no_ext}_origin.gif")

        # Check if already exists? (Optional)
        # if os.path.exists(output_path):
        #     continue

        print(f"   🎬 Processing: {filename}...")

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Resize
            if resize_factor != 1.0:
                height, width = frame.shape[:2]
                new_dim = (int(width * resize_factor), int(height * resize_factor))
                frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)

            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

            # Safety limit (GIFs can get huge)
            if len(frames) > 300:
                print("      ⚠️ Truncating to 300 frames to keep file size reasonable.")
                break

        cap.release()

        if frames:
            print(f"      💾 Saving GIF ({len(frames)} frames)...")
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=duration_ms,
                loop=0
            )
            print(f"      ✅ Saved: {output_path}")
        else:
            print(f"      ❌ Failed to read frames from {filename}")

    print("\n✅ Conversion Complete!")

if __name__ == "__main__":
    main()
