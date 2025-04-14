import argparse
import os
import time
import cv2
from core.render_3d import render_sbs_3d, aspect_ratios


def parse_args():
    parser = argparse.ArgumentParser(description="🎬 VisionDepth3D Studio CLI")

    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--depth", required=True, help="Depth map video path")
    parser.add_argument("--output", help="Output path (auto-names if not provided)")

    parser.add_argument("--format", default="Half-SBS",
                        choices=["Full-SBS", "Half-SBS", "VR", "Passive Interlaced", "Red-Cyan Anaglyph"])
    parser.add_argument("--aspect", default="Default (16:9)", help="Aspect ratio label from GUI")

    parser.add_argument("--codec", default="XVID", help="OpenCV codec (e.g., 'XVID', 'mp4v')")
    parser.add_argument("--ffmpeg", action="store_true", help="Use FFmpeg instead of OpenCV writer")
    parser.add_argument("--ffmpeg_codec", default="libx264", help="FFmpeg codec name (e.g., 'h264_nvenc')")
    parser.add_argument("--crf", type=int, default=23, help="CRF quality (0-51, lower = better)")

    parser.add_argument("--fg_shift", type=float, default=10.0)
    parser.add_argument("--mg_shift", type=float, default=-2.5)
    parser.add_argument("--bg_shift", type=float, default=-5.0)
    parser.add_argument("--sharpness", type=float, default=0.15)

    parser.add_argument("--feather", type=float, default=10.0)
    parser.add_argument("--blur", type=int, default=9)

    parser.add_argument("--width", type=int, help="Override output width")
    parser.add_argument("--height", type=int, help="Override output height")
    parser.add_argument("--fps", type=float, help="Override FPS")

    parser.add_argument("--no_track", action="store_true")
    parser.add_argument("--no_floating", action="store_true")
    parser.add_argument("--preserve_content", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Preview settings without rendering")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def auto_generate_output_path(input_path, fmt, aspect_label, width, height):
    base = os.path.splitext(os.path.basename(input_path))[0]
    suffix = f"_{fmt.replace('-', '')}_{aspect_label.replace(' ', '').replace(':', '').replace('(', '').replace(')', '')}_{width}x{height}.mp4"
    return os.path.join(os.getcwd(), base + suffix)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if args.width: width = args.width
    if args.height: height = args.height
    if args.fps: fps = args.fps

    # Generate output filename if not provided
    if not args.output:
        args.output = auto_generate_output_path(args.input, args.format, args.aspect, width, height)

    if args.verbose:
        print(f"🎞 Input: {args.input}")
        print(f"🧠 Depth: {args.depth}")
        print(f"📤 Output: {args.output}")
        print(f"🧾 Format: {args.format}, Aspect: {args.aspect}")
        print(f"📏 Resolution: {width}x{height}, FPS: {fps}")
        print(f"🎯 FG: {args.fg_shift}, MG: {args.mg_shift}, BG: {args.bg_shift}, Sharp: {args.sharpness}")
        print(f"🧪 Feather: {args.feather}, Blur: {args.blur}, CRF: {args.crf}")
        print(f"🎯 FFmpeg: {args.ffmpeg} ({args.ffmpeg_codec})")

    if args.dry_run:
        print("🧪 Dry run complete. Exiting before render.")
        return

    # Build mock aspect selector
    aspect_mock = type("Mock", (), {"get": lambda self: args.aspect})()

    start = time.time()
    render_sbs_3d(
        input_path=args.input,
        depth_path=args.depth,
        output_path=args.output,
        codec=args.codec,
        fps=fps,
        width=width,
        height=height,
        fg_shift=args.fg_shift,
        mg_shift=args.mg_shift,
        bg_shift=args.bg_shift,
        sharpness_factor=args.sharpness,
        output_format=args.format,
        selected_aspect_ratio=aspect_mock,
        aspect_ratios=aspect_ratios,
        feather_strength=args.feather,
        blur_ksize=args.blur,
        use_ffmpeg=args.ffmpeg,
        selected_ffmpeg_codec=args.ffmpeg_codec,
        crf_value=args.crf,
        use_subject_tracking=not args.no_track,
        use_floating_window=not args.no_floating,
        preserve_content=args.preserve_content,
        progress=None,
        progress_label=None,
        suspend_flag=None,
        cancel_flag=None,
    )
    print(f"✅ Render finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
