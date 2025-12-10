import argparse
import ffmpeg
import subprocess
import re
import os
import sys
import concurrent.futures
import threading

# Add the parent directory to sys.path to import VideoParams
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from VideoParams import VideoParams

# python apply_uniform_edits.py --brightness 76 --contrast 200 --saturation 0 --crop 866,289,414,208

output_folder = "/home/sully/Desktop/projects/VideoWeightTool/videos/uniform_edits"
os.makedirs(output_folder, exist_ok=True)

vid_dir = "/home/sully/Downloads/scale_videos"

# Global lock for thread-safe printing
print_lock = threading.Lock()
# Dictionary to track which line each video is on
video_lines = {}
total_videos = 0

def is_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def update_progress(base_name, status):
    """Thread-safe progress update for a specific video."""
    with print_lock:
        if base_name not in video_lines:
            return
        line_num = video_lines[base_name]
        lines_up = total_videos - line_num
        # Move cursor up, clear line, print, move back down
        sys.stdout.write(f"\033[{lines_up}A")  # Move up
        sys.stdout.write(f"\033[2K")            # Clear line
        sys.stdout.write(f"{status}\n")         # Print status
        sys.stdout.write(f"\033[{lines_up - 1}B")  # Move back down
        sys.stdout.flush()

def process_video_with_params(input_path, params: VideoParams, out_folder):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_name = f"UE_{base_name}.mp4"
    out_path = os.path.join(out_folder, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)

    stream = ffmpeg.input(input_path)

    try:
        probe = ffmpeg.probe(input_path)
        duration = float(probe['streams'][0]['duration'])
    except Exception:
        duration = None

    # Crop if provided
    if params.crop_coords:
        x, y, w, h = params.crop_coords
        stream = ffmpeg.filter(stream, 'crop', w, h, x, y)

    # Convert integer video params to ffmpeg eq filter floats
    adj_brightness = params.brightness / 255.0   # ffmpeg expects -1..1
    adj_saturation = params.saturation / 100.0   # 100 => 1.0
    adj_contrast = params.contrast / 100.0       # 100 => 1.0

    stream = ffmpeg.filter(stream, 'eq',
                           brightness=adj_brightness,
                           contrast=adj_contrast,
                           saturation=adj_saturation)

    stream = ffmpeg.output(stream, out_path,
                           vcodec='libx264',
                           acodec='aac',
                           preset='superfast',
                           crf=28,
                           threads=4)

    cmd = ffmpeg.compile(stream, overwrite_output=True)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        # read stderr for progress info
        for line in proc.stderr: # type: ignore
            # parse 'time=HH:MM:SS.xx' from ffmpeg stderr
            tm = re.search(r'time=(\d+):(\d{2}):(\d{2}\.\d+)', line)
            if tm and duration:
                hours, minutes, seconds = tm.groups()
                current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                pct = int((current_time / duration) * 100)
                pct = min(pct, 99)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                update_progress(base_name, f"[{base_name[:30]}] [{bar}] {pct:3}%")
        proc.wait()
        if proc.returncode != 0:
            update_progress(base_name, f"[{base_name[:30]}] FAILED (code {proc.returncode})")
            if os.path.exists(out_path):
                os.remove(out_path)
            return False
        update_progress(base_name, f"[{base_name[:30]}] Completed")
        return True
    except KeyboardInterrupt:
        try:
            proc.terminate()
            proc.wait()
        except Exception:
            pass
        if os.path.exists(out_path):
            os.remove(out_path)
        update_progress(base_name, f"[{base_name[:30]}] Cancelled")
        return False
    except Exception as e:
        update_progress(base_name, f"[{base_name[:30]}] Error: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False

def parse_crop_arg(val):
    try:
        parts = [int(p) for p in val.split(',')]
        if len(parts) != 4:
            raise ValueError()
        return tuple(parts)
    except Exception:
        raise argparse.ArgumentTypeError("Crop must be 'x,y,w,h'")

def cli_main():
    global total_videos, video_lines
    parser = argparse.ArgumentParser(description="Apply uniform edits to all videos in a directory.")
    parser.add_argument("--brightness", "-b", type=int, default=0, help="Brightness [-255..255] (0 = normal)")
    parser.add_argument("--contrast", "-c", type=int, default=100, help="Contrast [0..200] (100 = normal)")
    parser.add_argument("--saturation", "-s", type=int, default=100, help="Saturation [0..300] (100 = normal)")
    parser.add_argument("--crop", type=parse_crop_arg, default=None, help="Crop as 'x,y,w,h'")
    parser.add_argument("--input-dir", "-i", default=vid_dir, help="Input video directory")
    parser.add_argument("--output-dir", "-o", default=output_folder, help="Output directory")
    args = parser.parse_args()

    if not is_ffmpeg_installed():
        print("ffmpeg not found in PATH. Install ffmpeg to use this tool.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Build VideoParams for uniform edits & validate using existing logic
    params = VideoParams()
    params.brightness = args.brightness
    params.contrast = args.contrast
    params.saturation = args.saturation
    params.crop_coords = args.crop
    try:
        params.validate()
    except Exception as e:
        print(f"Invalid params: {e}")
        return 2

    # Collect video files
    video_files = []
    for filename in sorted(os.listdir(args.input_dir)):
        infile = os.path.join(args.input_dir, filename)
        if not os.path.isfile(infile):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ('.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg'):
            print(f"Skipping non-video file: {filename}")
            continue
        video_files.append(infile)

    # Initialize progress display
    total_videos = len(video_files)
    print(f"Processing {total_videos} videos...\n")
    
    for i, infile in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(infile))[0]
        video_lines[base_name] = i
        print(f"[{base_name[:30]}] [{'░' * 20}]   0%")

    # Process in parallel (adjust max_workers to your CPU cores)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_video_with_params, infile, params, args.output_dir): infile for infile in video_files}
        for future in concurrent.futures.as_completed(futures):
            infile = futures[future]
            filename = os.path.basename(infile)
            try:
                future.result()
            except Exception as e:
                print(f"\nError processing {filename}: {e}")

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())