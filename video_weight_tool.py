#!/usr/bin/env python3
"""
Video Weight Tool - Master TUI
A single entry point for the entire data lifecycle.
"""

import sys
import os
import subprocess
import time

# Ensure the root directory is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'DIM': '\033[2m',
    'ENDC': '\033[0m',
}

TITLE_WDITH = 40

ELEMENTS = {
    'SEPARATOR': f"\n{COLORS['DIM']}{'─' * 60}{COLORS['ENDC']}\n",
    'TOP_LINE': f"\n{COLORS['HEADER']}{'=' * TITLE_WDITH}",
    'BOT_LINE': f"{'=' * TITLE_WDITH}{COLORS['ENDC']}"
}

def stylized_input(prompt, default=None):
    if default:
        prompt_text = f"{COLORS['YELLOW']}{prompt}{COLORS['ENDC']} [{COLORS['DIM']}{default}{COLORS['ENDC']}]: "
    else: 
        prompt_text = f"{COLORS['YELLOW']}{prompt}{COLORS['ENDC']}: "

    response = input(prompt_text).strip()
    return response if response else default

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title_text, breadcrumb_path=None):
    clear_screen()

    print(ELEMENTS['TOP_LINE'])
    print(f"{title_text.center(TITLE_WDITH)}")
    print(ELEMENTS['BOT_LINE'])

    if breadcrumb_path:
        print(f"\n{COLORS['DIM']}{" > ".join(breadcrumb_path)}{COLORS['ENDC']}")

def run_script(path, args=[]):
    """Runs a python script in a subprocess"""
    full_path = os.path.join(ROOT_DIR, path)
    if not os.path.exists(full_path):
        print(f"{COLORS['RED']}Error: Script not found at {full_path}{COLORS['ENDC']}")
        return
    
    cmd = [sys.executable, full_path] + args
    print(f"{COLORS['BLUE']}Running: {' '.join(cmd)}{COLORS['ENDC']}")

    time.sleep(1)

    try:
        result = subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        print(f"\n{COLORS['GREEN']} Process completed successfully{COLORS['ENDC']}")
    except subprocess.CalledProcessError as e:
        print(f"\n{COLORS['RED']} Process failed with exit code {e.returncode}{COLORS['ENDC']}")
    except KeyboardInterrupt:
        print(f"\n{COLORS['YELLOW']} Process interrupted{COLORS['ENDC']}")
    
    input(f"\n{COLORS['DIM']}Press Enter to continue...{COLORS['ENDC']}")

def menu_labelling():
    breadcrumb = ["Main Menu", "Labelling & Data Entry"]

    while True:
        print_header("Labelling & Data Entry", breadcrumb)
        print("1. Standard labelling (Batch Manager)")
        print("2. Create ROIs")
        print("3. Manual Cropping (Legacy/Correction)")
        print(ELEMENTS['SEPARATOR'])
        print("0. Back to Main Menu")
        print()
        
        choice = stylized_input("Select option")
        
        if choice == '1':
            video_dir = stylized_input("Video Directory", "data/raw_videos")
            run_script("workflows/labelling/main.py", ["--video-dir", video_dir])
        elif choice == '2':
            video_dir = stylized_input("Video Directory", "data/raw_videos")
            run_script("workflows/labelling/main.py", ["--rois-only", "--video-dir", video_dir])
        elif choice == '3':
            run_script("workflows/manual/main.py")
        elif choice == '0':
            break

def menu_pipeline():
    breadcrumb = ["Main Menu", "Data Processing Pipelines"]
    while True:
        print_header("Data Processing Pipelines", breadcrumb)
        print("1. Extract Frames from Videos (using Labels)")
        print("2. Split Dataset (Train/Val/Test)")
        print("3. Check Dataset Stats")
        print("4. Cleanup (Remove missing files from CSVs)")
        print(ELEMENTS['SEPARATOR'])
        print("0. Back to Main Menu")
        print()
        
        choice = stylized_input("Select option")
        
        if choice == '1':
            run_script("pipelines/preparation/extract.py")
        elif choice == '2':
            run_script("pipelines/preparation/split.py")
        elif choice == '3':
            run_script("pipelines/maintenance/stats.py")
        elif choice == '4':
            run_script("pipelines/maintenance/cleanup.py")
        elif choice == '0':
            break

def menu_training():
    breadcrumb = ["Main Menu", "Training & Inference"]
    while True:
        print_header("Model Training & Inference", breadcrumb)

        print("1. Train New Model")
        print("2. Evaluate training results")
        print("3. Run Inference on Video")
        print(ELEMENTS['SEPARATOR'])
        print("0. Back to Main Menu")
        print()
        
        choice = stylized_input("Select option")
        
        if choice == '1':
            epochs = stylized_input("Number of epochs", "50")
            try:
                epochs_val = int(epochs)
                if epochs_val <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                print(f"{COLORS['YELLOW']}Invalid epochs value; using default 50{COLORS['ENDC']}")
                epochs_val = 50
            run_script("pipelines/train.py", ["--epochs", str(epochs_val)])

        elif choice == '2':
                    run_script("pipelines/evaluate.py")

        elif choice == '3':
            video_path = stylized_input("Video Path")
            if not video_path: 
                print(f"{COLORS['RED']}Invalid choice. Please try again...{COLORS['ENDC']}")
                input(f"\n{COLORS['DIM']}Press Enter to continue...{COLORS['ENDC']}")
                continue
                
            args = ["--video", video_path]

            while True:
                conservative_run = stylized_input("Would you like to run inference conservatively? (y/n)", "y").strip().lower()
                if conservative_run in ('y','n'):
                    break
                print(f"{COLORS['RED']}Please answer 'y' or 'n'.{COLORS['ENDC']}")
                    
            if conservative_run == 'y':
                args.append("--conservative")
            
            run_script("pipelines/inference.py", ["--video", args])

        elif choice == '0':
            break

def main():
    while True:
        print_header("Video Weight Tool - Master TUI", ["Main Menu"])
        print("1. Labelling & Data Entry")
        print("2. Data Processing Pipelines")
        print("3. Training & Inference")
        print(ELEMENTS['SEPARATOR'])
        print("0. Exit")
        print()
        
        choice = stylized_input("Select option")
        
        if choice == '1':
            menu_labelling()
        elif choice == '2':
            menu_pipeline()
        elif choice == '3':
            menu_training()
        elif choice == '0':
            print(f"{COLORS['GREEN']}Goodbye.{COLORS['ENDC']}")
            sys.exit(0)
        else:
            print(f"{COLORS['RED']}Invalid choice. Please try again.{COLORS['ENDC']}")
            input(f"\n{COLORS['DIM']}Press Enter to continue...{COLORS['ENDC']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{COLORS['YELLOW']}Operation Cancelled by user\n{COLORS['ENDC']}")
