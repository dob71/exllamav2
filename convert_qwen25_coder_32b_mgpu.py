import os
import subprocess
import argparse
import time
import glob
import sys
from pathlib import Path
import tempfile
import atexit

# Runs under Windows only (tested under Win 11).
# This script is for running quantization jobs on multiple GPUs.
# It was only tested under Windows quantizing Qwen2.5-Coder
# using the long context input and foocusing on the code (more
# c++, python code samples). Only 24GB cards were available,
# so the longest context tested was 10000 tokens (the original
# code and datasets were limited to 2048).
# Usage: update the Configuration and run it.
# Note: decent results w/ 8bit quantization and cache for the 32B
#       main model and 4bit for 7B draft, contex 128K, fits to
#       3 24GB GPUs (with manually tweaked gpu_split)

# Configuration
MODEL_NAME = "Qwen2.5-Coder-32B-Instruct"
IN_DIR = f"d:/hf_models/{MODEL_NAME}"
OUT_DIR = f"d:/tabby_models/{MODEL_NAME}"
#ROPE_OPT = "-rs 4 -ra 1"  # Comment out to disable
NO_RECOVER_OPT = "-nr"
LEN_OPT = "-ml 32768 -l 8192 -cdf 4.0"
#LEN_OPT = "-ml 32768 -l 10000 -cdf 4.0"
QUANTS = ["4.0", "6.0", "8.0"]
QUANTS_HEAD = "8"
TEMP_DIR = "d:/tabby_models/temp"
FORCE_MEASURE = False
FORCE_RECOVER = False
FORCE_PAUSE = False
FORCE_MGPUs = []
SELECTED_GPU = 0

TEMP = tempfile.gettempdir()

# set up exit handler to clean up "running" files
CLEANUP_FILE = None
def exit_handler():
    if not CLEANUP_FILE is None:
        print(f"Exit handler removing: {CLEANUP_FILE}")

def parse_arguments():
    global FORCE_MEASURE, FORCE_RECOVER, FORCE_PAUSE, SELECTED_GPU, FORCE_MGPUs, QUANTS, TEMP
    parser = argparse.ArgumentParser(description="Model quantization script")
    parser.add_argument("-measure", action="store_true", help="Force measurement generation")
    parser.add_argument("-retry", action="store_true", help="Disable recovery")
    parser.add_argument("-pause", action="store_true", help="Pause after operations")
    parser.add_argument("-gpu", type=str, help="Specify GPU(s) to use (e.g., '1' or '0,1')")
    parser.add_argument("-q", type=str, help="Specify quantization levels (space-separated)")
    args = parser.parse_args()

    if args.measure:
        FORCE_MEASURE = True
    if args.retry:
        FORCE_RECOVER = True
    if args.pause:
        FORCE_PAUSE = True
    if args.gpu:
        SELECTED_GPU = args.gpu
        FORCE_MGPUs = args.gpu.split(",") if "," in args.gpu else [args.gpu]
    if args.q:
        QUANTS = args.q.split()

def validate_requirements():
    if not os.path.exists(IN_DIR):
        print(f"ERROR: Input directory '{IN_DIR}' does not exist.")
        if FORCE_PAUSE:
            input("Press Enter to continue...")
        sys.exit(1)
    if not QUANTS:
        print("ERROR: QUANTS is not defined.")
        if FORCE_PAUSE:
            input("Press Enter to continue...")
        sys.exit(1)
    if not SELECTED_GPU and not FORCE_MGPUs:
        print("ERROR: No GPU specified. Use /gpu 1 or /gpu 0,1")
        if FORCE_PAUSE:
            input("Press Enter to continue...")
        sys.exit(1)

def generate_measurement():
    if not FORCE_MEASURE and os.path.exists(os.path.join(IN_DIR, "exl2_measurement.json")):
        print("Measurement file exists, skipping measurement generation.")
        return

    print(f"Generating measurement file (using GPU {SELECTED_GPU})...")
    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
    cmd = [
        "python", "exllamav2/convert.py",
        "-i", IN_DIR,
        "-om", os.path.join(IN_DIR, "exl2_measurement.json"),
        "-o", f"{TEMP_DIR}/",
        *LEN_OPT.split()
    ]
    if not FORCE_RECOVER:
        cmd.extend(NO_RECOVER_OPT.split())
    if 'ROPE_OPT' in globals():
        cmd.extend(ROPE_OPT.split())
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Failed to generate measurement file")
        if FORCE_PAUSE:
            input("Press Enter to continue...")
        sys.exit(1)
    if FORCE_PAUSE:
        input("Press Enter to continue...")

def run_quantization_single_gpu(quant):
    global CLEANUP_FILE

    out_dir_q = f"{OUT_DIR}-q{quant}"
    print(f"Quantizing to '{out_dir_q}' with b={quant}, GPU={SELECTED_GPU}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
    running_file = f"{TEMP}/task_gpu{SELECTED_GPU}.running"
    
    with open(running_file, "w") as f:
        f.write("Running")
    CLEANUP_FILE = running_file
    atexit.register(exit_handler)
    
    cmd = [
        "python", "exllamav2/convert.py",
        "-i", IN_DIR,
        "-m", os.path.join(IN_DIR, "exl2_measurement.json"),
        "-o", f"{TEMP_DIR}/{SELECTED_GPU}/",
        "-cf", f"{out_dir_q}/",
        "-b", quant,
        "-hb", QUANTS_HEAD,
        *LEN_OPT.split()
    ]
    if not FORCE_RECOVER:
        cmd.extend(NO_RECOVER_OPT.split())
    if 'ROPE_OPT' in globals():
        cmd.extend(ROPE_OPT.split())

    try:    
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"ERROR: Quantization failed for q={quant}")
        else:
            # Remove safetensors files
            for file in glob.glob(os.path.join(out_dir_q, "model*.safetensors")):
                os.remove(file)
            for file in glob.glob(os.path.join(out_dir_q, "model.safetensors.index.json")):
                os.remove(file)
            print(f"Quantization success for b={quant}")
    except Exception as e:
        print(f"ERROR: error occurred for q={quant}: {e}")

    if FORCE_PAUSE:
        input("Press Enter to continue...")
    
    atexit.unregister(exit_handler)
    CLEANUP_FILE = None
    os.remove(running_file)

def run_quantization_multi_gpu():
    # Clean up existing running files
    for file in glob.glob(f"{TEMP}/task_gpu*.running"):
        os.remove(file)
    
    need_lf = False    
    for quant in QUANTS:
        for tt in range(36000):  # Max wait iterations
            for gpu in FORCE_MGPUs:
                running_file = f"{TEMP}/task_gpu{gpu}.running"
                if not os.path.exists(running_file):
                    # Start new process in a new console window using CREATE_NEW_CONSOLE (works for Windows only)
                    cmd = [
                        'cmd', '/k', sys.executable, __file__,
                        *sys.argv[1:],  # Preserve original arguments
                        "-gpu", gpu,
                        "-q", quant
                    ]
                    subprocess.Popen(
                        cmd,
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                    time.sleep(1)  # Brief pause to prevent race conditions
                    break
            else:
                print(f"All GPUs busy, waiting for a GPU to free up {tt}sec (36000 max)...", end="\r")
                need_lf = True
                time.sleep(1)
                continue
            break
    if need_lf:
        print()
    
    need_lf = False    
    for tt in range(36000):  # Max wait iterations
        if not glob.glob(f"{TEMP}/task_gpu*.running"):
            break
        print(f"All quantization tasks are distributed to GPUs, waiting for completion {tt}sec (36000 max)...", end="\r")
        need_lf = True
        time.sleep(1)
    if need_lf:
        print()
    
    print("All quantization tasks completed.")
    if FORCE_PAUSE:
        input("Press Enter to continue...")

def main():
    parse_arguments()
    validate_requirements()
    generate_measurement()
    
    if FORCE_MGPUs and len(FORCE_MGPUs) > 1:
        run_quantization_multi_gpu()
    else:
        print("Performing quantizations...")
        for quant in QUANTS:
            run_quantization_single_gpu(quant)

if __name__ == "__main__":
    TEMP_DIR = TEMP_DIR.format(SELECTED_GPU or "0")  # Default to GPU 0 if not specified
    main()
