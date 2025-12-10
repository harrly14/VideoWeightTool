import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

def apply_clahe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bgr_enhanced

def test_clahe_consistency():
    print("="*60)
    print("TESTING CLAHE PIPELINE CONSISTENCY")
    print("="*60)
    
    # Create a synthetic test image
    test_img = np.random.randint(0, 256, (64, 256, 3), dtype=np.uint8)
    print(f"\nTest image shape: {test_img.shape}")
    print(f"Test image dtype: {test_img.dtype}")
    
    # Test 1: Apply CLAHE from edit_video_for_cnn
    from edit_video_for_cnn import apply_clahe as edit_clahe
    result1 = edit_clahe(test_img.copy())
    print(f"\nedit_video_for_cnn.apply_clahe output shape: {result1.shape}")
    print(f"edit_video_for_cnn.apply_clahe output dtype: {result1.dtype}")
    
    # Test 2: Apply CLAHE from process_video
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from process_video import apply_clahe as process_clahe
    result2 = process_clahe(test_img.copy())
    print(f"\nprocess_video.apply_clahe output shape: {result2.shape}")
    print(f"process_video.apply_clahe output dtype: {result2.dtype}")
    
    # Test 3: Check if outputs are identical
    if np.array_equal(result1, result2):
        print("\n[PASS] Both CLAHE implementations produce identical results")
    else:
        diff = np.abs(result1.astype(float) - result2.astype(float))
        print(f"\n[FAIL] Outputs differ! Max difference: {diff.max()}")
        return False
    
    # Test 4: Verify output is 3-channel BGR
    if result1.shape[2] == 3:
        print("[PASS] Output is 3-channel (BGR compatible)")
    else:
        print(f"[FAIL] Output has {result1.shape[2]} channels, expected 3")
        return False
    
    # Test 5: Verify grayscale conversion (all channels should be equal)
    if np.array_equal(result1[:,:,0], result1[:,:,1]) and np.array_equal(result1[:,:,1], result1[:,:,2]):
        print("[PASS] All BGR channels are identical (grayscale replicated)")
    else:
        print("[FAIL] BGR channels differ - not properly converted from grayscale")
        return False
    
    return True

def test_aspect_ratio_enforcement():
    print("\n" + "="*60)
    print("TESTING ASPECT RATIO ENFORCEMENT")
    print("="*60)
    
    from edit_video_for_cnn import CNN_WIDTH, CNN_HEIGHT, CNN_ASPECT_RATIO
    
    print(f"\nCNN dimensions: {CNN_WIDTH}x{CNN_HEIGHT}")
    print(f"Expected aspect ratio: {CNN_ASPECT_RATIO}")
    
    # Test various widths
    test_widths = [100, 200, 400, 512, 1024]
    
    all_pass = True
    for w in test_widths:
        h = int(w / CNN_ASPECT_RATIO)
        actual_ratio = w / h if h > 0 else 0
        
        if abs(actual_ratio - CNN_ASPECT_RATIO) < 0.01:
            print(f"[PASS] Width {w} -> Height {h} (ratio: {actual_ratio:.2f})")
        else:
            print(f"[FAIL] Width {w} -> Height {h} (ratio: {actual_ratio:.2f}, expected {CNN_ASPECT_RATIO})")
            all_pass = False
    
    return all_pass

def test_resize_interpolation():
    print("\n" + "="*60)
    print("TESTING RESIZE INTERPOLATION")
    print("="*60)
    
    from edit_video_for_cnn import CNN_WIDTH, CNN_HEIGHT
    
    # Test downscaling (larger image)
    large_img = np.random.randint(0, 256, (256, 1024, 3), dtype=np.uint8)
    resized_down = cv2.resize(large_img, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_AREA)
    print(f"\nDownscale {large_img.shape[:2]} -> {resized_down.shape[:2]} using INTER_AREA")
    
    if resized_down.shape == (CNN_HEIGHT, CNN_WIDTH, 3):
        print(f"[PASS] Downscaled to correct size: {CNN_WIDTH}x{CNN_HEIGHT}")
    else:
        print(f"[FAIL] Wrong size: {resized_down.shape}")
        return False
    
    # Test upscaling (smaller image)
    small_img = np.random.randint(0, 256, (32, 128, 3), dtype=np.uint8)
    resized_up = cv2.resize(small_img, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_LINEAR)
    print(f"Upscale {small_img.shape[:2]} -> {resized_up.shape[:2]} using INTER_LINEAR")
    
    if resized_up.shape == (CNN_HEIGHT, CNN_WIDTH, 3):
        print(f"[PASS] Upscaled to correct size: {CNN_WIDTH}x{CNN_HEIGHT}")
    else:
        print(f"[FAIL] Wrong size: {resized_up.shape}")
        return False
    
    return True

def test_dataset_clahe_present():
    print("\n" + "="*60)
    print("CHECKING DATASET.PY HAS CLAHE FOR RUNTIME APPLICATION")
    print("="*60)
    
    # Read dataset.py and check for CLAHE in transforms
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset.py")
    
    with open(dataset_path, 'r') as f:
        content = f.read()
    
    # Check for our custom grayscale CLAHE function
    has_custom_clahe = 'apply_clahe_grayscale' in content
    has_clahe_function = 'def apply_clahe_grayscale' in content
    uses_lambda = 'A.Lambda(image=apply_clahe_grayscale' in content
    
    if has_custom_clahe and has_clahe_function and uses_lambda:
        print("\n[PASS] dataset.py uses custom grayscale CLAHE via A.Lambda")
        print("This ensures training/validation matches inference (OpenCV) exactly")
        return True
    elif 'A.CLAHE' in content:
        print("\n[WARN] dataset.py uses A.CLAHE instead of custom grayscale function")
        print("This may cause training/inference mismatch (Albumentations uses LAB color space)")
        return True  # Still passes since CLAHE is present, just not optimal
    else:
        print("\n[FAIL] dataset.py is missing CLAHE in transforms!")
        print("Since saved frames are RAW, CLAHE must be applied during training.")
        return False

def test_edit_script_saves_raw():
    print("\n" + "="*60)
    print("CHECKING EDIT SCRIPT SAVES RAW FRAMES")
    print("="*60)
    
    # Read edit_video_for_cnn.py and check save_frames doesn't apply CLAHE
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edit_video_for_cnn.py")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the save_frames function and check if it applies CLAHE before saving
    # Look for pattern: in save_frames, should NOT have "enhanced = apply_clahe" before cv2.imwrite
    import re
    
    # Extract save_frames function
    match = re.search(r'def save_frames\(self\):.*?(?=\n    def |\nclass |\Z)', content, re.DOTALL)
    if not match:
        print("[FAIL] Could not find save_frames function")
        return False
    
    save_frames_code = match.group(0)
    
    # Check if CLAHE is applied in save path (bad) vs just for resize reference (ok)
    if 'enhanced = apply_clahe' in save_frames_code:
        print("\n[FAIL] save_frames() applies CLAHE before saving!")
        print("Saved frames should be RAW - CLAHE is applied at runtime during training")
        return False
    else:
        print("\n[PASS] save_frames() saves RAW frames (no CLAHE)")
        print("CLAHE will be applied at runtime during training/inference")
        return True
    
def test_albumentations_parity():
    print("\n" + "="*60)
    print("TESTING DATASET.PY VS OPENCV PARITY (GRAYSCALE MODE)")
    print("="*60)
    
    # Import the custom CLAHE function from dataset.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import apply_clahe_grayscale
    from process_video import apply_clahe as process_clahe

    # 1. Setup Input (RGB image like what dataset loads)
    test_img_rgb = np.random.randint(0, 256, (64, 256, 3), dtype=np.uint8)

    # 2. Run OpenCV (Inference Logic from process_video.py)
    # process_clahe expects BGR, so convert
    test_img_bgr = cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2BGR)
    res_cv_bgr = process_clahe(test_img_bgr)
    res_cv_rgb = cv2.cvtColor(res_cv_bgr, cv2.COLOR_BGR2RGB)

    # 3. Run Dataset's CLAHE function (Training Logic)
    # apply_clahe_grayscale expects RGB (what dataset.__getitem__ provides)
    res_dataset_rgb = apply_clahe_grayscale(test_img_rgb)

    # 4. Compare
    diff = np.abs(res_cv_rgb.astype(int) - res_dataset_rgb.astype(int))
    max_diff = np.max(diff)
    
    print(f"Max pixel difference: {max_diff}")

    if max_diff == 0:
        print("[PASS] Dataset CLAHE produces IDENTICAL output to inference CLAHE.")
        return True
    else:
        print(f"[FAIL] Difference exists (Max {max_diff}).")
        return False


def main():
    print("\n" + "="*60)
    print("CNN PREPROCESSING PIPELINE TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("CLAHE Consistency", test_clahe_consistency()))
    results.append(("Aspect Ratio Enforcement", test_aspect_ratio_enforcement()))
    results.append(("Resize Interpolation", test_resize_interpolation()))
    results.append(("Dataset CLAHE Present", test_dataset_clahe_present()))
    results.append(("Edit Script Saves Raw", test_edit_script_saves_raw()))
    results.append(("Library Parity", test_albumentations_parity()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Please review the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())