import os
import argparse
import cv2
import numpy as np
import requests
import glob

def download_file_from_url(url: str, filename: str) -> None:
    tmp_filename = f"{filename}.tmp"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tmp_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_filename, filename)

def run_skyseg(onnx_session, input_size: list, image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image = (image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = image.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: image})
    result = np.array(result).squeeze()
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    else:
        result = np.zeros_like(result)
    return (result * 255).astype("uint8")

def process_images(input_dir, output_dir):
    try:
        import onnxruntime
    except ImportError:
        print("[ERROR] onnxruntime is required. Please run: pip install onnxruntime-gpu onnx")
        print("        (Note: the 'vggt_omega' environment already has this installed!)")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(os.path.dirname(__file__), "skyseg.onnx")
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading skyseg.onnx model to {model_path}...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", model_path)
        
    print("[INFO] Loading ONNX session...")
    # Add execution providers if GPU is available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    onnx_session = onnxruntime.InferenceSession(model_path, providers=providers)
    
    # Process images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not images:
        print(f"[WARNING] No images found in {input_dir}")
        return
        
    print(f"[INFO] Found {len(images)} images to process.")
    
    for i, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        
        print(f"  [{i+1}/{len(images)}] Processing {filename}...")
        image = cv2.imread(img_path)
        if image is None:
            print(f"    Failed to read {filename}, skipping.")
            continue
            
        # run skyseg
        result_map = run_skyseg(onnx_session, [320, 320], image)
        result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]))
        
        # In VGGT's logic: result_map < 32 means KEEP (non-sky)
        # So result_map >= 32 means SKY.
        # We want to completely black out the sky.
        sky_mask = (result_map >= 32)
        
        image[sky_mask] = [0, 0, 0]
        
        cv2.imwrite(out_path, image)
        
    print(f"\n[SUCCESS] Processed {len(images)} images and saved perfectly masked versions to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images to completely mask out the sky with pitch-black pixels.")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory containing raw images")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory to save masked images")
    args = parser.parse_args()
    
    process_images(args.input_dir, args.output_dir)
