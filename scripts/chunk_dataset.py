import os
import shutil
import argparse

def chunk_dataset(source_dir, output_dir, chunk_size, overlap):
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images and sort them alphabetically
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = [f for f in os.listdir(source_dir) if f.endswith(valid_extensions)]
    images.sort()
    
    if not images:
        print(f"No images found in {source_dir}.")
        return
        
    print(f"Found {len(images)} images.")
    
    stride = chunk_size - overlap
    if stride <= 0:
        print("Error: Overlap must be strictly less than chunk_size.")
        return
        
    chunk_idx = 0
    for i in range(0, len(images), stride):
        chunk_images = images[i:i + chunk_size]
        
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx:02d}", "images")
        os.makedirs(chunk_dir, exist_ok=True)
        
        for img in chunk_images:
            src_path = os.path.join(source_dir, img)
            dst_path = os.path.join(chunk_dir, img)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                
        print(f"Created {chunk_dir} with {len(chunk_images)} images.")
        print(f"  Range: {chunk_images[0]} -> {chunk_images[-1]}")
        
        chunk_idx += 1
        
        # If this chunk already reaches or exceeds the end of the list, stop.
        if i + chunk_size >= len(images):
            break
            
    print(f"\nSuccessfully created {chunk_idx} chunks in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split image dataset into overlapping chunks for staged processing.")
    parser.add_argument("-s", "--source", required=True, help="Source directory containing images.")
    parser.add_argument("-o", "--output", required=True, help="Output directory to place chunks.")
    parser.add_argument("-c", "--chunk_size", type=int, default=25, help="Number of images per chunk.")
    parser.add_argument("-v", "--overlap", type=int, default=5, help="Number of overlapping images between adjacent chunks.")
    
    args = parser.parse_args()
    chunk_dataset(args.source, args.output, args.chunk_size, args.overlap)
