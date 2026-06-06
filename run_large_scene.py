import os
import sys
import glob
import argparse
import open3d as o3d
import subprocess

def run_large_scene():
    parser = argparse.ArgumentParser(description="Run MAtCha on a large sequence using DA3 global alignment.")
    parser.add_argument('-s', '--source_path', required=True, type=str, help='Input image directory')
    parser.add_argument('-o', '--output_path', required=True, type=str, help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=25, help='Number of images per chunk')
    parser.add_argument('--overlap', type=int, default=5, help='Overlap between chunks')
    parser.add_argument('--da3_dir', type=str, default=None, help='Pre-computed DA3 directory')
    parser.add_argument('--conf_coef', type=float, default=0.0, help='Confidence threshold to filter DA3 depths (e.g. 0.5)')
    parser.add_argument('--use_mast3r', action='store_true', help='Skip DA3 global alignment and use MASt3R SfM independently per chunk. Warning: merged mesh will not be aligned.')
    parser.add_argument('--use_vggt', action='store_true', help='Use VGGT-Omega for global alignment instead of DA3.')
    
    # Parse known args, so the rest can be forwarded to train.py
    args, unknown = parser.parse_known_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1. Run DA3 or VGGT on the whole sequence if not provided
    da3_dir = args.da3_dir
    if da3_dir is None and not args.use_mast3r:
        if args.use_vggt:
            print("[INFO] Running VGGT-Omega on the entire sequence...")
            da3_dir = os.path.abspath(os.path.join(args.output_path, 'vggt_output'))
            vggt_run_cmd = [
                "conda", "run", "-n", "vggt_omega",
                "python", "scripts/vggt_to_matcha.py",
                "-s", os.path.abspath(args.source_path),
                "-o", da3_dir
            ]
            print(f"[INFO] Command: {' '.join(vggt_run_cmd)}")
            status = subprocess.run(vggt_run_cmd)
            if status.returncode != 0:
                print("[ERROR] VGGT execution failed!")
                sys.exit(1)
        else:
            print("[INFO] Running DA3 on the entire sequence...")
            da3_dir = os.path.abspath(os.path.join(args.output_path, 'da3_output'))
            da3_run_cmd = [
                "conda", "run", "-n", "depth_anything_v3",
                "python", "da3_streaming.py",
                "--image_dir", os.path.abspath(args.source_path),
                "--config", "./configs/base_config.yaml",
                "--output_dir", da3_dir
            ]
            print(f"[INFO] Command: {' '.join(da3_run_cmd)}")
            status = subprocess.run(da3_run_cmd, cwd="Depth-Anything-3/da3_streaming")
            if status.returncode != 0:
                print("[ERROR] DA3 execution failed!")
                sys.exit(1)
            
    # 2. Get total number of images
    img_list = sorted(
        glob.glob(os.path.join(args.source_path, "*.jpg")) + 
        glob.glob(os.path.join(args.source_path, "*.png")) + 
        glob.glob(os.path.join(args.source_path, "*.JPG"))
    )
    total_imgs = len(img_list)
    print(f"[INFO] Found {total_imgs} images.")
    
    if total_imgs == 0:
        print("[ERROR] No images found!")
        sys.exit(1)
    
    # 3. Chunking
    chunks = []
    start = 0
    while start < total_imgs:
        end = min(start + args.chunk_size, total_imgs)
        chunks.append(list(range(start, end)))
        if end == total_imgs:
            break
        start = start + args.chunk_size - args.overlap
        
    print(f"[INFO] Created {len(chunks)} chunks.")
    
    # 4. Train each chunk
    for i, chunk_idx in enumerate(chunks):
        print(f"\n====================================")
        print(f"[INFO] Processing Chunk {i+1}/{len(chunks)}: frames {chunk_idx[0]} to {chunk_idx[-1]}")
        chunk_out = os.path.join(args.output_path, f"chunk_{i:03d}")
        
        train_cmd = [
            sys.executable, "train.py",
            "-s", args.source_path,
            "-o", chunk_out,
            "--image_idx"
        ] + [str(idx) for idx in chunk_idx] + unknown
        
        if not args.use_mast3r:
            train_cmd = [
                sys.executable, "train.py",
                "-s", args.source_path,
                "-o", chunk_out,
                "--da3_dir", da3_dir,
                "--image_idx"
            ] + [str(idx) for idx in chunk_idx] + unknown
        
        print(f"[INFO] Command: {' '.join(train_cmd)}")
        status = subprocess.run(train_cmd)
        if status.returncode != 0:
            print(f"[ERROR] Chunk {i+1} failed!")
            sys.exit(1)
            
    # 5. Merge Meshes
    print("\n[INFO] Merging Tetra meshes...")
    merged_mesh = o3d.geometry.TriangleMesh()
    for i in range(len(chunks)):
        chunk_mesh_path = os.path.join(args.output_path, f"chunk_{i:03d}", "tetra_meshes", "tetra_mesh_binary_search_7.ply")
            
        if os.path.exists(chunk_mesh_path):
            print(f"Loading {chunk_mesh_path}...")
            m = o3d.io.read_triangle_mesh(chunk_mesh_path)
            merged_mesh += m
        else:
            print(f"[WARNING] {chunk_mesh_path} not found.")
            
    if len(merged_mesh.vertices) > 0:
        print("[INFO] Cleaning up merged mesh (removing duplicated vertices/triangles)...")
        merged_mesh = merged_mesh.remove_duplicated_vertices()
        merged_mesh = merged_mesh.remove_duplicated_triangles()
        
        final_mesh_path = os.path.join(args.output_path, "merged_tetra_mesh.ply")
        o3d.io.write_triangle_mesh(final_mesh_path, merged_mesh)
        print(f"[INFO] Saved final merged mesh to {final_mesh_path}")
    else:
        print("[WARNING] No meshes were successfully loaded to merge.")
        
    print("[INFO] Done!")

if __name__ == '__main__':
    run_large_scene()
