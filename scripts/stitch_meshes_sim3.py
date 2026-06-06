import os
import sys
import json
import numpy as np
import open3d as o3d
import argparse

def robust_sim3(c2w_A, c2w_B):
    """
    Computes a robust Sim3 transform from B to A using full camera poses.
    c2w_A, c2w_B are lists of 4x4 matrices.
    This prevents rotational ambiguity when camera centers form a straight line (collinear).
    """
    N = len(c2w_A)
    O_A = np.array([c[:3, 3] for c in c2w_A])
    O_B = np.array([c[:3, 3] for c in c2w_B])
    
    mean_A = O_A.mean(axis=0)
    mean_B = O_B.mean(axis=0)
    
    O_A_centered = O_A - mean_A
    O_B_centered = O_B - mean_B
    
    var_A = np.mean(np.sum(O_A_centered**2, axis=1))
    var_B = np.mean(np.sum(O_B_centered**2, axis=1))
    
    s = np.sqrt(var_A / var_B) if var_B > 0 else 1.0
    
    O_B_scaled = O_B_centered * s
    
    # Weight for the rotation axes so they have similar magnitude to the origin vectors
    w = np.sqrt(var_A)  
    
    pts_A = list(O_A_centered)
    pts_B = list(O_B_scaled)
    
    for i in range(N):
        R_A = c2w_A[i][:3, :3]
        R_B = c2w_B[i][:3, :3]
        for j in range(3):
            pts_A.append(R_A[:, j] * w)
            pts_B.append(R_B[:, j] * w)
            
    pts_A = np.array(pts_A)
    pts_B = np.array(pts_B)
    
    cov = (pts_B.T @ pts_A)
    U, D, V_T = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S[2, 2] = -1
    R = U @ S @ V_T
    
    t = mean_A - s * R @ mean_B
    
    return s, R, t

def load_cameras(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cameras = {}
    for i, filepath in enumerate(data['filepaths']):
        basename = os.path.basename(filepath)
        c2w = np.array(data['cams2world'][i])
        center = c2w[:3, 3]
        cameras[basename] = {
            'c2w': c2w,
            'center': center
        }
    return cameras

def stitch_meshes(mesh1_path, mesh2_path, cam1_json, cam2_json, output_path, downsample_ratio=1.0):
    print("Loading camera data...")
    cams1 = load_cameras(cam1_json)
    cams2 = load_cameras(cam2_json)
    
    # Find common images
    common_images = set(cams1.keys()).intersection(set(cams2.keys()))
    print(f"Found {len(common_images)} overlapping images between chunks.")
    
    if len(common_images) < 3:
        print("Error: Need at least 3 common images to reliably compute Sim3 transform.")
        return
        
    poses1 = []
    poses2 = []
    for img in sorted(list(common_images)):
        poses1.append(cams1[img]['c2w'])
        poses2.append(cams2[img]['c2w'])
        
    print("Computing Robust Sim3 transformation (Chunk2 -> Chunk1)...")
    s, R, t = robust_sim3(poses1, poses2)
    print(f"  Scale: {s:.4f}")
    print(f"  Translation: {t}")
    
    print(f"Loading {mesh1_path}...")
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    if downsample_ratio < 1.0:
        print(f"  Downsampling Chunk 1...")
        mesh1 = mesh1.simplify_quadric_decimation(int(len(mesh1.triangles) * downsample_ratio))
        
    print(f"Loading {mesh2_path}...")
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
    if downsample_ratio < 1.0:
        print(f"  Downsampling Chunk 2...")
        mesh2 = mesh2.simplify_quadric_decimation(int(len(mesh2.triangles) * downsample_ratio))
        
    print("Transforming Chunk 2 into Chunk 1 coordinate space...")
    mesh2.scale(s, center=(0,0,0))
    mesh2.rotate(R, center=(0,0,0))
    mesh2.translate(t)
    
    print("Merging meshes...")
    merged = mesh1 + mesh2
    
    print("Cleaning up merged mesh (removing duplicated vertices/triangles)...")
    merged = merged.remove_duplicated_vertices()
    merged = merged.remove_duplicated_triangles()
    
    print(f"Saving merged mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, merged)
    print("Done! Mesh stitching complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch two MAtCha chunk meshes together using Robust Sim3 (fixes collinear ambiguity).")
    parser.add_argument("--mesh1", required=True, help="Path to chunk 1 PLY mesh.")
    parser.add_argument("--mesh2", required=True, help="Path to chunk 2 PLY mesh.")
    parser.add_argument("--cam1", required=True, help="Path to chunk 1 mast3r_sfm/cameras.json.")
    parser.add_argument("--cam2", required=True, help="Path to chunk 2 mast3r_sfm/cameras.json.")
    parser.add_argument("--output", required=True, help="Output PLY path for the final merged mesh.")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Optional decimation to prevent Out-Of-Memory errors.")
    
    args = parser.parse_args()
    stitch_meshes(args.mesh1, args.mesh2, args.cam1, args.cam2, args.output, args.downsample_ratio)
