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
    
    # We want to find R such that R @ O_B_scaled \approx O_A_centered
    # AND R @ v_B \approx v_A for the rotation columns.
    
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

def get_transform_matrix(s, R, t):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T

def stitch_n_meshes(mesh_paths, cam_paths, output_path, downsample_ratio=1.0):
    assert len(mesh_paths) == len(cam_paths), "Number of meshes must match number of camera files!"
    N = len(mesh_paths)
    
    if N == 0:
        print("No meshes provided.")
        return
        
    print(f"[INFO] Loading Chunk 0 mesh: {mesh_paths[0]}")
    merged = o3d.io.read_triangle_mesh(mesh_paths[0])
    
    if downsample_ratio < 1.0:
        target_triangles = int(len(merged.triangles) * downsample_ratio)
        print(f"  Downsampling Chunk 0 to {target_triangles} triangles...")
        merged = merged.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    
    # Keep track of global transformation for each chunk to put it into chunk 0's coordinate space
    T_global = [np.eye(4)] * N
    
    for i in range(1, N):
        print(f"\n[INFO] --- Aligning Chunk {i} to Chunk {i-1} ---")
        cams_prev = load_cameras(cam_paths[i-1])
        cams_curr = load_cameras(cam_paths[i])
        
        common_images = set(cams_prev.keys()).intersection(set(cams_curr.keys()))
        print(f"  Found {len(common_images)} overlapping images.")
        
        if len(common_images) < 3:
            print(f"  [ERROR] Need at least 3 common images between chunk {i} and {i-1}. Cannot stitch!")
            sys.exit(1)
            
        poses_prev = []
        poses_curr = []
        for img in sorted(list(common_images)):
            poses_prev.append(cams_prev[img]['c2w'])
            poses_curr.append(cams_curr[img]['c2w'])
            
        # Compute Sim3 from original i to original i-1 using robust method
        s, R, t = robust_sim3(poses_prev, poses_curr)
        T_local = get_transform_matrix(s, R, t)
        
        # Compute global transform: T_global[i] = T_global[i-1] * T_local
        T_global[i] = T_global[i-1] @ T_local
        
        # Load mesh i
        print(f"  Loading mesh {mesh_paths[i]}...")
        mesh_i = o3d.io.read_triangle_mesh(mesh_paths[i])
        
        if downsample_ratio < 1.0:
            target_triangles = int(len(mesh_i.triangles) * downsample_ratio)
            print(f"  Downsampling Chunk {i} to {target_triangles} triangles...")
            mesh_i = mesh_i.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        
        # Transform mesh i to global frame
        # Open3D transform uses a 4x4 matrix
        print(f"  Applying global transformation...")
        mesh_i.transform(T_global[i])
        
        print(f"  Merging with global mesh...")
        merged += mesh_i
        
    print("\n[INFO] Cleaning up merged mesh (removing duplicated vertices/triangles)...")
    merged = merged.remove_duplicated_vertices()
    merged = merged.remove_duplicated_triangles()
    
    print(f"[INFO] Saving merged mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, merged)
    print("[INFO] Done! Mesh stitching complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch N MAtCha chunk meshes together sequentially using overlapping camera poses.")
    parser.add_argument("--input_dir", required=True, help="Path to the main output directory containing all the chunk_XXX folders.")
    parser.add_argument("--output", required=True, help="Output PLY path for the final merged mesh.")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Downsample ratio for each chunk (e.g. 0.1 to keep 10% of triangles). Helps prevent out-of-memory errors.")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)
        
    # Find all chunk directories
    chunk_dirs = [d for d in os.listdir(input_dir) if d.startswith("chunk_") and os.path.isdir(os.path.join(input_dir, d))]
    chunk_dirs.sort()
    
    if len(chunk_dirs) == 0:
        print(f"[ERROR] No chunk directories found in {input_dir}")
        sys.exit(1)
        
    meshes = []
    cams = []
    
    for chunk_name in chunk_dirs:
        chunk_path = os.path.join(input_dir, chunk_name)
        mesh_path = os.path.join(chunk_path, "tetra_meshes", "tetra_mesh_binary_search_7.ply")
        cam_path = os.path.join(chunk_path, "mast3r_sfm", "cameras.json")
        
        if os.path.exists(mesh_path) and os.path.exists(cam_path):
            meshes.append(mesh_path)
            cams.append(cam_path)
        else:
            print(f"[WARNING] Skipping {chunk_name} because mesh or camera json is missing.")
            
    if len(meshes) < 2:
        print("[ERROR] Found less than 2 valid chunks to stitch together!")
        sys.exit(1)
        
    print(f"[INFO] Found {len(meshes)} valid chunks to stitch.")
    stitch_n_meshes(meshes, cams, args.output, args.downsample_ratio)
