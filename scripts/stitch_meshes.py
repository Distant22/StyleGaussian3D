import os
import sys
import json
import numpy as np
import open3d as o3d
import argparse

def umeyama(X, Y):
    """
    Estimates Sim3 transformation from X to Y.
    Y = s * R * X + t
    X and Y are (N, 3) arrays of points.
    Returns (s, R, t)
    """
    assert X.shape == Y.shape
    assert X.shape[1] == 3
    
    n, m = X.shape
    
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    
    X_centered = X - mean_X
    Y_centered = Y - mean_Y
    
    var_X = np.mean(np.sum(X_centered**2, axis=1))
    
    cov = (Y_centered.T @ X_centered) / n
    
    U, D, V_T = np.linalg.svd(cov)
    S = np.eye(m)
    
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S[m-1, m-1] = -1
        
    R = U @ S @ V_T
    s = 1.0 / var_X * np.trace(np.diag(D) @ S) if var_X > 0 else 1.0
    t = mean_Y - s * R @ mean_X
    
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

def stitch_meshes(mesh1_path, mesh2_path, cam1_json, cam2_json, output_path):
    print("Loading camera data...")
    cams1 = load_cameras(cam1_json)
    cams2 = load_cameras(cam2_json)
    
    # Find common images
    common_images = set(cams1.keys()).intersection(set(cams2.keys()))
    print(f"Found {len(common_images)} overlapping images between chunks.")
    
    if len(common_images) < 3:
        print("Error: Need at least 3 common images to reliably compute Sim3 transform.")
        return
        
    # Extract centers for Umeyama
    centers1 = []
    centers2 = []
    for img in sorted(list(common_images)):
        centers1.append(cams1[img]['center'])
        centers2.append(cams2[img]['center'])
        
    centers1 = np.array(centers1)
    centers2 = np.array(centers2)
    
    print("Computing Sim3 transformation (Chunk2 -> Chunk1)...")
    s, R, t = umeyama(centers2, centers1)
    print(f"  Scale: {s:.4f}")
    print(f"  Translation: {t}")
    
    print(f"Loading {mesh1_path}...")
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    
    print(f"Loading {mesh2_path}...")
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
    
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
    parser = argparse.ArgumentParser(description="Stitch two MAtCha chunk meshes together using overlapping camera poses.")
    parser.add_argument("--mesh1", required=True, help="Path to chunk 1 PLY mesh.")
    parser.add_argument("--mesh2", required=True, help="Path to chunk 2 PLY mesh.")
    parser.add_argument("--cam1", required=True, help="Path to chunk 1 mast3r_sfm/cameras.json.")
    parser.add_argument("--cam2", required=True, help="Path to chunk 2 mast3r_sfm/cameras.json.")
    parser.add_argument("--output", required=True, help="Output PLY path for the final merged mesh.")
    
    args = parser.parse_args()
    stitch_meshes(args.mesh1, args.mesh2, args.cam1, args.cam2, args.output)
