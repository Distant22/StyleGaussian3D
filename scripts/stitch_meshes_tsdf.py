import os
import sys
import json
import numpy as np
import open3d as o3d
import argparse
from PIL import Image

def umeyama(X, Y):
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
        focal = data['focals'][i]
        center = c2w[:3, 3]
        cameras[basename] = {
            'c2w': c2w,
            'focal': focal,
            'center': center
        }
    return cameras

def stitch_meshes_tsdf(rgb1_dir, depth1_dir, cam1_json, rgb2_dir, depth2_dir, cam2_json, output_path, voxel_size=0.01):
    """
    Fuses two chunks into a single mesh using Open3D ScalableTSDFVolume.
    Expects directories containing RGB and Depth images (rendered from MAtCha or from MASt3R).
    """
    print("Loading camera data...")
    cams1 = load_cameras(cam1_json)
    cams2 = load_cameras(cam2_json)
    
    common_images = set(cams1.keys()).intersection(set(cams2.keys()))
    if len(common_images) < 3:
        print("Error: Need at least 3 common images to reliably compute Sim3 transform.")
        return
        
    centers1 = []
    centers2 = []
    for img in sorted(list(common_images)):
        centers1.append(cams1[img]['center'])
        centers2.append(cams2[img]['center'])
        
    centers1 = np.array(centers1)
    centers2 = np.array(centers2)
    
    s, R, t = umeyama(centers2, centers1)
    
    # 4x4 transform
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    
    print("Initializing TSDF Volume...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 5.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Process Chunk 1
    print("Integrating Chunk 1...")
    for basename, cam in cams1.items():
        rgb_path = os.path.join(rgb1_dir, basename)
        depth_path = os.path.join(depth1_dir, basename.replace('.jpg', '.png'))
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue
            
        color = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        
        # Pinhole camera intrinsic
        width, height = color.get_max_bound() # approx
        fx = fy = cam['focal']
        cx = width / 2.0
        cy = height / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)
        
        extrinsic = np.linalg.inv(cam['c2w']) # world to camera
        volume.integrate(rgbd, intrinsic, extrinsic)
        
    # Process Chunk 2
    print("Integrating Chunk 2 (applying Umeyama transform)...")
    for basename, cam in cams2.items():
        rgb_path = os.path.join(rgb2_dir, basename)
        depth_path = os.path.join(depth2_dir, basename.replace('.jpg', '.png'))
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue
            
        color = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        
        width, height = color.get_max_bound() # approx
        fx = fy = cam['focal']
        cx = width / 2.0
        cy = height / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)
        
        # Transform Chunk 2's camera to Chunk 1's space
        # Original: C_world = C_cam2world
        # Transformed: C_world' = T * C_cam2world
        transformed_c2w = T @ cam['c2w']
        
        # Open3D needs World-to-Camera (extrinsic)
        extrinsic = np.linalg.inv(transformed_c2w)
        
        volume.integrate(rgbd, intrinsic, extrinsic)
        
    print("Extracting TSDF mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    print(f"Saving merged TSDF mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print("Done! TSDF mesh stitching complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch two MAtCha chunks using TSDF fusion on rendered depth maps.")
    parser.add_argument("--rgb1", required=True, help="Path to chunk 1 RGB images dir.")
    parser.add_argument("--depth1", required=True, help="Path to chunk 1 Rendered Depth maps dir.")
    parser.add_argument("--cam1", required=True, help="Path to chunk 1 mast3r_sfm/cameras.json.")
    
    parser.add_argument("--rgb2", required=True, help="Path to chunk 2 RGB images dir.")
    parser.add_argument("--depth2", required=True, help="Path to chunk 2 Rendered Depth maps dir.")
    parser.add_argument("--cam2", required=True, help="Path to chunk 2 mast3r_sfm/cameras.json.")
    
    parser.add_argument("--output", required=True, help="Output PLY path for the final merged mesh.")
    parser.add_argument("--voxel", type=float, default=0.01, help="Voxel size for TSDF volume.")
    
    args = parser.parse_args()
    stitch_meshes_tsdf(args.rgb1, args.depth1, args.cam1, args.rgb2, args.depth2, args.cam2, args.output, args.voxel)
