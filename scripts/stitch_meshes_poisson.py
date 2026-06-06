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

def run_poisson_safe(pcd, depth):
    """
    Safely runs Poisson reconstruction. If it encounters the 'Failed to close loop' error
    (common with overlapping conflicting normals), it applies a tiny jitter and retries.
    """
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        return mesh, densities
    except Exception as e:
        print(f"Poisson failed at depth {depth}. Applying slight jitter and recomputing normals to fix topological loops...")
        pts = np.asarray(pcd.points)
        # Add 1mm jitter
        jitter = np.random.normal(0, 0.001, pts.shape)
        pcd.points = o3d.utility.Vector3dVector(pts + jitter)
        # Re-estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Retry
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        return mesh, densities

def stitch_meshes_poisson(mesh1_path, mesh2_path, cam1_json, cam2_json, output_path, num_points=1000000, depth=10):
    print("Loading camera data...")
    cams1 = load_cameras(cam1_json)
    cams2 = load_cameras(cam2_json)
    
    common_images = set(cams1.keys()).intersection(set(cams2.keys()))
    print(f"Found {len(common_images)} overlapping images between chunks.")
    
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
    
    print("Computing Sim3 transformation (Chunk2 -> Chunk1)...")
    s, R, t = umeyama(centers2, centers1)
    print(f"  Scale: {s:.4f}")
    
    print(f"Loading {mesh1_path}...")
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    mesh1.compute_vertex_normals()
    
    print(f"Loading {mesh2_path}...")
    mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
    mesh2.compute_vertex_normals()
    
    print("Transforming Chunk 2 into Chunk 1 coordinate space...")
    mesh2.scale(s, center=(0,0,0))
    mesh2.rotate(R, center=(0,0,0))
    mesh2.translate(t)
    
    print(f"Sampling {num_points} points from each mesh...")
    pcd1 = mesh1.sample_points_uniformly(number_of_points=num_points)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=num_points)
    
    # Store original colors if they exist
    has_colors = mesh1.has_vertex_colors() and mesh2.has_vertex_colors()
    
    print("Combining point clouds...")
    combined_pcd = pcd1 + pcd2
    
    print("Cleaning up combined point cloud (Statistical Outlier Removal)...")
    # This removes floating artifacts in the overlap region which often cause the Poisson loop error
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print("Downsampling combined point cloud to unify density...")
    bbox = combined_pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    voxel_size = extent / 2000.0  # Finer voxel size
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # After downsampling, normals can get messed up if surfaces intersect. 
    # Let's re-normalize to ensure unit length.
    normals = np.asarray(combined_pcd.normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # prevent div by zero
    combined_pcd.normals = o3d.utility.Vector3dVector(normals / norms)
    
    print(f"Running Poisson surface reconstruction (depth={depth}). This may take a while...")
    merged_mesh, densities = run_poisson_safe(combined_pcd, depth)
    
    print("Cleaning up Poisson 'bubble' artifacts based on density...")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    merged_mesh.remove_vertices_by_mask(vertices_to_remove)
    
    if has_colors:
        print("Transferring colors from original point cloud to new mesh using KDTree...")
        mesh_vertices = np.asarray(merged_mesh.vertices)
        pcd_points = np.asarray(combined_pcd.points)
        pcd_colors = np.asarray(combined_pcd.colors)
        
        try:
            from scipy.spatial import cKDTree
            print("Using scipy cKDTree for fast vectorized nearest neighbor search...")
            tree = cKDTree(pcd_points)
            _, indices = tree.query(mesh_vertices, k=1)
            mesh_colors = pcd_colors[indices]
            merged_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        except ImportError:
            print("Scipy not found. Falling back to Open3D KDTree (this may take a minute for large meshes)...")
            pcd_tree = o3d.geometry.KDTreeFlann(combined_pcd)
            mesh_colors = np.zeros_like(mesh_vertices)
            for i in range(len(mesh_vertices)):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
                mesh_colors[i, :] = pcd_colors[idx[0], :]
            merged_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    print(f"Saving merged Poisson mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, merged_mesh)
    print("Done! Poisson mesh stitching complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch two MAtCha chunk meshes together using Poisson Surface Reconstruction.")
    parser.add_argument("--mesh1", required=True, help="Path to chunk 1 PLY mesh.")
    parser.add_argument("--mesh2", required=True, help="Path to chunk 2 PLY mesh.")
    parser.add_argument("--cam1", required=True, help="Path to chunk 1 mast3r_sfm/cameras.json.")
    parser.add_argument("--cam2", required=True, help="Path to chunk 2 mast3r_sfm/cameras.json.")
    parser.add_argument("--output", required=True, help="Output PLY path for the final merged mesh.")
    parser.add_argument("--depth", type=int, default=10, help="Poisson tree depth (higher = more detail).")
    parser.add_argument("--points", type=int, default=1000000, help="Number of points to sample per mesh.")
    
    args = parser.parse_args()
    stitch_meshes_poisson(args.mesh1, args.mesh2, args.cam1, args.cam2, args.output, args.points, args.depth)
