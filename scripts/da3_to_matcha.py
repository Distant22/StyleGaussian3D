import os
import glob
import json
import argparse
import shutil
import torch
import numpy as np
from tqdm import tqdm

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q

def depth_to_points_world(depth, intrinsics, extrinsics_c2w):
    H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
    ones = torch.ones_like(x)
    pix_coords = torch.stack([x, y, ones], dim=-1).float() # (H, W, 3)
    intrinsics_inv = torch.inverse(intrinsics) # (3, 3)
    cam_coords = (pix_coords @ intrinsics_inv.T) * depth.unsqueeze(-1) # (H, W, 3)
    cam_coords_h = torch.cat([cam_coords, ones.unsqueeze(-1)], dim=-1) # (H, W, 4)
    world_coords = cam_coords_h @ extrinsics_c2w.T # (H, W, 4)
    return world_coords[..., :3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', type=str, required=True, help='Path to original images')
    parser.add_argument('--da3_dir', type=str, required=True, help='Path to DA3 outputs (exps)')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output directory for MAtCha (e.g. image_output/plaza_1/da3_sfm)')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, help='View indices to use')
    parser.add_argument('--conf_coef', type=float, default=0.0, help='Confidence threshold coefficient (e.g. 0.5) to mask out noisy depths. 0.0 means no masking.')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    images_out_dir = os.path.join(args.output_path, 'images')
    os.makedirs(images_out_dir, exist_ok=True)

    # 1. Read images in source_path
    img_list = sorted(
        glob.glob(os.path.join(args.source_path, "*.jpg")) + 
        glob.glob(os.path.join(args.source_path, "*.png")) + 
        glob.glob(os.path.join(args.source_path, "*.JPG"))
    )
    
    # 2. Read DA3 poses and intrinsics
    poses_path = os.path.join(args.da3_dir, 'camera_poses.txt')
    intrinsics_path = os.path.join(args.da3_dir, 'intrinsic.txt')
    
    with open(poses_path, 'r') as f:
        poses_lines = f.readlines()
    with open(intrinsics_path, 'r') as f:
        intrinsics_lines = f.readlines()
        
    cams_json = {
        "filepaths": [],
        "focals": [],
        "cams2world": []
    }
    
    charts_depths = []
    charts_confs = []
    charts_pts = []
    
    intrinsics_list = []
    depth_shapes = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Processing frames...")
    for i, img_path in enumerate(tqdm(img_list)):
        if i >= len(poses_lines):
            break
            
        if args.image_idx is not None and i not in args.image_idx:
            continue
            
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(images_out_dir, img_name)
        if not os.path.exists(out_img_path):
            shutil.copy2(img_path, out_img_path)
            
        rel_img_path = os.path.relpath(out_img_path, start=os.getcwd())
        
        # Parse intrinsic
        # fx, fy, cx, cy
        intr_vals = list(map(float, intrinsics_lines[i].strip().split()))
        fx, fy, cx, cy = intr_vals
        focal_avg = (fx + fy) / 2.0
        
        # Parse pose (C2W)
        pose_vals = list(map(float, poses_lines[i].strip().split()))
        c2w = np.array(pose_vals).reshape(4, 4)
        
        cams_json["filepaths"].append(rel_img_path)
        cams_json["focals"].append(focal_avg)
        cams_json["cams2world"].append(c2w.tolist())
        
        # Load frame NPZ
        npz_path = os.path.join(args.da3_dir, 'results_output', f'frame_{i}.npz')
        frame_data = np.load(npz_path)
        depth = frame_data['depth']
        conf = frame_data['conf']
        intrinsics = frame_data['intrinsics'] # 3x3
        
        intrinsics_list.append((fx, fy, cx, cy))
        depth_shapes.append(depth.shape)
        
        depth_t = torch.from_numpy(depth).to(device)
        conf_t = torch.from_numpy(conf).to(device)
        intrinsics_t = torch.from_numpy(intrinsics).to(device)
        c2w_t = torch.from_numpy(c2w).float().to(device)
        
        # Compute pts in world coordinates
        pts_t = depth_to_points_world(depth_t, intrinsics_t, c2w_t)
        
        charts_depths.append(depth_t.cpu().numpy())
        charts_confs.append(conf_t.cpu().numpy())
        charts_pts.append(pts_t.cpu().numpy())
        
    print(f"Saving cameras.json...")
    with open(os.path.join(args.output_path, 'cameras.json'), 'w') as f:
        json.dump(cams_json, f)
        
    print(f"Saving charts_data.npz...")
    charts_depths_np = np.stack(charts_depths, axis=0)
    charts_confs_np = np.stack(charts_confs, axis=0)
    charts_pts_np = np.stack(charts_pts, axis=0)
    
    # Mask out noisy depths if requested
    if args.conf_coef > 0.0:
        percentile_val = args.conf_coef
        # If user passed 0.5 instead of 50.0, automatically fix it for them to match the 0-100 scale of percentiles
        if percentile_val <= 1.0:
            percentile_val *= 100.0
            
        print(f"Applying VGGT EXACT percentile mask (Bottom {percentile_val}% of each image)...")
        mask = np.zeros_like(charts_confs_np, dtype=bool)
        
        # VGGT computes the percentile individually per image
        for i in range(len(charts_confs_np)):
            conf_threshold = np.percentile(charts_confs_np[i], percentile_val)
            mask[i] = charts_confs_np[i] < conf_threshold
        
        # User's Sky-Carving Hypothesis: 
        # Instead of setting depth to 0 (which lets the model hallucinate floaters near the camera),
        # explicitly set it to 1000 with high confidence to aggressively carve out empty space!
        
        # CRITICAL FIX: We must also push the 3D points (charts_pts_np) out to 1000.0 depth!
        # Otherwise the 2D depth map will say "empty sky" while the 3D points say "solid wall", crashing the Gaussians.
        for i in range(len(charts_depths_np)):
            img_mask = mask[i]
            if not np.any(img_mask):
                continue
            
            # Camera center
            C = np.array(cams_json["cams2world"][i])[:3, 3]
            
            old_depths = charts_depths_np[i][img_mask]
            old_depths[old_depths == 0] = 1e-5 # prevent div by zero
            
            # Ray direction = (Point - Center) / Depth
            rays = (charts_pts_np[i][img_mask] - C) / old_depths[:, None]
            
            # New point = Center + Ray * 1000.0
            charts_pts_np[i][img_mask] = C + rays * 1000.0

        charts_depths_np[mask] = 1000.0  
        charts_confs_np[mask] = 1.0
        print(f"Masked out {np.sum(mask)} / {mask.size} pixels (Set to INF depth for sky carving)")
    
    np.savez(
        os.path.join(args.output_path, 'charts_data.npz'),
        prior_depths=np.ascontiguousarray(charts_depths_np),
        depths=np.ascontiguousarray(charts_depths_np),
        pts=np.ascontiguousarray(charts_pts_np),
        confs=np.ascontiguousarray(charts_confs_np),
        scale_factor=np.array(1.0, dtype=np.float32)
    )
    
    # Save Colmap format to satisfy 2DGS Scene loader
    print("Saving Colmap sparse/0 directory...")
    sparse_dir = os.path.join(args.output_path, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f_cam, \
         open(os.path.join(sparse_dir, "images.txt"), "w") as f_img:
        
        for i in range(len(cams_json["filepaths"])):
            cam_id = i + 1
            fx, fy, cx, cy = intrinsics_list[i]
            H, W = depth_shapes[i]
            f_cam.write(f"{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
            
            c2w = np.array(cams_json["cams2world"][i])
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat2qvec(R)
            name = os.path.basename(cams_json["filepaths"][i])
            
            f_img.write(f"{cam_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n\n")

    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f_pts:
        f_pts.write("1 0 0 0 255 0 0 0.0\n")
        f_pts.write("2 1 0 0 0 255 0 0.0\n")
        f_pts.write("3 0 1 0 0 0 255 0.0\n")

    print("Done!")
