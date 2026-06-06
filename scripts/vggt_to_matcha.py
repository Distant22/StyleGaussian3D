import os
import glob
import json
import argparse
import shutil
import torch
import numpy as np
from tqdm import tqdm

import sys
# Make sure VGGT is importable if this is run in the vggt_omega environment
try:
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
except ImportError:
    print("Error: Could not import vggt_omega. Make sure you are in the vggt_omega conda environment.")
    sys.exit(1)

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
    parser = argparse.ArgumentParser(description="Run VGGT-Omega and format output for MAtCha")
    parser.add_argument('-s', '--source_path', type=str, required=True, help='Path to original images')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output directory for MAtCha (e.g. image_output/plaza_1/vggt_sfm)')
    parser.add_argument('--checkpoint', type=str, default='/workspace/Bill/MAtCha/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt', help='VGGT checkpoint')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution for VGGT')
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
    
    print(f"Found {len(img_list)} images. Copying to output directory...")
    filepaths_for_json = []
    
    # We will pass the full paths to VGGT load_and_preprocess_images
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(images_out_dir, img_name)
        if not os.path.exists(out_img_path):
            shutil.copy2(img_path, out_img_path)
        rel_img_path = os.path.relpath(out_img_path, start=os.getcwd())
        filepaths_for_json.append(rel_img_path)

    # 2. Load VGGT-Omega Model
    print(f"Loading VGGT-Omega model from {args.checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    
    print(f"Loading and preprocessing {len(img_list)} images at resolution {args.resolution}...")
    # NOTE: If this OOMs, the user will need to resize images or chunk, but VGGT claims 300+ fits in 32GB at 512.
    images_tensor = load_and_preprocess_images(img_list, image_resolution=args.resolution).to(device)
    
    print("Running VGGT-Omega inference... (This may take a moment)")
    with torch.inference_mode():
        predictions = model(images_tensor)
        
    print("Decoding poses and intrinsics...")
    extrinsics_w2c, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    ) # extrinsics_w2c: (1, N, 3, 4), intrinsics: (1, N, 3, 3)
    
    # Strip batch dimension
    extrinsics_w2c = extrinsics_w2c[0] # (N, 3, 4)
    intrinsics = intrinsics[0] # (N, 3, 3)
    depths = predictions["depth"][0] # (N, H, W)
    depth_confs = predictions["depth_conf"][0] # (N, H, W)
    
    # 3. Format into MAtCha expectations (Mimicking DA3 raw output format)
    poses_lines = []
    intrinsics_lines = []
    
    results_out_dir = os.path.join(args.output_path, 'results_output')
    os.makedirs(results_out_dir, exist_ok=True)
    
    print("Processing and formatting charts data...")
    for i in tqdm(range(len(img_list))):
        # Extract intrinsic
        intr = intrinsics[i] # (3, 3)
        fx = intr[0, 0].item()
        fy = intr[1, 1].item()
        cx = intr[0, 2].item()
        cy = intr[1, 2].item()
        
        # Extract extrinsic (W2C to C2W)
        w2c_3x4 = extrinsics_w2c[i] # (3, 4)
        w2c_4x4 = torch.eye(4, device=device)
        w2c_4x4[:3, :] = w2c_3x4
        c2w_4x4 = torch.inverse(w2c_4x4) # (4, 4)
        
        poses_lines.append(" ".join([str(x) for x in c2w_4x4.cpu().numpy().flatten()]))
        intrinsics_lines.append(f"{fx} {fy} {cx} {cy}")
        
        # Depth and points
        d = depths[i].squeeze().cpu().numpy() # (H, W)
        c = depth_confs[i].squeeze().cpu().numpy() # (H, W)
        
        # Save individual frame NPZ
        np.savez(
            os.path.join(results_out_dir, f'frame_{i}.npz'),
            depth=d,
            conf=c,
            intrinsics=intr.cpu().numpy()
        )
        
    print("Saving camera_poses.txt and intrinsic.txt...")
    with open(os.path.join(args.output_path, 'camera_poses.txt'), 'w') as f:
        f.write("\n".join(poses_lines))
        
    with open(os.path.join(args.output_path, 'intrinsic.txt'), 'w') as f:
        f.write("\n".join(intrinsics_lines))

    print(f"Done! VGGT output formatted as pseudo-DA3 at: {args.output_path}")
