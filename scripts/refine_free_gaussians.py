import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

from rich.console import Console


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('--white_background', type=bool, default=False)
    
    # For dense RGB and depth supervision from a COLMAP dataset (optional)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default')
    
    args = parser.parse_args()
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set output path
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
        args.output_path = os.path.join(args.output_path, 'refined_free_gaussians')
    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Refined free gaussians will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/free_gaussians_refinement', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dense supervision (optional)
    if args.dense_data_path is not None:
        dense_arg = " ".join([
            "--dense_data_path", args.dense_data_path,
        ])
    else:
        dense_arg = ""

    # Define command
    command = " ".join([
        sys.executable, "2d-gaussian-splatting/train_with_charts.py",
        "-s", args.mast3r_scene,
        "-m", args.output_path,
        "--iterations", str(config['iterations']),
        "--densify_until_iter", str(config['densify_until_iter']),
        "--opacity_reset_interval", str(config['opacity_reset_interval']),
        "--depth_ratio", str(config['depth_ratio']),
        "--use_mip_filter" if config['use_mip_filter'] else "",
        dense_arg,
        "--normal_consistency_from", str(config['normal_consistency_from']),
        "--distortion_from", str(config['distortion_from']),
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
        "--dense_regul", args.dense_regul,
    ])
    
    # Run command
    CONSOLE.print(f"[INFO] Running command:\n{command}")
    os.system(command)