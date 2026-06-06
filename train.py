import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--source_path', type=str, required=True, help='Path to the source directory')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to the output directory')
    
    # Image selection parameters
    parser.add_argument('--n_images', type=int, default=None, 
        help='Number of images to use for optimization, sampled with constant spacing. If not provided, all images will be used.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images.')
    parser.add_argument('--randomize_images', action='store_true', 
        help='Shuffle training images before sampling with constant spacing. If image_idx is provided, this will be ignored.')
    
    # Dense supervision (Optional)
    parser.add_argument('--dense_supervision', action='store_true', 
        help='Use dense RGB supervision with a COLMAP dataset. Should only be used with --sfm_config posed.')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    
    # Output mesh parameters
    parser.add_argument('--use_multires_tsdf', action='store_true', help='Use multi-resolution TSDF fusion instead of adaptive tetrahedralization for mesh extraction (not recommended).')
    parser.add_argument('--no_interpolated_views', action='store_true', help='Disable interpolated views for mesh extraction.')
    
    # SfM config
    parser.add_argument('--sfm_config', type=str, default='unposed', help='Config for SfM. Should be "unposed" or "posed".')
    
    # Chart alignment config
    parser.add_argument('--alignment_config', type=str, default='default', help='Config for charts alignment')
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    

    
    # Free Gaussians config
    parser.add_argument('--free_gaussians_config', type=str, default=None, 
        help='Config for Free Gaussians refinement. '\
        'By default, the config used is "default" for sparse supervision, and "long" for dense supervision.'
    )
    
    # Multi-resolution TSDF config
    parser.add_argument('--tsdf_config', type=str, default='default', help='Config for multi-resolution TSDF fusion')
    
    # Tetrahedralization config
    parser.add_argument('--tetra_config', type=str, default='default', help='Config for adaptive tetrahedralization')
    parser.add_argument('--tetra_downsample_ratio', type=float, default=0.5, 
        help='Downsample ratio for tetrahedralization. We recommend starting with 0.5 and then decreasing to 0.25 '\
        'if the mesh is too dense, or increasing to 1.0 if the mesh is too sparse.'
    )
    
    # DA3 Options
    parser.add_argument('--use_da3', action='store_true', help='Use DA3 global poses instead of MASt3R SfM.')
    parser.add_argument('--da3_dir', type=str, default=None, help='Path to pre-computed DA3 directory.')

    parser.add_argument('--conf_coef', type=float, default=0.0, help='Confidence threshold coefficient to mask noisy DA3 depths.')
    
    # Run specific step
    parser.add_argument('--sfm_only', action='store_true', help='Only run the SfM step')
    parser.add_argument('--alignment_only', action='store_true', help='Only run the chart alignment step')
    parser.add_argument('--refinement_only', action='store_true', help='Only run the chart refinement step')
    parser.add_argument('--mesh_only', action='store_true', help='Only run the mesh extraction step')
    
    args = parser.parse_args()
    
    # Set output paths
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
    mast3r_scene_path = os.path.join(args.output_path, 'mast3r_sfm')
    aligned_charts_path = os.path.join(args.output_path, 'mast3r_sfm')
    free_gaussians_path = os.path.join(args.output_path, 'free_gaussians')
    tsdf_meshes_path = os.path.join(args.output_path, 'tsdf_meshes')
    tetra_meshes_path = os.path.join(args.output_path, 'tetra_meshes')
    
    # Dense supervision (Optional)
    if args.dense_supervision:
        dense_arg = " ".join([
            "--dense_data_path", args.source_path,
        ])
        if args.sfm_config != 'posed':
            print("[WARNING] Dense supervision is only supported for posed SfM. Switching to posed SfM.")
            args.sfm_config = 'posed'
    else:
        dense_arg = ""
        
    # Free Gaussians refinement default config
    if args.free_gaussians_config is None:
        args.free_gaussians_config = 'long' if args.dense_supervision else 'default'
    
    # Defining commands
    sfm_command = " ".join([
        sys.executable, "scripts/run_sfm.py",
        "--source_path", args.source_path,
        "--output_path", mast3r_scene_path,
        "--config", args.sfm_config,
        # "--env", args.sfm_env,
        "--n_images" if args.n_images is not None else "", str(args.n_images) if args.n_images is not None else "",
        "--image_idx" if args.image_idx is not None else "", " ".join([str(i) for i in args.image_idx]) if args.image_idx is not None else "",
        "--randomize_images" if args.randomize_images else "",
    ])
    
    align_charts_command = " ".join([
        sys.executable, "scripts/align_charts.py",
        "--source_path", mast3r_scene_path,
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", aligned_charts_path,
        "--config", args.alignment_config,
        "--depth_model", args.depth_model,
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
    ])
    
    refine_free_gaussians_command = " ".join([
        sys.executable, "scripts/refine_free_gaussians.py",
        "--mast3r_scene", mast3r_scene_path,
        "--output_path", free_gaussians_path,
        "--config", args.free_gaussians_config,
        dense_arg,
        "--dense_regul", args.dense_regul,
    ])
    
    tsdf_command = " ".join([
        sys.executable, "scripts/extract_tsdf_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tsdf_meshes_path,
        "--config", args.tsdf_config,
    ])
    
    tetra_command = " ".join([
        sys.executable, "scripts/extract_tetra_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tetra_meshes_path,
        "--config", args.tetra_config,
        "--downsample_ratio", str(args.tetra_downsample_ratio),
        "--interpolate_views" if not args.no_interpolated_views else "",
        dense_arg,
    ])
    
    # Running commands
    run_all = not any([args.sfm_only, args.alignment_only, args.refinement_only, args.mesh_only])
    
    if args.use_da3 or args.da3_dir is not None:
        if args.sfm_only or args.alignment_only or run_all:
            if args.da3_dir is None:
                print("[INFO] --use_da3 is set but --da3_dir is not provided. Running DA3 automatically...")
                args.da3_dir = os.path.abspath(os.path.join(args.output_path, 'da3_output'))
                da3_run_cmd = " ".join([
                    "conda", "run", "-n", "depth_anything_v3",
                    "python", "da3_streaming.py",
                    "--image_dir", os.path.abspath(args.source_path),
                    "--config", "./configs/base_config.yaml",
                    "--output_dir", args.da3_dir
                ])
                print(f"[INFO] Running DA3 Command: {da3_run_cmd}")
                da3_run_status = os.system(f"cd Depth-Anything-3/da3_streaming && {da3_run_cmd}")
                if da3_run_status != 0:
                    print("[ERROR] DA3 execution failed!")
                    sys.exit(1)
            
            da3_command = " ".join([
                sys.executable, "scripts/da3_to_matcha.py",
                "--source_path", args.source_path,
                "--da3_dir", args.da3_dir,
                "--output_path", mast3r_scene_path,
                "--image_idx" if args.image_idx is not None else "", " ".join([str(i) for i in args.image_idx]) if args.image_idx is not None else "",
                "--conf_coef", str(args.conf_coef),
            ])
            print(f"[INFO] Running command: {da3_command}")
            os.system(da3_command)
    else:
        if args.sfm_only or run_all:
            print(f"[INFO] Running command: {sfm_command}")
            os.system(sfm_command)
            
        if args.alignment_only or run_all:
            print(f"[INFO] Running command: {align_charts_command}")
            os.system(align_charts_command)
    if args.refinement_only or run_all:
        os.system(refine_free_gaussians_command)
    if args.mesh_only or run_all:
        if args.use_multires_tsdf:
            os.system(tsdf_command)
        else:
            os.system(tetra_command)
