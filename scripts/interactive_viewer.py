import os
import json
import math
import argparse
import gradio as gr
import numpy as np

def load_scene_data(mesh_dir):
    mesh_path = os.path.join(mesh_dir, "merged_tetra_mesh.ply")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Merged mesh not found at {mesh_path}")
        
    poses_path = os.path.join(mesh_dir, "vggt_output", "camera_poses.txt")
    intrinsics_path = os.path.join(mesh_dir, "vggt_output", "intrinsic.txt")
    
    if not os.path.exists(poses_path):
        poses_path = os.path.join(mesh_dir, "da3_output", "camera_poses.txt")
        intrinsics_path = os.path.join(mesh_dir, "da3_output", "intrinsic.txt")
        
    cameras = []
    
    if os.path.exists(poses_path) and os.path.exists(intrinsics_path):
        print(f"Loading global camera poses from {poses_path}...")
        with open(poses_path, 'r') as f:
            poses_lines = f.readlines()
        with open(intrinsics_path, 'r') as f:
            intrinsics_lines = f.readlines()
            
        for pose_line, intr_line in zip(poses_lines, intrinsics_lines):
            pose_vals = list(map(float, pose_line.strip().split()))
            c2w = np.array(pose_vals).reshape(4, 4).tolist()
            
            intr_vals = list(map(float, intr_line.strip().split()))
            fx = intr_vals[0]
            fov_rad = 2 * math.atan(512 / (2 * fx))
            fov_deg = math.degrees(fov_rad)
            
            cameras.append({
                "c2w": c2w,
                "fov": fov_deg
            })
    else:
        print("Global poses not found. Falling back to aggregating chunk cameras.json...")
        import glob
        chunk_dirs = sorted(glob.glob(os.path.join(mesh_dir, "chunk_*")))
        seen_files = set()
        
        for chunk_dir in chunk_dirs:
            cam_json_path = os.path.join(chunk_dir, "mast3r_sfm", "cameras.json")
            if not os.path.exists(cam_json_path):
                continue
                
            with open(cam_json_path, 'r') as f:
                cams_data = json.load(f)
                
            for filepath, c2w, focal in zip(cams_data["filepaths"], cams_data["cams2world"], cams_data["focals"]):
                if filepath in seen_files:
                    continue
                seen_files.add(filepath)
                
                fov_rad = 2 * math.atan(512 / (2 * focal))
                fov_deg = math.degrees(fov_rad)
                cameras.append({
                    "c2w": c2w,
                    "fov": fov_deg
                })
                
    if not cameras:
        raise FileNotFoundError(f"Could not find any camera poses in {mesh_dir}")
        
    aspect_ratio = 16 / 9
    import glob
    images = glob.glob(os.path.join(mesh_dir, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(mesh_dir, "**", "*.png"), recursive=True)
    if images:
        try:
            from PIL import Image
            with Image.open(images[0]) as img:
                aspect_ratio = img.width / img.height
        except Exception:
            pass
            
    return mesh_path, cameras, aspect_ratio

def build_html(mesh_path, cameras, initial_roll_deg=0.0, initial_pitch_deg=0.0, initial_yaw_deg=0.0, initial_camera_index=0, move_speed=1.0):
    template_path = os.path.join(os.path.dirname(__file__), "viewer_template.html")
    with open(template_path, 'r') as f:
        html = f.read()
        
    # We mounted args.mesh_dir as /mesh_data in FastAPI!
    # Append a cache-busting version tag based on file mtime+size so that browsers don't reuse
    # a cached PLY from a previous scene that happened to produce the same filename
    # (e.g. two scenes both using --max_faces 3000000 -> same merged_tetra_mesh_downsampled_3000000.ply).
    ply_filename = os.path.basename(mesh_path)
    try:
        st = os.stat(mesh_path)
        cache_tag = f"{int(st.st_mtime)}-{st.st_size}"
    except OSError:
        cache_tag = "0"
    ply_url = f"/mesh_data/{ply_filename}?v={cache_tag}"

    html = html.replace('"INJECT_PLY_URL"', f'"{ply_url}"')
    html = html.replace('INJECT_CAMERAS_JSON', json.dumps(cameras))
    html = html.replace('INJECT_INITIAL_ROLL_DEG', json.dumps(float(initial_roll_deg)))
    html = html.replace('INJECT_INITIAL_PITCH_DEG', json.dumps(float(initial_pitch_deg)))
    html = html.replace('INJECT_INITIAL_YAW_DEG', json.dumps(float(initial_yaw_deg)))
    html = html.replace('INJECT_INITIAL_CAMERA_INDEX', json.dumps(int(initial_camera_index)))
    html = html.replace('INJECT_MOVE_SPEED', json.dumps(float(move_speed)))
    
    # Escape HTML so it can be safely injected into the iframe's srcdoc attribute.
    import html as pyhtml
    return pyhtml.escape(html)

def get_or_create_downsampled_mesh(original_mesh_path, target_faces):
    downsampled_path = original_mesh_path.replace(".ply", f"_downsampled_{target_faces}.ply")
    
    # 1. Reuse if already processed
    if os.path.exists(downsampled_path):
        print(f"[INFO] Found existing downsampled mesh: {downsampled_path}")
        return downsampled_path
        
    try:
        import open3d as o3d
    except ImportError:
        print("[WARNING] open3d is not installed in this environment. Cannot downsample mesh.")
        print("         To enable downsampling, run: pip install open3d")
        return original_mesh_path
        
    print(f"[INFO] Loading original mesh to check size: {original_mesh_path}")
    mesh = o3d.io.read_triangle_mesh(original_mesh_path)
    current_faces = len(mesh.triangles)
    
    if current_faces <= target_faces:
        print(f"[INFO] Mesh only has {current_faces} faces (below target {target_faces}). Using original.")
        return original_mesh_path
        
    print(f"[INFO] Downsampling mesh from {current_faces} to {target_faces} faces... (This may take a minute)")
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    
    print(f"[INFO] Saving downsampled mesh to: {downsampled_path}")
    o3d.io.write_triangle_mesh(downsampled_path, mesh_smp)
    
    return downsampled_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, required=True, help="Path to scene output directory")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--max_faces", type=int, default=500000, help="Target number of faces for the browser viewer (default: 500k)")
    parser.add_argument("--initial_roll_deg", type=float, default=None,
                        help="Initial camera roll correction in degrees. Overrides viewer_config.json. Mast3r/DA3 world frames are not gravity-aligned, so try different values if the view is rotated.")
    parser.add_argument("--initial_pitch_deg", type=float, default=None,
                        help="Initial camera pitch correction in degrees (rotation around camera right). Overrides viewer_config.json.")
    parser.add_argument("--initial_yaw_deg", type=float, default=None,
                        help="Initial camera yaw correction in degrees (rotation around camera up). Overrides viewer_config.json.")
    parser.add_argument("--initial_camera_index", type=int, default=None,
                        help="Initial camera frame index. Overrides viewer_config.json.")
    parser.add_argument("--move_speed", type=float, default=None,
                        help="WASD/QE movement speed in world units per second. Defaults to scene_extent / 4.")
    args = parser.parse_args()

    # Per-scene config: viewer_config.json inside mesh_dir, if present, provides defaults that CLI args can override.
    scene_config_path = os.path.join(args.mesh_dir, "viewer_config.json")
    scene_config = {}
    if os.path.exists(scene_config_path):
        try:
            with open(scene_config_path, 'r') as f:
                scene_config = json.load(f)
            print(f"[INFO] Loaded per-scene config: {scene_config_path} -> {scene_config}")
        except Exception as e:
            print(f"[WARN] Failed to read {scene_config_path}: {e}")

    initial_roll_deg = args.initial_roll_deg if args.initial_roll_deg is not None else float(scene_config.get("initial_roll_deg", 0.0))
    initial_pitch_deg = args.initial_pitch_deg if args.initial_pitch_deg is not None else float(scene_config.get("initial_pitch_deg", 0.0))
    initial_yaw_deg = args.initial_yaw_deg if args.initial_yaw_deg is not None else float(scene_config.get("initial_yaw_deg", 0.0))
    initial_camera_index = args.initial_camera_index if args.initial_camera_index is not None else int(scene_config.get("initial_camera_index", 0))
    
    try:
        mesh_path, cameras, aspect_ratio = load_scene_data(args.mesh_dir)
    except Exception as e:
        print(f"Error loading scene: {e}")
        exit(1)
        
    # Process downsampling
    mesh_path = get_or_create_downsampled_mesh(mesh_path, args.max_faces)
        
    num_cameras = len(cameras)
    initial_camera_index = max(0, min(num_cameras - 1, initial_camera_index))

    # Auto-compute movement speed from camera position spread so WASD feels reasonable across scenes of any scale.
    cam_positions = [c["c2w"] for c in cameras]
    cam_xyz = [[row[3] for row in c2w[:3]] for c2w in cam_positions]
    if len(cam_xyz) >= 2:
        xs = [p[0] for p in cam_xyz]; ys = [p[1] for p in cam_xyz]; zs = [p[2] for p in cam_xyz]
        scene_extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    else:
        scene_extent = 1.0
    move_speed = args.move_speed if args.move_speed is not None else max(scene_extent / 4.0, 0.1)
    print(f"[INFO] Scene extent: {scene_extent:.3f}, WASD speed: {move_speed:.3f} units/s")

    html_srcdoc = build_html(mesh_path, cameras,
                             initial_roll_deg=initial_roll_deg,
                             initial_pitch_deg=initial_pitch_deg,
                             initial_yaw_deg=initial_yaw_deg,
                             initial_camera_index=initial_camera_index,
                             move_speed=move_speed)
    
    with gr.Blocks(title="MAtCha Interactive Viewer") as demo:
        gr.Markdown(f"# 🏙️ MAtCha Street View: {os.path.basename(args.mesh_dir)}")
        gr.Markdown("Click inside the viewer to focus, then drag to look around. Use **WASD** to walk, **Q/E** to fly down/up, hold **Shift** for speed boost. Slider or buttons jump along the camera trajectory.")
        
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous Frame", variant="secondary")
            cam_slider = gr.Slider(minimum=0, maximum=num_cameras-1, step=1, value=initial_camera_index, label="Camera Frame")
            next_btn = gr.Button("Next Frame ➡️", variant="primary")

        with gr.Row():
            roll_ccw_btn = gr.Button("↺ Roll -90°", variant="secondary")
            roll_slider = gr.Slider(minimum=-180, maximum=180, step=1, value=initial_roll_deg, label="Roll (deg) — spin around view axis")
            roll_cw_btn = gr.Button("↻ Roll +90°", variant="secondary")

        with gr.Row():
            save_btn = gr.Button("💾 Save current settings as scene default", variant="primary")
            save_status = gr.Markdown(visible=True, value=f"Config: `{scene_config_path}`")
            
        # The iframe using the dynamically calculated aspect ratio!
        iframe_html = f'<iframe id="viewer_iframe" srcdoc="{html_srcdoc}" style="width: 100%; aspect-ratio: {aspect_ratio}; max-height: 85vh; border: 2px solid #ccc; border-radius: 8px;"></iframe>'
        viewer = gr.HTML(iframe_html)
        
        # JS to send message to iframe
        send_msg_js = """
        function(idx) {
            var iframe = document.getElementById('viewer_iframe');
            if(iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({type: 'jumpToCamera', index: idx}, '*');
            }
            return idx;
        }
        """

        set_roll_js = """
        function(deg) {
            var iframe = document.getElementById('viewer_iframe');
            if(iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({type: 'setRoll', deg: deg}, '*');
            }
            return deg;
        }
        """
        
        def decrement(idx): return max(0, idx - 1)
        def increment(idx): return min(num_cameras - 1, idx + 1)

        def _wrap(deg): return ((deg + 180) % 360) - 180
        def roll_ccw(deg): return _wrap(deg - 90)
        def roll_cw(deg):  return _wrap(deg + 90)

        def save_scene_config(roll, cam_idx):
            cfg = {
                "initial_roll_deg": float(roll),
                "initial_pitch_deg": float(initial_pitch_deg),
                "initial_yaw_deg": float(initial_yaw_deg),
                "initial_camera_index": int(cam_idx),
            }
            try:
                with open(scene_config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                return f"✅ Saved `{cfg}` to `{scene_config_path}`"
            except Exception as e:
                return f"❌ Failed to save: {e}"
            
        prev_btn.click(fn=decrement, inputs=cam_slider, outputs=cam_slider)
        next_btn.click(fn=increment, inputs=cam_slider, outputs=cam_slider)
        cam_slider.change(fn=None, inputs=cam_slider, js=send_msg_js)

        roll_ccw_btn.click(fn=roll_ccw, inputs=roll_slider, outputs=roll_slider)
        roll_cw_btn.click(fn=roll_cw, inputs=roll_slider, outputs=roll_slider)
        roll_slider.change(fn=None, inputs=roll_slider, js=set_roll_js)

        save_btn.click(fn=save_scene_config,
                       inputs=[roll_slider, cam_slider],
                       outputs=save_status)
        
    print(f"Launching Gradio app for mesh: {mesh_path}")
    
    # Use FastAPI directly to absolutely guarantee static files are served correctly
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    
    app = FastAPI()
    # Mount the mesh directory so it's accessible via /mesh_data/...
    app.mount("/mesh_data", StaticFiles(directory=os.path.abspath(args.mesh_dir)), name="mesh_data")
    
    # Mount Gradio onto the FastAPI app
    app = gr.mount_gradio_app(app, demo, path="/")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
