import os
import argparse
import platform
import subprocess

def get_cuda_paths():
    cuda_dir = None
    # Check common locations
    for path in ['/usr/local/cuda', '/usr/local/cuda-13.0', '/usr/local/cuda-12.0', '/usr/local/cuda-11.8']:
        if os.path.exists(path):
            cuda_dir = path
            break
    if not cuda_dir:
        cuda_dir = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    
    arch = None
    targets_dir = os.path.join(cuda_dir, 'targets')
    if os.path.exists(targets_dir):
        subdirs = os.listdir(targets_dir)
        if 'sbsa-linux' in subdirs:
            arch = 'sbsa-linux'
        elif 'x86_64-linux' in subdirs:
            arch = 'x86_64-linux'
        elif subdirs:
            arch = subdirs[0]
            
    if arch:
        cpath = os.path.join(cuda_dir, 'targets', arch, 'include')
        ld_path = os.path.join(cuda_dir, 'targets', arch, 'lib')
    else:
        cpath = os.path.join(cuda_dir, 'include')
        ld_path = os.path.join(cuda_dir, 'lib64') if os.path.exists(os.path.join(cuda_dir, 'lib64')) else os.path.join(cuda_dir, 'lib')
        
    bin_path = os.path.join(cuda_dir, 'bin')
    return bin_path, cpath, ld_path, cuda_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')
    
    parser.add_argument('--env_name', type=str, default='matcha', help='Name of the conda environment to create')
    args = parser.parse_args()
    
    is_aarch64 = platform.machine() == 'aarch64'
    
    # Create a new conda environment
    env_exists = False
    try:
        output = subprocess.check_output("conda env list", shell=True).decode('utf-8')
        # Check if the environment name exists as a separate word/path
        for line in output.split('\n'):
            parts = line.split()
            if parts and parts[0] == args.env_name:
                env_exists = True
                break
    except Exception:
        pass

    if env_exists:
        print(f"[INFO] Conda environment {args.env_name} already exists. Skipping creation.")
    else:
        print(f"[INFO] Creating the conda environment {args.env_name} for MAtCha...")
        if is_aarch64:
            phystwin_exists = False
            try:
                output = subprocess.check_output("conda env list", shell=True).decode('utf-8')
                if 'phystwin' in output:
                    phystwin_exists = True
            except Exception:
                pass
                
            if phystwin_exists:
                print("[INFO] Detected aarch64 and phystwin environment. Cloning phystwin...")
                os.system(f"conda create --name {args.env_name} --clone phystwin -y")
                print("[INFO] Installing additional conda dependencies (cmake, cgal-cpp, gmp)...")
                os.system(f"conda install -n {args.env_name} -c conda-forge cmake cgal-cpp gmp -y")
                print("[INFO] Installing additional pip packages (roma, faiss-cpu)...")
                os.system(f"conda run -n {args.env_name} pip install roma==1.5.0 faiss-cpu")
            else:
                print("[WARNING] Detected aarch64 but phystwin environment was not found. Installing from scratch...")
                os.system(f"conda create --name {args.env_name} -y python=3.10")
                os.system(f"conda install -n {args.env_name} -c conda-forge cmake cgal-cpp gmp -y")
                os.system(f"conda run -n {args.env_name} pip install torch torchvision torchaudio")
                print("[INFO] Installing pytorch3d from source...")
                os.system(f"conda run -n {args.env_name} pip install iopath")
                os.system(f"conda run -n {args.env_name} pip install git+https://github.com/facebookresearch/pytorch3d.git@b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881")
                os.system(f"conda run -n {args.env_name} pip install roma==1.5.0 open3d==0.18.0 opencv-python scipy einops trimesh pyglet==1.5.29 tensorboard scikit-learn cython faiss-cpu tqdm matplotlib huggingface-hub[torch] gradio plyfile")
        else:
            os.system(f"conda env create -f environment.yml -n {args.env_name}")
        
    print(f"[INFO] Conda environment {args.env_name} created.")
    
    # Configure CUDA environments dynamically
    bin_path, cpath, ld_path, cuda_dir = get_cuda_paths()
    os.environ["PATH"] = f"{bin_path}:{os.environ.get('PATH', '')}"
    os.environ["CPATH"] = f"{cpath}:{os.environ.get('CPATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["CUDA_HOME"] = cuda_dir
    os.environ["CUDACXX"] = os.path.join(bin_path, "nvcc")
    print(f"[INFO] CUDA paths configured:")
    print(f"  CUDA_HOME: {cuda_dir}")
    print(f"  PATH: {bin_path}")
    print(f"  CPATH: {cpath}")
    print(f"  LD_LIBRARY_PATH: {ld_path}")
    print(f"  CUDACXX: {os.environ['CUDACXX']}")
    
    # Install 2D Gaussian Splatting rasterizer
    print(f"\n[INFO] Installing the 2D Gaussian Splatting rasterizer in the conda environment {args.env_name}...")
    os.chdir("2d-gaussian-splatting/submodules/diff-surfel-rasterization/")
    os.system(f"conda run -n {args.env_name} pip install -e . --no-build-isolation")
    print(f"[INFO] 2D Gaussian Splatting rasterizer installed in the conda environment {args.env_name}.")
    
    # Install simple-knn
    print(f"\n[INFO] Installing simple-knn in the conda environment {args.env_name}...")
    os.chdir("../simple-knn/")
    os.system(f"conda run -n {args.env_name} pip install -e . --no-build-isolation")
    print(f"[INFO] simple-knn installed in the conda environment {args.env_name}.")
    
    # Install tetra-triangulation
    print(f"\n[INFO] Installing tetra-triangulation in the conda environment {args.env_name}...")
    os.chdir("../tetra-triangulation/")
    os.system(f"conda run -n {args.env_name} cmake -DCMAKE_CUDA_COMPILER={os.path.join(bin_path, 'nvcc')} .")
    os.system(f"conda run -n {args.env_name} make")
    os.system(f"conda run -n {args.env_name} pip install -e . --no-build-isolation")
    print(f"[INFO] tetra-triangulation installed in the conda environment {args.env_name}.")
    os.chdir("../../../")
        
    # Install ASMK
    print(f"\n[INFO] Installing ASMK...")
    os.chdir("./mast3r/asmk/cython")
    os.system(f"conda run -n {args.env_name} cythonize *.pyx")
    os.chdir("../")
    os.system(f"conda run -n {args.env_name} pip install . --no-build-isolation")
    print("[INFO] ASMK installed.")
    
    # Compile cuda kernels for RoPE
    print(f"\n[INFO] Compiling cuda kernels for RoPE...")
    os.chdir("../dust3r/croco/models/curope/")
    os.system(f"conda run -n {args.env_name} python setup.py build_ext --inplace")
    print("[INFO] RoPE cuda kernels compiled.")
    
    os.chdir("../../../../../")
    print("[INFO] MAtCha installation complete.")