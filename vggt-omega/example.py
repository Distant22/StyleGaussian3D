import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "/workspace/Bill/MAtCha/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt"
image_names = ["images/csie_main_cont/LINE_ALBUM_csie_main_cont_260530_1.jpg", "images/csie_main_cont/LINE_ALBUM_csie_main_cont_260530_2.jpg", "images/csie_main_cont/LINE_ALBUM_csie_main_cont_260530_3.jpg"]

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]

# print(depth, depth_conf)