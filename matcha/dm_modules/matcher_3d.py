import torch
from matcha.dm_scene.cameras import CamerasWrapper, P3DCameras
from matcha.dm_utils.rendering import depths_to_points_parallel


def get_points_depth_in_depthmap_parallel(
    pts:torch.Tensor, 
    depthmap:torch.Tensor, 
    cameras:CamerasWrapper,
    padding_mode='zeros',  # 'reflection', 'border'
    znear=1e-6,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (n_depths, N, 3).
        depthmap (torch.Tensor): Has shape (n_depths, H, W) or (n_depths, H, W, 1).
        p3d_camera (P3DCameras): Should contain n_depths cameras.

    Returns:
        _type_: _description_
    """
    n_depths, image_height, image_width = depthmap.shape[:3]

    pts_projections = cameras.transform_points_world_to_view(pts)  # (n_depths, N, 3)
    fov_mask = pts_projections[..., 2] > 0.  # (n_depths, N)
    pts_projections.clamp(min=torch.tensor([[[-1e8, -1e8, znear]]]).to(pts_projections.device))
    
    pts_projections = cameras.project_points(pts_projections, points_are_already_in_view_space=True, znear=znear)  # (n_depths, N, 2)
    fov_mask = fov_mask & pts_projections.isfinite().all(dim=-1)  # (n_depths, N)
    pts_projections = pts_projections.nan_to_num(nan=0., posinf=0., neginf=0.)
    
    if False:
        print("TOREMOVE-pts_projections:", pts_projections.shape)
        print("TOREMOVE-pts_projections Min/Max/Mean/Std:", pts_projections.min(), pts_projections.max(), pts_projections.mean(), pts_projections.std())
        
    factor = -1 * min(image_height, image_width)
    factors = torch.tensor([[[factor / image_width, factor / image_height]]]).to(pts.device)  # (1, 1, 2)
    # pts_projections[..., 0] = factor / image_width * pts_projections[..., 0]
    # pts_projections[..., 1] = factor / image_height * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2] * factors  # (n_depths, N, 2)
    pts_projections = pts_projections.view(n_depths, -1, 1, 2)

    depth_view = depthmap.reshape(n_depths, 1, image_height, image_width)  # (n_depths, 1, H, W)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=False,
    )  # (n_depths, 1, N, 1)
    map_z = map_z[:, 0, :, 0]  # (n_depths, N)
    fov_mask = (map_z > 0.) & fov_mask
    map_z = map_z * fov_mask
    
    return map_z, fov_mask


class Matcher3D:
    def __init__(
        self, 
        cameras:CamerasWrapper,
        reference_pts:torch.Tensor=None, 
        reference_depths:torch.Tensor=None,
    ):
        """_summary_

        Args:
            reference_pts (torch.Tensor): Should have shape (n_charts, height, width, 3).
            reference_depths (torch.Tensor): Should have shape (n_charts, height, width).
            camera (CamerasWrapper): _description_
            match_thr (float): _description_
        """
        self.cameras = cameras
        self.znear = 1e-6
        self.update_references(reference_pts, reference_depths)
        
    @torch.no_grad()
    def update_references(
        self, 
        reference_pts:torch.Tensor=None, 
        reference_depths:torch.Tensor=None,
    ):
        if reference_pts is None and reference_depths is None:
            raise ValueError("Either reference_pts or reference_depths should be provided.")
        
        if reference_depths is None:  
            reference_depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                reference_pts
            )[..., 2]  # (n_charts, height, width)
            
        if reference_pts is None:
            reference_pts = depths_to_points_parallel(
                reference_depths,
                cameras=self.cameras,
            ).view(*reference_depths.shape, 3)  # (n_charts, height, width, 3)
            
        self.reference_pts = reference_pts  # (n_charts, height, width, 3)
        self.reference_depths = reference_depths  # (n_charts, height, width)
        self.n_charts, self.height, self.width, _ = reference_pts.shape
        self.reference_pts = reference_pts
        
    @torch.no_grad()
    def match(
        self, 
        matching_thr:float, 
        normal_threshold=None
    ):
        if normal_threshold is not None:
            raise NotImplementedError("Normal threshold not implemented yet.")
        
        self.valid_pairs = []
        self.valid_masks = []
        
        chunk_size = 20 
        
        for i_start in range(0, self.n_charts, chunk_size):
            i_end = min(self.n_charts, i_start + chunk_size)
            curr_n = i_end - i_start
            
            chunk_cameras = CamerasWrapper(
                [self.cameras.gs_cameras[i] for i in range(i_start, i_end)],
                no_p3d_cameras=self.cameras.no_p3d_cameras
            )
            chunk_depths = self.reference_depths[i_start:i_end] 
            
            for j in range(self.n_charts):
                pts_j = self.reference_pts[j].view(1, -1, 3).repeat(curr_n, 1, 1) 
                
                true_depth = chunk_cameras.p3d_cameras.get_world_to_view_transform().transform_points(pts_j)[..., 2]
                
                projected_depths, fov_mask = get_points_depth_in_depthmap_parallel(
                    pts=pts_j,
                    depthmap=chunk_depths,
                    cameras=chunk_cameras,
                    padding_mode='zeros',
                    znear=self.znear,
                )
                
                depth_errors = (true_depth - projected_depths).abs()
                depth_errors[~fov_mask] = 1e8
                
                is_match = depth_errors < matching_thr
                
                for local_i in range(curr_n):
                    global_i = i_start + local_i
                    mask = is_match[local_i].view(self.height, self.width)
                    if mask.any():
                        self.valid_pairs.append((global_i, j))
                        self.valid_masks.append(mask.detach().clone())
                        
        print(f"[Matcher3D] Found {len(self.valid_pairs)} valid pairs out of {self.n_charts * self.n_charts} total pairs.")
    
    def compute_matching_loss(self, depths, points=None, confidence=None):
        if points is None:
            points = depths_to_points_parallel(depths, cameras=self.cameras)
            
        total_loss = 0.0
        
        chunk_size = 20
        n_pairs = len(self.valid_pairs)
        
        for start_idx in range(0, n_pairs, chunk_size):
            end_idx = min(n_pairs, start_idx + chunk_size)
            curr_n = end_idx - start_idx
            
            i_list = [self.valid_pairs[idx][0] for idx in range(start_idx, end_idx)]
            j_list = [self.valid_pairs[idx][1] for idx in range(start_idx, end_idx)]
            ref_masks = torch.stack([self.valid_masks[idx] for idx in range(start_idx, end_idx)]) # (curr_n, H, W)
            
            chunk_cameras = CamerasWrapper(
                [self.cameras.gs_cameras[i] for i in i_list],
                no_p3d_cameras=self.cameras.no_p3d_cameras
            )
            
            chunk_depths = depths[i_list] # (curr_n, H, W)
            chunk_pts = points[j_list].view(curr_n, -1, 3) # (curr_n, H*W, 3)
            
            true_depth = chunk_cameras.p3d_cameras.get_world_to_view_transform().transform_points(chunk_pts)[..., 2]
            
            projected_depths, fov_mask = get_points_depth_in_depthmap_parallel(
                pts=chunk_pts,
                depthmap=chunk_depths,
                cameras=chunk_cameras,
                padding_mode='zeros',
                znear=self.znear,
            )
            
            depth_errors = (true_depth - projected_depths).abs().nan_to_num() # (curr_n, H*W)
            
            mask = fov_mask.view(curr_n, self.height, self.width) & ref_masks
            error = depth_errors.view(curr_n, self.height, self.width) * mask
            
            if confidence is not None:
                chunk_confidence = confidence[j_list] # (curr_n, H, W)
                error = error * chunk_confidence
                
            total_loss = total_loss + error.sum()
            
        mean_loss = total_loss / (self.n_charts * self.n_charts * self.height * self.width)
        return mean_loss
