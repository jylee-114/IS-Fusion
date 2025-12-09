import torch
import torch.nn as nn

from mmdet3d.models import VOXEL_ENCODERS
from mmdet3d.utils import spherical_fusion


@VOXEL_ENCODERS.register_module()
class SphericalFusionVFE(nn.Module):
    """VFE that converts Cartesian (x, y, z) to Spherical (r, theta, phi)
    and averages points within each voxel.

    Config example (PointPillars / CenterPoint etc):

        voxel_encoder=dict(
            type='SphericalFusionVFE',
            in_channels=5,          # (x,y,z,intensity,time etc)
            fuse_mode='concat',    # concat / replace / spherical_only
            use_normalize=True,
            r_max=80.0,
        )
    """

    def __init__(self,
                 in_channels: int,
                 fuse_mode: str = 'concat',
                 use_normalize: bool = True,
                 r_max: float = 80.0,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.fuse_mode = fuse_mode
        self.use_normalize = use_normalize
        self.r_max = float(r_max)

        assert self.fuse_mode in ['concat', 'replace', 'spherical_only']

        # Calculate output channels
        if self.fuse_mode == 'concat':
            # xyz(3) + spherical(3) + others(C-3)
            self.out_channels = in_channels + 3
        elif self.fuse_mode == 'replace':
            # xyz(3) -> spherical(3) substitution
            self.out_channels = in_channels
        else:  # spherical_only
            self.out_channels = 3

    def forward(self,
                features: torch.Tensor,
                num_points: torch.Tensor,
                coors: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (M, T, C_in)  Point features within voxels
            num_points: (M,)        Number of points per voxel
            coors: (M, 4)           (batch_idx, z, y, x) - not used here

        Returns:
            voxel_features: (M, C_out)
        """
        # 1) Separate xyz and others
        xyz = features[..., 0:3]
        others = features[..., 3:] if self.in_channels > 3 else None

        # 2) Spherical fusion
        fused = spherical_fusion(
            points_xyz=xyz,
            others=others,
            fuse_mode=self.fuse_mode,
            use_normalize=self.use_normalize,
            r_max=self.r_max,
        )   # (M, T, C_out)

        # 3) Voxel-wise mean aggregation (same as MeanVFE)
        points_sum = fused.sum(dim=1)  # (M, C_out)
        num_points = num_points.type_as(fused).view(-1, 1)
        num_points_clamped = torch.clamp_min(num_points, 1.0)
        voxel_out = points_sum / num_points_clamped

        return voxel_out