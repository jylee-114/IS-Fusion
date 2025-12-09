import torch


def cartesian_to_spherical(points_xyz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert Cartesian (x, y, z) to Spherical (r, theta, phi).

    Args:
        points_xyz: (..., 3) tensor (x, y, z)
        eps: small value for numerical stability

    Returns:
        sph: (..., 3) tensor (r, theta, phi)
             r     : sqrt(x^2 + y^2 + z^2)
             theta : polar angle from +z axis in [0, pi]
             phi   : azimuth angle in xy-plane in [-pi, pi]
    """
    assert points_xyz.size(-1) == 3, 'Last dimension of points_xyz must be 3.'

    x = points_xyz[..., 0]
    y = points_xyz[..., 1]
    z = points_xyz[..., 2]

    # Radius
    r = torch.sqrt(x * x + y * y + z * z + eps)

    # Theta: angle from +z axis
    cos_theta = torch.clamp(z / (r + eps), -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    # phi: azimuth in xy-plane
    phi = torch.atan2(y, x + eps)

    sph = torch.stack([r, theta, phi], dim=-1)
    return sph


def normalize_spherical(
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    r_max: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale spherical coordinates to approximately [0, 1]."""
    r_n = r / max(r_max, 1e-6)
    theta_n = theta / torch.pi                      # [0, pi] -> [0, 1]
    phi_n = (phi + torch.pi) / (2.0 * torch.pi)     # [-pi, pi] -> [0, 1]
    return r_n, theta_n, phi_n


def cartesian_to_spherical_with_norm(
    points_xyz: torch.Tensor,
    r_max: float = 80.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert Cartesian to Spherical coordinates and apply normalization."""
    sph = cartesian_to_spherical(points_xyz, eps=eps)
    r, theta, phi = sph[..., 0], sph[..., 1], sph[..., 2]
    r_n, theta_n, phi_n = normalize_spherical(r, theta, phi, r_max=r_max)
    sph_norm = torch.stack([r_n, theta_n, phi_n], dim=-1)
    return sph_norm


def spherical_fusion(
    points_xyz: torch.Tensor,
    others: torch.Tensor | None = None,
    fuse_mode: str = 'concat',
    use_normalize: bool = True,
    r_max: float = 80.0,
) -> torch.Tensor:
    """Fuse xyz and spherical coordinates in the desired mode.

    Args:
        points_xyz: (..., 3)
        others: (..., C_other) intensity etc.
        fuse_mode:
            - 'concat'         : [xyz, spherical, others]
            - 'replace'        : [spherical, others]
            - 'spherical_only' : [spherical] only
        use_normalize: If True, use normalized spherical coordinates
        r_max: Maximum distance for normalization
    """
    if use_normalize:
        sph = cartesian_to_spherical_with_norm(points_xyz, r_max=r_max)
    else:
        sph = cartesian_to_spherical(points_xyz)

    if fuse_mode == 'concat':
        if others is None:
            fused = torch.cat([points_xyz, sph], dim=-1)
        else:
            fused = torch.cat([points_xyz, sph, others], dim=-1)
    elif fuse_mode == 'replace':
        if others is None:
            fused = sph
        else:
            fused = torch.cat([sph, others], dim=-1)
    elif fuse_mode == 'spherical_only':
        fused = sph
    else:
        raise ValueError(f'Unknown fuse_mode: {fuse_mode}')

    return fused