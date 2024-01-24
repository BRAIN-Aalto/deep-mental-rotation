import torch
from math import pi


def deg2rad(angles: torch.Tensor) -> torch.Tensor:
    return angles * pi / 180.


def rad2deg(angles: torch.Tensor) -> torch.Tensor:
    return angles * 180. / pi


def rotation_matrix_y(angle: torch.Tensor) -> torch.Tensor:
    """Returns rotation matrix about y-axis.

    Args:
        angle (torch.Tensor): shape (batch_size,)
            Rotation angle in degrees.
    """
    # Initialize rotation matrix
    rotation_matrix = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    # Fill out matrix entries
    angle_rad = deg2rad(angle)

    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)
    
    rotation_matrix[:, 0, 0] = cos_angle
    rotation_matrix[:, 0, 2] = sin_angle
    rotation_matrix[:, 1, 1] = 1.
    rotation_matrix[:, 2, 0] = -sin_angle
    rotation_matrix[:, 2, 2] = cos_angle
    return rotation_matrix # (batch_size, 3, 3)


def rotation_matrix_z(angle: torch.Tensor) -> torch.Tensor:
    """Returns rotation matrix about z-axis.

    Args:
        angle (torch.Tensor): shape (batch_size,)
            Rotation angle in degrees.
    """
    # Initialize rotation matrix
    rotation_matrix = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    # Fill out matrix entries
    angle_rad = deg2rad(angle)

    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    rotation_matrix[:, 0, 0] = cos_angle
    rotation_matrix[:, 0, 1] = -sin_angle
    rotation_matrix[:, 1, 0] = sin_angle
    rotation_matrix[:, 1, 1] = cos_angle
    rotation_matrix[:, 2, 2] = 1.
    return rotation_matrix # (batch_size, 3, 3)


def reflection_matrix_x(batch_size: int, device: str) -> torch.Tensor:
    """
    Returns reflection matrix about plane z = 0.
    """
    # Initialize rotation matrix
    reflection_matrix = torch.zeros(batch_size, 3, 3, device=device)

    reflection_matrix[:, 0, 0] = -1
    reflection_matrix[:, 1, 1] = 1
    reflection_matrix[:, 2, 2] = 1

    return reflection_matrix # (batch_size, 3, 3)


def azimuth_elevation_to_rotation_matrix(
    azimuth: torch.Tensor, # (batch_size,)
    elevation: torch.Tensor # (batch_size,)
) -> torch.Tensor: # (batch_size, 3, 3)
    """Returns rotation matrix matching the default view
    (i.e. both azimuth and elevation are zero) to the view
    defined by the azimuth, elevation pair.

    Args:
        azimuth (torch.Tensor): shape (batch_size,).
            Azimuth of camera in **degrees**.

        elevation (torch.Tensor): shape (batch_size,).
            Elevation of camera in **degrees**.

    Notes:
        The azimuth and elevation refer to the **position of the camera**. This
        function returns the rotation of the *scene representation*,
        i.e. the inverse of the camera transformation.
    """
    # In the coordinate system we define (see README), azimuth rotation
    # corresponds to negative rotation about y axis and elevation rotation to a
    # negative rotation about z axis

    # negate angle because rotation matrix is defined in the right-handed
    # coordinate system, but we have rotations defined in the left-handed one
    azimuth_matrix = rotation_matrix_y(-azimuth) 
    elevation_matrix = rotation_matrix_z(-elevation)

    # We first perform elevation rotation followed by azimuth when rotating camera
    camera_matrix = azimuth_matrix @ elevation_matrix
    # Object rotation matrix is inverse (i.e. transpose) of camera rotation matrix
    return transpose_matrix(camera_matrix)


def rotation_matrix_source_to_target(
    azimuth_source: torch.Tensor, # (batch_size,)
    elevation_source: torch.Tensor, # (batch_size,)
    azimuth_target: torch.Tensor, # (batch_size,)
    elevation_target: torch.Tensor, # (batch_size,)
) -> torch.Tensor: # (batch_size, 3, 3)
    """Returns rotation matrix matching two views
    defined by azimuth, elevation pairs.

    Args:
        azimuth_source (torch.Tensor): shape (batch_size,).
            Azimuth of source view in degrees.

        elevation_source (torch.Tensor): shape (batch_size,).
            Elevation of source view in degrees.

        azimuth_target (torch.Tensor): shape (batch_size,).
            Azimuth of target view in degrees.

        elevation_target (torch.Tensor): shape (batch_size,).
            Elevation of target view in degrees.
    """
    # Calculate rotation matrix for each view
    rotation_source = azimuth_elevation_to_rotation_matrix(azimuth_source, elevation_source)
    rotation_target = azimuth_elevation_to_rotation_matrix(azimuth_target, elevation_target)
    
    # Calculate rotation matrix bringing source view to target view
    # Note: for rotation matrix, inverse is same as transpose
    return rotation_target @ transpose_matrix(rotation_source)


def rotate_from_source_mirror_rotate_to_target_matrix(
    azimuth_source: torch.Tensor, # (batch_size,)
    elevation_source: torch.Tensor, # (batch_size,)
    azimuth_target: torch.Tensor, # (batch_size,)
    elevation_target: torch.Tensor, # (batch_size,) 
):
    """
    """
    # Calculate rotation matrix for each view
    rotation_source = azimuth_elevation_to_rotation_matrix(azimuth_source, elevation_source)
    rotation_target = azimuth_elevation_to_rotation_matrix(azimuth_target, elevation_target)
    
    # Calculate rotation matrix bringing source view to target view
    # Note: for rotation matrix, inverse is same as transpose
    return rotation_target @ reflection_matrix_x(batch_size=azimuth_source.shape[0], device=azimuth_source.device) @ transpose_matrix(rotation_source)



def transpose_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Transposes a batch of matrices.

    Args:
        matrix (torch.Tensor): Batch of matrices of shape (batch_size, n, m).
    """
    return matrix.transpose(1, 2)
