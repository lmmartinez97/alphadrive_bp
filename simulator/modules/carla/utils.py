import numpy as np

from typing import Tuple

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def get_angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.

    Args:
        v1 tuple-like: First vector.
        v2 tuple-like: Second vector.

    Returns:
        float: Angle (in radians) between 'v1' and 'v2'.

    Examples:
        >>> get_angle_between(np.array([1, 0, 0]), np.array([0, 1, 0]))
        1.5707963267948966
        >>> get_angle_between(np.array([1, 0, 0]), np.array([1, 0, 0]))
        0.0
        >>> get_angle_between(np.array([1, 0, 0]), np.array([-1, 0, 0]))
        3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_vector_angle(v):
    """ 
    Returns the angle in radians between vector 'v' and the x-axis:
        Args:
            v1 tuple-like: First vector.
            v2 tuple-like: Second vector.

        Returns:
            float: Angle (in radians) between 'v1' and 'v2'.

        Examples:
            >>> get_vector_angle((1, 0, 0))
            0.0
            >>> get_vector_angle((0, 1, 0))
            1.5707963267948966
            >>> get_vector_angle((-1, 0, 0))
            3.141592653589793
            >>> get_vector_angle((0, -1, 0))
            4.71238898038469
    """
    if len(v) == 2:
        return get_angle_between(v, (1, 0))
    else:
        return get_angle_between(v, (1, 0, 0))
    
def get_straight_angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """
    Returns the angle in radians between the straight line formed by the end-point of vectors 'v1' and 'v2'
    and the x-axis.

    Args:
        v1 tuple-like: First vector.
        v2 tuple-like: Second vector.

    Returns:
        float: Angle (in radians) between the straight line formed by 'v1' and 'v2'
               and the x-axis.

    Examples:
        >>> get_angle_straight(np.array([1, 0]), np.array([0, 1]))
        1.5707963267948966
        >>> get_angle_straight(np.array([1, 0]), np.array([-1, 0]))
        3.141592653589793
        >>> get_angle_straight(np.array([0, 0]), np.array([0, 1]))
        1.5707963267948966
    """
    return get_vector_angle(np.subtract(np.array(v2),np.array(v1)))