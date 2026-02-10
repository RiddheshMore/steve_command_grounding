from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def _pose_from_dimension(dim):
    if dim == 2:
        return Pose2D
    elif dim == 3:
        return Pose3D


class Pose:
    def __init__(self, dimension: int, coordinates: np.ndarray, rot_matrix: np.ndarray):
        self.dimension: int = dimension
        self.coordinates: np.ndarray = coordinates
        self.rot_matrix: np.ndarray = rot_matrix

    def as_tuple(self) -> tuple:
        return tuple(self.coordinates.tolist())

    def as_ndarray(self) -> np.ndarray:
        return self.coordinates

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(self.dimension + 1)
        matrix[: self.dimension, : self.dimension] = self.rot_matrix
        matrix[: self.dimension, self.dimension] = self.coordinates
        return matrix

    def transform(self, tform: np.ndarray, side: str = "left") -> None:
        self_mat = self.as_matrix()
        if side == "left":
            new_mat = tform @ self_mat
        elif side == "right":
            new_mat = self_mat @ tform
        else:
            raise ValueError(f"Unknown side {side}!")
        self.rot_matrix = new_mat[:3, :3]
        self.coordinates = new_mat[:3, 3]

    @abc.abstractmethod
    def as_pose(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dimension(self, dimension: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def direction(self, normalized: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self):
        coords = (f"{coord:.2f}" for coord in self.coordinates)
        coords = f'({", ".join(coords)})'
        direction = (f"{x:.2f}" for x in self.direction())
        direction = f'({", ".join(direction)})'
        return f"Pose{self.dimension}D(coords={coords}, direction={direction})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        PoseClass = _pose_from_dimension(self.dimension)
        return PoseClass(self.coordinates.copy(), self.rot_matrix.copy())

    def copy(self):
        return self.__copy__()

    def __matmul__(self, other):
        assert isinstance(other, Pose), f"other is type {type(other)}!"
        max_dim = max(self.dimension, other.dimension)
        if max_dim > self.dimension or max_dim > other.dimension:
            this = self.to_dimension(max_dim)
            other = other.to_dimension(max_dim)
        else:
            this = self

        rot_matrix = this.as_matrix() @ other.as_matrix()
        return _pose_from_dimension(max_dim).from_matrix(rot_matrix)


def _rot_matrix_from_angle(angle: float) -> np.ndarray:
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _angle_from_rot_matrix(rot_matrix: np.ndarray) -> float:
    return math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])


def _rotation_from_direction(
    direction: tuple[float, float, float],
    roll: float,
    invariant_direction: tuple[float, float, float],
    degrees: bool,
    invert: bool = True,
) -> np.ndarray:
    direction = np.asarray(direction)
    invariant_direction = np.asarray(invariant_direction)
    start_vector = invariant_direction.reshape((1, 3))
    end_vector = direction.reshape((1, 3))
    if invert:
        pitch_yaw_rotation = Rotation.align_vectors(-end_vector, start_vector)[0]
    else:
        pitch_yaw_rotation = Rotation.align_vectors(end_vector, start_vector)[0]

    if degrees:
        roll = roll / 360 * math.tau
    roll_rotation = Rotation.from_euler("xyz", [roll, 0, 0])
    full_rotation = pitch_yaw_rotation * roll_rotation
    return full_rotation.as_matrix()


class Pose2D(Pose):
    def __init__(
        self,
        coordinates: Optional[(tuple | np.ndarray)] = None,
        rot_matrix: Optional[(np.ndarray | float)] = None,
    ):
        self.dimension = 2
        coordinates = self.compute_coordinates(coordinates)
        rot_matrix = self.compute_rot_matrix(rot_matrix)
        super().__init__(self.dimension, coordinates, rot_matrix)

    def compute_coordinates(self, coordinates: Optional[(tuple | np.ndarray)]) -> np.ndarray:
        if coordinates is None:
            coordinates = np.zeros((self.dimension,))
        elif isinstance(coordinates, tuple):
            coordinates = np.asarray(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = coordinates.reshape((-1,))
        assert coordinates.shape == (self.dimension,)
        return coordinates

    def compute_rot_matrix(self, rot_matrix: Optional[(np.ndarray | float)]) -> np.ndarray:
        if rot_matrix is None:
            rot_matrix = np.eye(self.dimension)
        elif isinstance(rot_matrix, np.ndarray):
            rot_matrix = rot_matrix
        elif isinstance(rot_matrix, (float, int)):
            rot_matrix = _rot_matrix_from_angle(float(rot_matrix))
        assert rot_matrix.shape == (self.dimension, self.dimension)
        return rot_matrix

    def to_dimension(self, dimension: int):
        if dimension == 2:
            return self
        elif dimension == 3:
            coordinates = np.zeros((3,))
            coordinates[:2] = self.coordinates[:2]
            rot_matrix = np.eye(3)
            rot_matrix[:2, :2] = self.rot_matrix
            return Pose3D(coordinates, rot_matrix)

    def direction(self, normalized: bool = True) -> np.ndarray:
        result = self.rot_matrix @ np.asarray([1, 0])
        if normalized:
            result = result / np.linalg.norm(result, 2)
        return result

    def set_rot_from_angle(self, angle: float, degrees: bool = False) -> None:
        rot_matrix = Rotation.from_euler("z", angle, degrees=degrees).as_matrix()
        self.rot_matrix = rot_matrix[:2, :2]

    @staticmethod
    def from_array(array: np.ndarray) -> Pose2D:
        assert array.shape == (3,)
        coordinates = (array[0], array[1])
        rot_matrix = _rot_matrix_from_angle(array[2])
        return Pose2D(coordinates, rot_matrix)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose2D:
        assert matrix.shape == (3, 3)
        return Pose2D(matrix[:2, 2], matrix[:2, :2])


class Pose3D(Pose):
    def __init__(
        self,
        coordinates: Optional[(tuple | np.ndarray)] = None,
        rot_matrix: Optional[(np.ndarray | Rotation)] = None,
    ):
        self.dimension = 3
        coordinates = self.compute_coordinates(coordinates)
        rot_matrix = self.compute_rot_matrix(rot_matrix)
        super().__init__(self.dimension, coordinates, rot_matrix)

    def get_yaw(self) -> float:
        """Extract yaw from rotation matrix."""
        # Assume Z-axis verticality, extract yaw from matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(self.rot_matrix)
        # Sequence xyz is usually what's used in these ports
        _, _, yaw = r.as_euler("xyz")
        return yaw

    def compute_coordinates(self, coordinates: Optional[(tuple | np.ndarray)]) -> np.ndarray:
        if coordinates is None:
            coordinates = np.zeros((self.dimension,))
        elif isinstance(coordinates, tuple):
            coordinates = np.asarray(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = coordinates.reshape((-1,))
        assert coordinates.shape == (self.dimension,)
        return coordinates

    def compute_rot_matrix(self, rot_matrix: Optional[(np.ndarray | Rotation)]) -> np.ndarray:
        if rot_matrix is None:
            rot_matrix = np.eye(self.dimension)
        elif isinstance(rot_matrix, np.ndarray):
            rot_matrix = rot_matrix
        elif isinstance(rot_matrix, Rotation):
            rot_matrix = rot_matrix.as_matrix()
        assert rot_matrix.shape == (self.dimension, self.dimension)
        return rot_matrix

    def to_dimension(self, dimension: int):
        if dimension == 2:
            coordinates = self.coordinates[:2]
            _, _, yaw = Rotation.from_matrix(self.rot_matrix).as_euler("xyz")
            rot_matrix = _rot_matrix_from_angle(yaw)
            return Pose2D(coordinates, rot_matrix)
        elif dimension == 3:
            return self

    def set_rot_from_rpy(self, rpy: tuple[float, float, float], degrees: bool = False) -> None:
        self.rot_matrix = Rotation.from_euler("xyz", rpy, degrees=degrees).as_matrix()

    def set_rot_from_direction(
        self,
        direction: tuple[float, float, float],
        roll: float = 0,
        invariant_direction: tuple[float, float, float] = (1, 0, 0),
        degrees: bool = False,
    ) -> None:
        self.rot_matrix = _rotation_from_direction(direction, roll, invariant_direction, degrees, invert=False)

    def set_from_scipy_rotation(self, rotation: Rotation) -> None:
        rot_matrix = rotation.as_matrix()
        self.rot_matrix = rot_matrix

    def direction(self, normalized: bool = True) -> np.ndarray:
        result = self.rot_matrix @ np.asarray([1, 0, 0])
        if normalized:
            result = result / np.linalg.norm(result, 2)
        return result

    def inverse(self, inplace: bool = False) -> Pose3D:
        matrix = self.as_matrix()
        matrix_inv = np.linalg.inv(matrix)
        if inplace:
            self.rot_matrix = matrix_inv[:3, :3]
            self.coordinates = matrix_inv[:3, 3]
        return self.from_matrix(matrix_inv)

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose3D:
        assert matrix.shape == (4, 4)
        return Pose3D(matrix[:3, 3], matrix[:3, :3])

    @staticmethod
    def from_scipy_rotation(rotation: Rotation) -> Pose3D:
        pose = Pose3D()
        pose.set_from_scipy_rotation(rotation)
        return pose


def average_pose3Ds(poses: list[Pose3D]) -> Pose3D:
    coordinates = np.stack([pose.coordinates for pose in poses], axis=0)
    rot_matrices = [pose.rot_matrix for pose in poses]
    avg_coordinate = np.mean(coordinates, axis=0)
    sum_matrix = sum(rot_matrices)
    U, _, VT = np.linalg.svd(sum_matrix, full_matrices=True)
    avg_rot_matrix = np.dot(U, VT)
    return Pose3D(avg_coordinate, avg_rot_matrix)


def from_a_to_b_distanced(start_pose: Pose2D, end_pose: Pose2D, distance: float) -> Pose2D:
    start = start_pose.as_ndarray()
    end = end_pose.as_ndarray()
    path = end - start
    yaw = math.atan2(path[1], path[0])
    path_norm = np.linalg.norm(path)
    wanted_norm = path_norm - distance
    rescaled_path = path / path_norm * wanted_norm
    destination = start + rescaled_path
    return Pose2D(destination, yaw)


def pose_distanced(pose: Pose3D, distance: float, negate: bool = True) -> Pose3D:
    m = -1 if negate else 1
    grasp_start_point_without_offset = m * pose.direction(normalized=True) * distance
    grasp_start_point = grasp_start_point_without_offset + pose.as_ndarray()
    return Pose3D(grasp_start_point, pose.rot_matrix)


def _polar_to_cartesian(vector: np.ndarray) -> np.ndarray:
    r = vector[..., 0]
    theta = vector[..., 1]
    phi = vector[..., 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def get_arc_view_poses(start_pose: Pose3D, target_pose: Pose3D, offset: float) -> list[Pose3D]:
    target_coordinates = target_pose.as_ndarray()
    start_rot_matrix = start_pose.rot_matrix
    distance = np.linalg.norm(target_coordinates - start_pose.as_ndarray())
    offset = np.radians(offset)
    thetas = np.linspace(0, np.pi, 5)
    rolls = np.zeros(thetas.shape)
    pitchs = np.sin(thetas) * offset
    yaws = -np.cos(thetas) * offset
    rpys = np.stack((rolls, pitchs, yaws), axis=1)
    poses = []
    for rpy in rpys:
        pitch_yaw_matrix = Rotation.from_euler("xyz", rpy).as_matrix()
        rot_matrix = start_rot_matrix @ pitch_yaw_matrix
        new_pose = Pose3D(target_coordinates, rot_matrix)
        poses.append(pose_distanced(new_pose, distance, negate=True))
    return poses


def get_circle_points(resolution: int, nr_circles: int, start_radius: float = 0.0, end_radius: float = 1.0, return_cartesian: bool = True) -> np.ndarray:
    radius = np.linspace(start_radius, end_radius, nr_circles)
    thetas = np.ones((resolution, nr_circles)) * np.pi / 2
    phi = np.linspace(0, math.tau, resolution + 1)[:-1]
    radii, phis = np.meshgrid(radius, phi)
    vectors = np.stack((radii, thetas, phis), axis=-1)
    vectors = np.transpose(vectors, (1, 0, 2))
    if return_cartesian:
        vectors = _polar_to_cartesian(vectors)
    return vectors
