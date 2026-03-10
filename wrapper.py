from dataclasses import dataclass
from typing import Dict, Union
import numpy as np
from pathlib import Path
import cv2
import trimesh

from .model import FoundationposeModel


@dataclass
class FoundationPoseWrapperConfig:
    downsample_width: int = None
    est_refine_iter: int = 5
    track_refine_iter: int = 2
    chunk_size: int = 63


def downsample_image_to_width(image, target_width):
    original_height, original_width = image.shape[:2]

    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)

    resized_image = cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_NEAREST
    )

    return resized_image


def adapt_camera_intrinsics_by_width(camera_matrix, original_width, target_width):
    scale_factor = target_width / original_width
    new_camera_matrix = camera_matrix.copy()

    new_camera_matrix[0, 0] *= scale_factor
    new_camera_matrix[1, 1] *= scale_factor
    new_camera_matrix[0, 2] *= scale_factor
    new_camera_matrix[1, 2] *= scale_factor

    return new_camera_matrix


class FoundationPoseWrapper:
    def __init__(
        self,
        camera_intrinsics: np.ndarray = None,
        cfg: Union[str, Path, FoundationPoseWrapperConfig] = None,
    ):
        if cfg is None:
            cfg = FoundationPoseWrapperConfig()
        self.cfg = cfg
        self.camera_intrinsics = camera_intrinsics

        self.est = {}
        self.poses = {}
        self.color = None
        self.depth = None
        self.masks = {}
        self._camera_intrinsics_downsampled = None

    def set_camera_intrinsics(self, camera_intrinsics: np.ndarray):
        """Sets the camera intrinsics. Call this before resetting the scene."""
        self.camera_intrinsics = camera_intrinsics

    def _downsample(self, color: np.ndarray, depth: np.ndarray):
        if self.cfg.downsample_width is None:
            self.color = color
            self.depth = depth
            self._camera_intrinsics_downsampled = self.camera_intrinsics
            return

        self.color = downsample_image_to_width(color, self.cfg.downsample_width)
        self.depth = downsample_image_to_width(depth, self.cfg.downsample_width)
        self._camera_intrinsics_downsampled = adapt_camera_intrinsics_by_width(
            self.camera_intrinsics,
            original_width=color.shape[1],  # Width
            target_width=self.cfg.downsample_width,
        )

    def reset_scene(self, color: np.ndarray, depth: np.ndarray):
        """Resets the wrapper to a new scene. Call this on the initial image."""

        self._downsample(color, depth)

        for _, v in self.est.items():
            del v
        self.est: Dict[str, FoundationposeModel] = {}
        self.poses = {}
        self.masks = {}

    def add_object(self, name: str, mesh: trimesh.Trimesh, mask: np.ndarray):
        """Adds an object to the scene. Call this sequentially for each object in the scene.
        Will trigger initial detection step.
        """

        est = FoundationposeModel(chunk_size=self.cfg.chunk_size)
        self.est[name] = est

        if mask.shape[0:2] != self.color.shape[0:2]:
            mask = downsample_image_to_width(
                mask.astype(np.uint8), self.cfg.downsample_width
            ).astype(bool)
        self.masks[name] = mask

        est.preprocess(mesh=mesh, intrinsics=self._camera_intrinsics_downsampled)

        pose = est.process(
            [self.color, self.depth], 0, mask=mask, iterations=self.cfg.est_refine_iter
        )

        self.poses[name] = pose
        return pose.cpu().numpy()

    def register_object(self, name: str):
        """Manually trigger detection step."""
        pose = self.est[name].process(
            [self.color, self.depth],
            0,
            mask=self.masks[name],
            iterations=self.cfg.est_refine_iter,
        )
        self.poses[name] = pose
        return pose.cpu().numpy()

    def step_scene(self, color: np.ndarray, depth: np.ndarray):
        """Steps the wrapper to the next frame. Call this on subsequent images.
        Corresponds to tracking step.
        """

        self._downsample(color, depth)

        for name, est in self.est.items():
            pose = est.process(
                [self.color, self.depth], 1, iterations=self.cfg.track_refine_iter
            )
            self.poses[name] = pose

        poses = {k: v.cpu().numpy() for k, v in self.poses.items()}
        return poses

    def get_poses(self) -> Dict[str, np.ndarray]:
        return self.poses

    @classmethod
    def load_mesh(cls, mesh_path: str):
        """Loads a mesh from a file."""
        mesh = trimesh.load(mesh_path)

        def as_mesh(scene_or_mesh):
            if isinstance(scene_or_mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        trimesh.Trimesh(
                            vertices=m.vertices, faces=m.faces, visual=m.visual
                        )
                        for m in scene_or_mesh.geometry.values()
                    ]
                )
            else:
                mesh = scene_or_mesh
            return mesh

        mesh = as_mesh(mesh)

        # Fixes some texture issues with certain mesh file formats
        mesh.export("/tmp/mesh.obj")
        mesh = trimesh.load_mesh("/tmp/mesh.obj", force="mesh")
        return mesh

    def render_results(self) -> np.ndarray:
        """Renders the results of the last frame. Returns the rendered image."""

        vis = self.color.copy()
        for name, pose in self.poses.items():
            vis = self.est[name].draw_image(vis, pose.cpu().numpy())
        return vis[..., ::-1]
