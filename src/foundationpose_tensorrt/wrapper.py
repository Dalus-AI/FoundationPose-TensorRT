from dataclasses import dataclass
from typing import Dict, Union
import numpy as np
from pathlib import Path
import cv2
import trimesh
import torch

from foundationpose_tensorrt.model import FoundationposeModel


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

        self._shared_est = FoundationposeModel(chunk_size=self.cfg.chunk_size)
        self.objects = {}
        self.color = None
        self.depth = None
        self._camera_intrinsics_downsampled = None

    @staticmethod
    def _to_pose_np(pose) -> np.ndarray:
        if hasattr(pose, "detach"):
            pose = pose.detach()
        if hasattr(pose, "cpu"):
            pose = pose.cpu()
        if hasattr(pose, "numpy"):
            pose = pose.numpy()
        return np.asarray(pose, dtype=np.float32).reshape(4, 4)

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
        self.objects = {}

    def add_object(self, name: str, mesh: trimesh.Trimesh, mask: np.ndarray):
        """Adds an object to the scene. Call this sequentially for each object in the scene.
        Will trigger initial detection step.
        """
        if mask.shape[0:2] != self.color.shape[0:2]:
            mask = downsample_image_to_width(
                mask.astype(np.uint8), self.cfg.downsample_width
            ).astype(bool)
        self._shared_est.preprocess(mesh=mesh, intrinsics=self._camera_intrinsics_downsampled)
        pose, tracking_pose = self._shared_est.register(
            rgb=self.color,
            depth=self.depth,
            ob_mask=mask,
            mesh=self._shared_est.mesh,
            iteration=self.cfg.est_refine_iter,
        )
        pose_np = self._to_pose_np(pose)
        self.objects[name] = {
            "mesh": mesh,
            "mask": mask,
            "pose": pose_np,
            "tracking_pose": tracking_pose,
        }
        return pose_np

    def register_object(self, name: str):
        """Manually trigger detection step."""
        obj = self.objects[name]
        self._shared_est.preprocess(
            mesh=obj["mesh"],
            intrinsics=self._camera_intrinsics_downsampled,
        )
        pose, tracking_pose = self._shared_est.register(
            rgb=self.color,
            depth=self.depth,
            ob_mask=obj["mask"],
            mesh=self._shared_est.mesh,
            iteration=self.cfg.est_refine_iter,
        )
        pose_np = self._to_pose_np(pose)
        obj["pose"] = pose_np
        obj["tracking_pose"] = tracking_pose
        return pose_np

    def step_scene(self, color: np.ndarray, depth: np.ndarray):
        """Steps the wrapper to the next frame. Call this on subsequent images.
        Corresponds to tracking step.
        """

        self._downsample(color, depth)

        for name, obj in self.objects.items():
            self._shared_est.preprocess(
                mesh=obj["mesh"],
                intrinsics=self._camera_intrinsics_downsampled,
            )
            pose, tracking_pose = self._shared_est.track_one(
                self.color,
                self.depth,
                obj["tracking_pose"],
                iteration=self.cfg.track_refine_iter,
            )
            obj["pose"] = self._to_pose_np(pose)
            obj["tracking_pose"] = tracking_pose

        poses = {k: self._to_pose_np(v["pose"]) for k, v in self.objects.items()}
        return poses

    def get_poses(self) -> Dict[str, np.ndarray]:
        return {k: self._to_pose_np(v["pose"]) for k, v in self.objects.items()}

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
        for _, obj in self.objects.items():
            self._shared_est.preprocess(
                mesh=obj["mesh"],
                intrinsics=self._camera_intrinsics_downsampled,
            )
            vis = self._shared_est.draw_image(vis, self._to_pose_np(obj["pose"]))
        return vis[..., ::-1]
