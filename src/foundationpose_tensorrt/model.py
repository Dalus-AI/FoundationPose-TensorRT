import os
from pathlib import Path
import trimesh
import numpy as np
from transformations import euler_matrix
from pytorch3d.transforms import so3_exp_map
from collections import OrderedDict, namedtuple
import tensorrt as trt
import torch

from foundationpose_tensorrt import postprocessor

Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))


class EngineWrapper:
    """TensorRT engine wrapper that performs automatic batch chunking based on engine stats."""

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        """
        Initializes the EngineWrapper.

        Args:
            engine_path (str): Path to the TensorRT .plan file.
            device (str): CUDA device to run on (e.g., "cuda:0").
        """
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

        with open(engine_path, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create engine context")

        self.bindings = self.get_binding_info(self.engine)
        self.input_names = [
            name
            for name, _ in self.bindings.items()
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name, _ in self.bindings.items()
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

        profile_shapes = self.engine.get_tensor_profile_shape(self.input_names[0], 0)
        max_shape = profile_shapes[2]
        self.chunk_size = max_shape[0]
        print(
            f"Engine '{engine_path.split('/')[-1]}' loaded. Max batch size (chunk_size) automatically set to: {self.chunk_size}"
        )

    @staticmethod
    def get_binding_info(engine):
        bindings = OrderedDict()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = tuple(engine.get_tensor_shape(name))
            bindings[name] = Binding(name, dtype, shape, None, None)
        return bindings

    def _infer_chunk(self, A: torch.Tensor, B: torch.Tensor):
        current_batch_size = A.shape[0]

        self.context.set_input_shape(self.input_names[0], A.shape)
        self.context.set_input_shape(self.input_names[1], B.shape)

        binding_addrs = {}
        binding_addrs[self.input_names[0]] = A.data_ptr()
        binding_addrs[self.input_names[1]] = B.data_ptr()

        outputs = []
        for name in self.output_names:
            shape = (current_batch_size, *self.bindings[name].shape[1:])
            if shape[-1] == -1:  # TODO: Ugly
                shape = (shape[0],)
            output_tensor = torch.empty(
                size=shape, dtype=torch.float32, device=self.device
            )
            binding_addrs[name] = output_tensor.data_ptr()
            outputs.append(output_tensor)

        ordered_binding_addrs = [
            binding_addrs[self.engine.get_tensor_name(i)]
            for i in range(self.engine.num_io_tensors)
        ]
        self.context.execute_v2(bindings=ordered_binding_addrs)

        return outputs

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        total_size = A.shape[0]

        if total_size <= self.chunk_size:
            # If the input batch is smaller or equal to what the engine can handle, process it directly.
            # We still call _infer_chunk to handle the binding logic.
            return self._infer_chunk(A, B)

        all_chunk_outputs = []
        # If the input batch is larger, split it into chunks of the size the engine supports.
        for i in range(0, total_size, self.chunk_size):
            A_chunk = A[i : i + self.chunk_size]
            B_chunk = B[i : i + self.chunk_size]

            chunk_outputs = self._infer_chunk(A_chunk, B_chunk)
            all_chunk_outputs.append(chunk_outputs)

        # Concatenate the results from all chunks
        num_outputs = len(self.output_names)
        final_outputs = []
        for out_idx in range(num_outputs):
            output_chunks = [
                chunk_output[out_idx] for chunk_output in all_chunk_outputs
            ]
            concatenated_output = torch.cat(output_chunks, dim=0)
            final_outputs.append(concatenated_output)

        return final_outputs


class FoundationposeModel:
    def __init__(self, chunk_size=63):
        self.scale = 1
        self.mean = [0.0, 0.0, 0.0]
        self.mean = np.asarray(self.mean).astype(np.float32)
        self.data_format = "channels_last"  # lol
        if self.data_format == "channels_first":
            self.mean = self.mean[:, np.newaxis, np.newaxis]

        # Initialize TensorRT engines
        scorer_file = f"scorer_cs{chunk_size}.plan"
        refiner_file = f"refiner_cs{chunk_size}.plan"
        abs_path = Path(os.path.abspath(__file__)).parent.parent.parent
        scorer_path = os.path.join(abs_path, "weights", "tensorrt", scorer_file)
        refiner_path = os.path.join(abs_path, "weights", "tensorrt", refiner_file)
        self.scorer_engine = EngineWrapper(scorer_path, device="cuda:0")
        self.refiner_engine = EngineWrapper(refiner_path, device="cuda:0")

    def make_rotation_grid(self, min_n_views=40, inplane_step=60, device="cuda"):
        cam_in_obs = postprocessor.sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        self.rot_grid = torch.as_tensor(rot_grid, device=device, dtype=torch.float)

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3, dtype=np.float32)
        uc = float((us.min() + us.max()) / 2.0)
        vc = float((vs.min() + vs.max()) / 2.0)
        valid = mask.astype(bool) & (depth >= 0.1)
        if not valid.any():
            return np.zeros(3, dtype=np.float32)

        zc = float(np.median(depth[valid]))
        K_f32 = np.asarray(K, dtype=np.float32)
        center = (np.linalg.inv(K_f32) @ np.array([uc, vc, 1.0], dtype=np.float32).reshape(3, 1)) * zc
        return center.reshape(3).astype(np.float32)

    def generate_random_pose_hypo(self, K, depth, mask, device):
        """
        @scene_pts: torch tensor (N,3)
        """
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(
            center, device=device, dtype=torch.float
        ).reshape(1, 3)
        return ob_in_cams

    def preprocess(self, mesh, intrinsics, min_n_views=40, inplane_step=60):
        self.mesh = mesh
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = np.asarray(to_origin, dtype=np.float32)
        self.extent_bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3).astype(np.float32)
        self.mesh_reset, self.mesh_tensor, self.diameter, self.model_center = (
            postprocessor.reset_object(
                model_pts=np.asarray(self.mesh.vertices, dtype=np.float32),
                model_normals=np.asarray(self.mesh.vertex_normals, dtype=np.float32),
                mesh=self.mesh,
            )
        )
        self.K = np.asarray(intrinsics, dtype=np.float32)
        self.make_rotation_grid(min_n_views=min_n_views, inplane_step=inplane_step)
        self.tracking_pose = None

    def refiner_predict(
        self,
        rgb,
        depth,
        K,
        ob_in_cams,
        xyz_map,
        normal_map=None,
        mesh=None,
        mesh_tensors=None,
        mesh_diameter=None,
        iteration=5,
    ):
        tf_to_center = np.eye(4)
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh

        crop_ratio = 1.2
        B_in_cams = torch.as_tensor(
            ob_centered_in_cams, device="cuda", dtype=torch.float
        )
        rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device="cuda", dtype=torch.float)

        trans_normalizer = [
            0.019999999552965164,
            0.019999999552965164,
            0.05000000074505806,
        ]
        trans_normalizer = torch.as_tensor(
            list(trans_normalizer), device="cuda", dtype=torch.float
        ).reshape(1, 3)
        render_size = (160, 160)

        for _ in range(iteration):
            pose_data = postprocessor.make_crop_data_batch(
                render_size,
                B_in_cams,
                mesh_centered,
                rgb_tensor,
                depth_tensor,
                K,
                crop_ratio=crop_ratio,
                normal_map=normal_map,
                xyz_map=xyz_map_tensor,
                mesh_tensors=mesh_tensors,
                mesh_diameter=mesh_diameter,
            )

            B_in_cams = []
            A = torch.cat(
                [pose_data.rgbAs.cuda(), pose_data.xyz_mapAs.cuda()], dim=1
            ).float()
            B = torch.cat(
                [pose_data.rgbBs.cuda(), pose_data.xyz_mapBs.cuda()], dim=1
            ).float()

            A = A.permute(0, 2, 3, 1).contiguous()
            B = B.permute(0, 2, 3, 1).contiguous()

            trans, rot = self.refiner_engine.forward(A, B)

            trans_delta = trans

            rot_mat_delta = torch.tanh(rot) * 0.3490658503988659
            rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)

            trans_delta *= mesh_diameter / 2

            B_in_cam = postprocessor.egocentric_delta_pose_to_pose(
                pose_data.poseA, trans_delta=trans_delta, rot_mat_delta=rot_mat_delta
            )
            B_in_cams.append(B_in_cam)

            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)

        B_in_cams_out = B_in_cams @ torch.tensor(
            tf_to_center[None], device="cuda", dtype=torch.float
        )

        torch.cuda.empty_cache()

        return B_in_cams_out

    @torch.inference_mode()
    def find_best_among_pairs(self, pose_data: postprocessor.BatchPoseData):
        ids = []
        scores = []

        A = torch.cat(
            [pose_data.rgbAs.cuda(), pose_data.xyz_mapAs.cuda()], dim=1
        ).float()
        B = torch.cat(
            [pose_data.rgbBs.cuda(), pose_data.xyz_mapBs.cuda()], dim=1
        ).float()

        if pose_data.normalAs is not None:
            A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
            B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)

        A = A.permute(0, 2, 3, 1).contiguous()
        B = B.permute(0, 2, 3, 1).contiguous()

        score_logit = self.scorer_engine.forward(A, B)[0]

        scores_cur = score_logit.float().reshape(-1)
        ids.append(scores_cur.argmax())
        scores.append(scores_cur)
        ids = torch.stack(ids, dim=0).reshape(-1)
        scores = torch.cat(scores, dim=0).reshape(-1)
        return ids, scores

    def scorer_predict(
        self,
        rgb,
        depth,
        K,
        ob_in_cams,
        mesh=None,
        mesh_tensors=None,
        mesh_diameter=None,
    ):
        """
        @rgb: np array (H,W,3)
        """
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")

        rgb = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)

        pose_data = postprocessor.make_crop_data_batch_score(
            (160, 160),
            ob_in_cams,
            mesh,
            rgb,
            depth,
            K,
            crop_ratio=1.1,
            mesh_tensors=mesh_tensors,
            mesh_diameter=mesh_diameter,
        )

        pose_data_iter = pose_data
        global_ids = torch.arange(len(ob_in_cams), device="cuda", dtype=torch.long)
        scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device="cuda")

        while 1:
            ids, scores = self.find_best_among_pairs(pose_data_iter)
            if len(ids) == 1:
                scores_global[global_ids] = scores + 100
                break
            global_ids = global_ids[ids]
            pose_data_iter = pose_data.select_by_indices(global_ids)

        scores = scores_global

        torch.cuda.empty_cache()

        return scores

    def register(self, rgb, depth, ob_mask, mesh, iteration=2, device="cuda"):
        if isinstance(depth, np.ndarray):
            depth = depth.astype(np.float32)
        depth = postprocessor.bilateral_filter_depth(depth, radius=2, device=device)

        normal_map = None
        self.H, self.W = depth.shape[:2]
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(
            K=self.K, depth=depth, mask=self.ob_mask, device=device
        )
        center = self.guess_translation(depth=depth, mask=self.ob_mask, K=self.K)
        poses = torch.as_tensor(poses, device=device, dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device=device)

        xyz_map = postprocessor.depth2xyzmap(depth, self.K)

        poses = self.refiner_predict(
            mesh=mesh,
            mesh_tensors=self.mesh_tensor,
            rgb=rgb,
            depth=depth,
            K=self.K,
            ob_in_cams=poses.data.cpu().numpy(),
            normal_map=normal_map,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            iteration=iteration,
        )

        scores = self.scorer_predict(
            mesh=mesh,
            rgb=rgb,
            depth=depth,
            K=self.K,
            ob_in_cams=poses.data.cpu().numpy(),
            mesh_tensors=self.mesh_tensor,
            mesh_diameter=self.diameter,
        )

        ids = torch.as_tensor(scores).argsort(descending=True)
        scores = scores[ids]
        poses = poses[ids]
        best_pose = poses[0] @ postprocessor.get_tf_to_centered_mesh(self.model_center)

        return best_pose, poses[0]

    def track_one(self, rgb, depth, pose_last, iteration=2):
        if isinstance(pose_last, np.ndarray):
            pose_last = torch.tensor(pose_last)

        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth = postprocessor.bilateral_filter_depth(depth, radius=2, device="cuda")

        xyz_map = postprocessor.depth2xyzmap_batch(
            depth[None],
            torch.as_tensor(self.K, dtype=torch.float, device="cuda")[None],
            zfar=np.inf,
        )[0]

        pose = self.refiner_predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensor,
            rgb=rgb,
            depth=depth,
            K=self.K,
            ob_in_cams=pose_last.reshape(1, 4, 4).data.cpu().numpy(),
            normal_map=None,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            iteration=iteration,
        )

        return (pose @ postprocessor.get_tf_to_centered_mesh(self.model_center)).reshape(
            4, 4
        ), pose

    def draw_image(self, color, pose):
        center_pose = np.array(pose) @ np.linalg.inv(np.array(self.to_origin))
        vis = postprocessor.draw_posed_3d_box(
            self.K, img=color, ob_in_cam=center_pose, bbox=self.extent_bbox
        )
        vis = postprocessor.draw_xyz_axis(
            color,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    def process(self, batched, sent_count, bbox=None, mask=None, iterations=2):
        image, depth = batched
        H, W = depth.shape

        if sent_count == 0:
            if bbox is not None:
                bbox = bbox.split(",")
                umin, vmin, umax, vmax = bbox
                umin, vmin, umax, vmax = int(umin), int(vmin), int(umax), int(vmax)
                mask = np.zeros((H, W))
                mask[vmin:vmax, umin:umax] = 1
            elif mask is None:
                raise ValueError(
                    "Either bbox or mask must be provided for the first frame."
                )

            pose, self.tracking_pose = self.register(
                rgb=image,
                depth=depth,
                ob_mask=mask,
                mesh=self.mesh,
                iteration=iterations,
            )
        else:
            pose, self.tracking_pose = self.track_one(
                image, depth, self.tracking_pose, iteration=iterations
            )
        return pose
