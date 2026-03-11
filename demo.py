import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from foundationpose_tensorrt import FoundationPoseWrapper, FoundationPoseWrapperConfig
from rfdetr import RFDETRSegMedium


DEFAULT_CLASS_MAP = {
    0: "relay",
    1: "switch",
    2: "switch",
}


def load_rfdetr_model(checkpoint_path=None, device="cuda"):
    """Load RF-DETR model with broad constructor compatibility."""
    base_kwargs = {"device": device}

    def _try_load(**kwargs):
        try:
            return RFDETRSegMedium(**kwargs)
        except TypeError:
            return None

    def _ensure_device(model):
        if hasattr(model, "to"):
            model = model.to(device)
        return model

    if not checkpoint_path:
        model = _try_load(**base_kwargs) or RFDETRSegMedium()
        return _ensure_device(model)

    attempts = [
        "pretrain_weights",
        "checkpoint_path",
        "checkpoint",
        "weights",
        "weights_path",
        "model_path",
    ]
    for arg_name in attempts:
        model = _try_load(**{**base_kwargs, arg_name: checkpoint_path})
        if model is None:
            model = _try_load(**{arg_name: checkpoint_path})
        if model is not None:
            return _ensure_device(model)

    raise RuntimeError(
        "Could not load checkpoint with RFDETRSegMedium constructor. "
        "Try a different RF-DETR version or constructor arg name."
    )


def intrinsics_matrix_from_color_frame(color_frame):
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    return np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def detections_masks_to_numpy(detections, height, width):
    masks = getattr(detections, "mask", None)
    if masks is None:
        return np.zeros((0, height, width), dtype=np.float32)
    masks_np = np.asarray(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.ndim != 3:
        return np.zeros((0, height, width), dtype=np.float32)
    if masks_np.shape[1] != height or masks_np.shape[2] != width:
        resized = []
        for m in masks_np:
            resized.append(cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST))
        masks_np = np.stack(resized, axis=0)
    return masks_np.astype(np.float32)


def parse_class_map(raw_json: str | None):
    if not raw_json:
        return dict(DEFAULT_CLASS_MAP)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return dict(DEFAULT_CLASS_MAP)
    if not isinstance(data, dict):
        return dict(DEFAULT_CLASS_MAP)
    out = dict(DEFAULT_CLASS_MAP)
    for k, v in data.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def class_id_to_label(class_id: int, class_map: dict[int, str]) -> str:
    if class_id in class_map:
        return class_map[class_id]
    if (class_id + 1) in class_map:
        return class_map[class_id + 1]
    return f"class_{class_id}"


def collect_instances(detections, masks, class_map, label_filter=None, min_pixels=50):
    if masks.shape[0] == 0:
        return [], [], np.zeros((0,), dtype=np.float32)

    class_ids = np.asarray(getattr(detections, "class_id", []), dtype=np.int32).reshape(-1)
    if class_ids.size == 0:
        return [], [], np.zeros((0,), dtype=np.float32)

    conf = np.asarray(
        getattr(detections, "confidence", np.ones_like(class_ids, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if conf.size != class_ids.size:
        conf = np.ones_like(class_ids, dtype=np.float32)

    labels = [class_id_to_label(int(cid), class_map) for cid in class_ids.tolist()]
    n = min(masks.shape[0], len(labels), len(conf))
    instances = []
    for i in range(n):
        label = labels[i]
        if label_filter is not None and label != label_filter:
            continue
        m = masks[i] > 0.5
        if int(np.count_nonzero(m)) < int(min_pixels):
            continue
        instances.append(
            {
                "det_idx": i,
                "label": label,
                "conf": float(conf[i]),
                "mask": m,
            }
        )
    return instances, labels[:n], conf[:n]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0.0, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(0.0, (bx2 - bx1) * (by2 - by1)))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def summarize_instances(instances, depth_mm):
    out = []
    h, w = depth_mm.shape[:2]
    for i, inst in enumerate(instances):
        m = np.asarray(inst["mask"], dtype=np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(m > 0.5)
        if len(xs) == 0:
            continue
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        valid = (m > 0.5) & (depth_mm > 0)
        mean_depth_mm = float(np.mean(depth_mm[valid])) if np.any(valid) else None
        out.append(
            {
                "cur_list_idx": i,
                "bbox_xyxy": (x1, y1, x2, y2),
                "center_xy": (cx, cy),
                "mean_depth_mm": mean_depth_mm,
                "label": inst["label"],
            }
        )
    return out


def compute_instance_update_status(
    current_instances,
    prev_instances,
    center_displacement_threshold_px=15.0,
    depth_change_threshold_mm=5.0,
    min_iou_to_match=0.05,
    max_match_center_distance_px=120.0,
):
    status = {}
    cur_to_prev = {}
    # 1) Build one-to-one mapping by closest center distance (label-consistent).
    pairs = []
    for cur in current_instances:
        cur_idx = int(cur["cur_list_idx"])
        cur_cx, cur_cy = cur["center_xy"]
        cur_label = cur.get("label")
        for prev in prev_instances:
            prev_idx = int(prev["cur_list_idx"])
            if cur_label != prev.get("label"):
                continue
            prev_cx, prev_cy = prev["center_xy"]
            dist = float(np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2))
            pairs.append((dist, cur_idx, prev_idx))
    pairs.sort(key=lambda x: x[0])

    used_cur = set()
    used_prev = set()
    for dist, cur_idx, prev_idx in pairs:
        if cur_idx in used_cur or prev_idx in used_prev:
            continue
        if np.isfinite(max_match_center_distance_px) and dist > max_match_center_distance_px:
            continue
        cur_to_prev[cur_idx] = prev_idx
        used_cur.add(cur_idx)
        used_prev.add(prev_idx)

    prev_by_idx = {int(p["cur_list_idx"]): p for p in prev_instances}
    cur_by_idx = {int(c["cur_list_idx"]): c for c in current_instances}

    # 2) Determine updated/not_updated from mapped pairs.
    for cur_idx, cur in cur_by_idx.items():
        if cur_idx not in cur_to_prev:
            status[cur_idx] = "updated"
            continue

        prev_idx = int(cur_to_prev[cur_idx])
        prev = prev_by_idx.get(prev_idx)
        if prev is None:
            status[cur_idx] = "updated"
            continue

        cur_bbox = cur["bbox_xyxy"]
        prev_bbox = prev["bbox_xyxy"]
        iou = bbox_iou(cur_bbox, prev_bbox)
        cur_cx, cur_cy = cur["center_xy"]
        prev_cx, prev_cy = prev["center_xy"]
        disp_2d = float(np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2))

        cur_depth_mm = cur.get("mean_depth_mm")
        prev_depth_mm = prev.get("mean_depth_mm")
        depth_changed = False
        if (
            cur_depth_mm is not None
            and prev_depth_mm is not None
            and np.isfinite(cur_depth_mm)
            and np.isfinite(prev_depth_mm)
        ):
            depth_changed = abs(float(cur_depth_mm) - float(prev_depth_mm)) > depth_change_threshold_mm

        if disp_2d > center_displacement_threshold_px or depth_changed or iou < min_iou_to_match:
            status[cur_idx] = "updated"
        else:
            status[cur_idx] = "not_updated"
    return status, cur_to_prev


def draw_seg_overlay(color_bgr, masks, labels, conf):
    out = color_bgr.copy()
    if masks.shape[0] == 0:
        return out

    rng = np.random.default_rng(123)
    colors = rng.integers(low=0, high=255, size=(masks.shape[0], 3), dtype=np.uint8)
    for i in range(masks.shape[0]):
        m = masks[i] > 0.5
        if not np.any(m):
            continue
        color = tuple(int(c) for c in colors[i])
        out[m] = (0.65 * out[m] + 0.35 * np.array(color, dtype=np.float32)).astype(np.uint8)

        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)

        label = labels[i] if i < len(labels) else f"obj_{i}"
        score = float(conf[i]) if i < len(conf) else 0.0
        cv2.putText(
            out,
            f"{label}:{score:.2f}",
            (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.1:
        return (img * 255.0).clip(0, 255).astype(np.uint8)
    return img.clip(0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense + RF-DETR + FoundationPose demo")
    parser.add_argument("--checkpoint", type=str, default=os.environ.get("RFDETR_CHECKPOINT"), help="RF-DETR checkpoint path")
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("RFDETR_THRESHOLD", "0.7")), help="RF-DETR confidence threshold")
    parser.add_argument("--target-label", type=str, default=os.environ.get("RFDETR_TARGET_LABEL", "all"), help="Instance label to track ('all' for every instance)")
    parser.add_argument("--blur-ksize", type=int, default=int(os.environ.get("RFDETR_BLUR_KSIZE", "5")), help="Gaussian blur kernel size for RF-DETR input")
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("DEMO_MAX_FRAMES", "0")), help="Stop after N frames; 0 means infinite")
    parser.add_argument("--device", type=str, default=os.environ.get("RFDETR_DEVICE", "cuda"), help="RF-DETR device")
    parser.add_argument("--center-thresh-px", type=float, default=float(os.environ.get("INSTANCE_CENTER_THRESH_PX", "15.0")), help="2D center movement threshold")
    parser.add_argument("--depth-thresh-mm", type=float, default=float(os.environ.get("INSTANCE_DEPTH_THRESH_MM", "5.0")), help="mean depth change threshold")
    parser.add_argument("--iou-thresh", type=float, default=float(os.environ.get("INSTANCE_IOU_THRESH", "0.05")), help="IoU threshold for instance matching")
    parser.add_argument("--match-max-center-px", type=float, default=float(os.environ.get("INSTANCE_MATCH_MAX_CENTER_PX", "120.0")), help="max center distance allowed for current->prev mapping")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[demo] started", flush=True)

    script_dir = Path(__file__).resolve().parent
    results_dir = Path(os.environ.get("FRAME_OUTPUT_DIR", str(script_dir / "demo_data" / "mustard0" / "results")))
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] Writing rendered frames to: {results_dir}", flush=True)

    remy_perception_dir = Path(os.environ.get("REMY_PERCEPTION_DIR", "/workspace/remy-pose-estimation/src/perception"))
    default_mesh = remy_perception_dir / "mesh" / "switch_obj_scaled" / "switch_scaled.obj"
    mesh_file = Path(os.environ.get("FOUNDATIONPOSE_MESH_FILE", str(default_mesh)))
    if not mesh_file.exists():
        raise FileNotFoundError(f"Switch mesh not found: {mesh_file}")

    class_map = parse_class_map(os.environ.get("RFDETR_CLASS_ID_TO_LABEL_JSON"))

    fp_cfg = FoundationPoseWrapperConfig(
        downsample_width=int(os.environ.get("FOUNDATIONPOSE_DOWNSAMPLE_WIDTH", "256")),
        est_refine_iter=int(os.environ.get("FOUNDATIONPOSE_EST_REFINE_ITER", "0")),
        track_refine_iter=int(os.environ.get("FOUNDATIONPOSE_TRACK_REFINE_ITER", "1")),
        chunk_size=int(os.environ.get("FOUNDATIONPOSE_CHUNK_SIZE", "32")),
    )
    fp_wrapper = FoundationPoseWrapper(cfg=fp_cfg)
    mesh = FoundationPoseWrapper.load_mesh(str(mesh_file))

    print(f"[demo] Loading RF-DETR (device={args.device}, checkpoint={args.checkpoint})", flush=True)
    seg_model = load_rfdetr_model(args.checkpoint, device=args.device)
    if hasattr(seg_model, "optimize_for_inference"):
        seg_model.optimize_for_inference()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    print(f"[demo] depth_scale={depth_scale}", flush=True)

    show_viz = os.environ.get("FP_SHOW_VIZ", "0") == "1"
    ui_available = False
    if show_viz:
        try:
            cv2.namedWindow("segmentation", cv2.WINDOW_NORMAL)
            cv2.namedWindow("foundationpose", cv2.WINDOW_NORMAL)
            ui_available = True
        except cv2.error as exc:
            print(f"[demo] HighGUI unavailable, running headless: {exc}", flush=True)

    initialized = False
    tracked_object_names = []
    prev_instances = []
    frame_idx = 0
    label_filter = None if str(args.target_label).strip().lower() == "all" else str(args.target_label).strip()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if frame_idx == 0:
                intrinsics = intrinsics_matrix_from_color_frame(color_frame)
                fp_wrapper.set_camera_intrinsics(intrinsics)
                print(f"[demo] camera intrinsics set:\n{intrinsics}", flush=True)

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            blur_ksize = max(1, int(args.blur_ksize))
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            infer_bgr = cv2.GaussianBlur(color_bgr, (blur_ksize, blur_ksize), 0) if blur_ksize > 1 else color_bgr
            infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)

            frame_t0 = time.perf_counter()
            seg_t0 = time.perf_counter()
            detections = seg_model.predict(infer_rgb, threshold=float(args.threshold))
            seg_ms = (time.perf_counter() - seg_t0) * 1000.0

            h, w = color_bgr.shape[:2]
            masks = detections_masks_to_numpy(detections, h, w)
            instances, labels, conf = collect_instances(detections, masks, class_map, label_filter=label_filter)
            cur_instances = summarize_instances(instances, depth_raw.astype(np.float32))
            if prev_instances:
                instance_update_status, cur_to_prev = compute_instance_update_status(
                    cur_instances,
                    prev_instances,
                    center_displacement_threshold_px=float(args.center_thresh_px),
                    depth_change_threshold_mm=float(args.depth_thresh_mm),
                    min_iou_to_match=float(args.iou_thresh),
                    max_match_center_distance_px=float(args.match_max_center_px),
                )
                changed_count = sum(1 for v in instance_update_status.values() if v == "updated")
            else:
                instance_update_status = {int(c["cur_list_idx"]): "updated" for c in cur_instances}
                cur_to_prev = {}
                changed_count = len(cur_instances)

            reg_ms = 0.0
            render = color_bgr
            reg_t0 = time.perf_counter()
            fp_wrapper.set_frame(color_rgb, depth_m)
            prev_idx_to_track = {
                int(p["cur_list_idx"]): str(p["track_name"])
                for p in prev_instances
                if "track_name" in p
            }
            active_track_names = set()
            next_prev_instances = []

            for cur in cur_instances:
                cur_idx = int(cur["cur_list_idx"])
                inst = instances[cur_idx]
                track_name = None
                if cur_idx in cur_to_prev:
                    prev_idx = int(cur_to_prev[cur_idx])
                    track_name = prev_idx_to_track.get(prev_idx)

                if track_name and track_name in fp_wrapper.objects:
                    if instance_update_status.get(cur_idx, "updated") == "updated":
                        fp_wrapper.objects[track_name]["mask"] = np.asarray(inst["mask"], dtype=bool)
                        fp_wrapper.register_object(track_name)
                    # not_updated: keep previous pose as-is
                    cur["track_name"] = track_name
                else:
                    track_name = f"{inst['label']}_{frame_idx}_{cur_idx}"
                    fp_wrapper.add_object(track_name, mesh, np.asarray(inst["mask"], dtype=bool))
                    cur["track_name"] = track_name

                active_track_names.add(track_name)
                next_prev_instances.append(cur)

            stale = [name for name in list(fp_wrapper.objects.keys()) if name not in active_track_names]
            for name in stale:
                del fp_wrapper.objects[name]

            tracked_object_names = sorted(list(fp_wrapper.objects.keys()))
            prev_instances = next_prev_instances
            reg_ms = (time.perf_counter() - reg_t0) * 1000.0
            initialized = len(tracked_object_names) > 0
            if frame_idx == 0 and initialized:
                print(f"[demo] initialized FoundationPose for {len(tracked_object_names)} instance(s)", flush=True)

            if initialized:
                render_raw = fp_wrapper.render_results()
                render = to_uint8(render_raw)
                if render.ndim == 3 and render.shape[2] == 3:
                    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)

            seg_overlay = draw_seg_overlay(color_bgr, masks, labels, conf)
            total_ms = (time.perf_counter() - frame_t0) * 1000.0
            fps = (1000.0 / total_ms) if total_ms > 0 else 0.0
            status = f"FPS:{fps:.1f}  det:{len(instances)}  tracked:{len(tracked_object_names)}"
            cv2.putText(seg_overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            render_path = results_dir / f"rendered.png"
            seg_path = results_dir / f"segmentation.png"
            rgb_path = results_dir / f"rgb.png"
            depth_path = results_dir / f"depth.png"

            cv2.imwrite(str(render_path), render)
            cv2.imwrite(str(seg_path), seg_overlay)
            cv2.imwrite(str(rgb_path), color_bgr)
            cv2.imwrite(str(depth_path), depth_raw)

            if frame_idx % 1 == 0:
                print(
                    f"[timing] frame={frame_idx} changed={changed_count} seg_ms={seg_ms:.1f} reg_ms={reg_ms:.1f} total_ms={total_ms:.1f} fps={fps:.1f} det={len(instances)} tracked={len(tracked_object_names)}",
                    flush=True,
                )

            if ui_available:
                try:
                    cv2.imshow("segmentation", seg_overlay)
                    cv2.imshow("foundationpose", render)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        initialized = False
                        print("[demo] tracking reset requested", flush=True)
                except cv2.error as exc:
                    print(f"[demo] HighGUI failed, disabling windows: {exc}", flush=True)
                    ui_available = False

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

    finally:
        pipeline.stop()
        if ui_available:
            cv2.destroyAllWindows()

    print(f"[demo] finished. saved_frames={frame_idx}", flush=True)


if __name__ == "__main__":
    main()
