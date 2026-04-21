# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Run FoundationPose on a custom RGB-D sequence and mesh.

Masking is only required for the registration frame (initial pose). Subsequent
frames use track_one (RGB + depth + previous pose); per-frame masks are not used.

Pixel-space arguments (--intrinsics/--cam_k_file, --mask_bbox, --mask_path at native
resolution) refer to full-resolution frames; `--shorter_side` only affects internal
resize and scales intrinsics/masks accordingly so you need not adjust other CLI values.

Use `--mesh_scale` when the mesh file units differ from depth (meters), e.g. 0.001 for mm models.
"""

import argparse
from pathlib import Path


def build_parser():
  parser = argparse.ArgumentParser(
    description='FoundationPose on custom RGB-D folders + mesh. '
    'Mask or bbox is only needed for the registration frame.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    '--mesh_file',
    type=str,
    required=True,
    help='Path to object mesh (trimesh-supported: e.g. .obj, .ply, .glb).',
  )
  parser.add_argument(
    '--mesh_scale',
    type=float,
    default=1.0,
    help='Multiply all mesh vertex coordinates by this factor so the model matches depth units '
    '(meters after --depth_scale). Use 0.001 if the file is in millimeters. Default 1 = meters.',
  )

  parser.add_argument(
    '--rgb_dir',
    type=str,
    required=True,
    help='Directory of RGB frames, sorted by filename (*.png preferred, then *.jpg/*.jpeg).',
  )
  parser.add_argument(
    '--depth_dir',
    type=str,
    required=True,
    help='Directory of depth PNGs; same basename as each RGB frame (millimeters uint16 unless --depth_scale says otherwise).',
  )

  parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='Output root: writes ob_in_cam/*.txt poses, optional track_vis/, debug meshes. Nothing is deleted.',
  )

  intr = parser.add_mutually_exclusive_group(required=True)
  intr.add_argument(
    '--cam_k_file',
    type=str,
    help='Path to 3x3 camera matrix for native/full-resolution RGB (np.loadtxt). '
    'Scaled internally when --shorter_side is set.',
  )
  intr.add_argument(
    '--intrinsics',
    type=float,
    nargs=4,
    metavar=('FX', 'FY', 'CX', 'CY'),
    help='Pinhole intrinsics (zero skew) for native/full-resolution RGB: fx fy cx cy (pixels). '
    'Scaled internally when --shorter_side is set.',
  )

  mask = parser.add_mutually_exclusive_group(required=True)
  mask.add_argument(
    '--mask_path',
    type=str,
    help='Binary/label mask image for the registration frame at native RGB resolution '
    '(resized internally to match --shorter_side processing).',
  )
  mask.add_argument(
    '--mask_bbox',
    type=int,
    nargs=4,
    metavar=('X1', 'Y1', 'X2', 'Y2'),
    help='Inclusive axis-aligned box on the registration frame in native/full-resolution pixels '
    '(same coordinates as original RGB). Scaled internally when --shorter_side is set.',
  )

  parser.add_argument(
    '--register_frame',
    type=int,
    default=0,
    help='Index into sorted RGB list for registration (mask/bbox applies to this frame). Frames before this are skipped.',
  )

  parser.add_argument(
    '--skip',
    type=int,
    default=1,
    help='Stride along the sorted list after register_frame: 1 = every frame, 2 = every other, etc.',
  )

  parser.add_argument(
    '--depth_scale',
    type=float,
    default=1000.0,
    help='Divide raw depth values by this to get meters (1000 = depth stored as millimeters in PNG).',
  )
  parser.add_argument(
    '--zfar',
    type=float,
    default=float('inf'),
    help='Set depth pixels with z >= this (meters) to zero (infinite = no far clip).',
  )
  parser.add_argument(
    '--shorter_side',
    type=float,
    default=None,
    help='Resize RGB/depth so min(height,width) equals this (pixels); scales intrinsics and '
    '--mask_bbox to match. E.g. 720 on 1080p input to reduce GPU memory. Omit for native resolution.',
  )

  parser.add_argument(
    '--est_refine_iter',
    type=int,
    default=5,
    help='Iterations for the initial register() refinement on the registration frame.',
  )
  parser.add_argument(
    '--track_refine_iter',
    type=int,
    default=2,
    help='Iterations for track_one() on each subsequent frame.',
  )
  parser.add_argument(
    '--debug',
    type=int,
    default=1,
    help='0: silent; 1: live overlay; 2: also save track_vis/*.png; 3: also dump model_tf.obj and scene_complete.ply on registration.',
  )
  parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='Random seed (passed to set_seed for reproducibility).',
  )

  return parser


def load_cam_k_from_txt(path):
  import numpy as np

  K = np.loadtxt(path).reshape(3, 3)
  return K.astype(np.float64)


def cam_k_from_pinhole(fx, fy, cx, cy):
  import numpy as np

  return np.array(
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    dtype=np.float64,
  )


def sorted_rgb_paths(rgb_dir: Path):
  paths = sorted(rgb_dir.glob('*.png'))
  if len(paths) == 0:
    paths = sorted(rgb_dir.glob('*.jpg'))
  if len(paths) == 0:
    paths = sorted(rgb_dir.glob('*.jpeg'))
  return paths


def depth_path_for_rgb(rgb_path: Path, depth_dir: Path) -> Path:
  stem = rgb_path.stem
  p = depth_dir / f'{stem}.png'
  if not p.is_file():
    raise FileNotFoundError(f'Expected depth PNG {p} for RGB {rgb_path}')
  return p


def load_rgb(path, target_wh):
  import cv2
  import imageio

  color = imageio.imread(path)[..., :3]
  if target_wh is not None:
    tw, th = target_wh
    color = cv2.resize(color, (tw, th), interpolation=cv2.INTER_NEAREST)
  return color


def load_depth(path: Path, target_wh, depth_scale, zfar):
  import cv2
  import numpy as np

  if path.suffix.lower() != '.png':
    raise ValueError(f'Depth must be a PNG file; got {path}')
  depth = cv2.imread(str(path), -1)
  if depth is None:
    raise FileNotFoundError(path)
  depth = depth.astype(np.float64) / depth_scale
  if target_wh is not None:
    tw, th = target_wh
    depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)
  depth[(depth < 0.001) | (depth >= zfar)] = 0
  return depth


def load_binary_mask(path, target_wh):
  import cv2

  mask = cv2.imread(str(path), -1)
  if mask is None:
    raise FileNotFoundError(path)
  if len(mask.shape) == 3:
    for c in range(mask.shape[2]):
      if mask[..., c].sum() > 0:
        mask = mask[..., c]
        break
  mask = cv2.resize(mask, target_wh, interpolation=cv2.INTER_NEAREST)
  return mask.astype(bool)


def mask_from_bbox(width, height, x1, y1, x2, y2):
  """Axis-aligned box; indices are inclusive (integer pixel coordinates)."""
  import numpy as np

  x1, x2 = sorted((int(x1), int(x2)))
  y1, y2 = sorted((int(y1), int(y2)))
  x1 = np.clip(x1, 0, width - 1)
  x2 = np.clip(x2, 0, width - 1)
  y1 = np.clip(y1, 0, height - 1)
  y2 = np.clip(y2, 0, height - 1)
  m = np.zeros((height, width), dtype=bool)
  m[y1 : y2 + 1, x1 : x2 + 1] = True
  return m


def apply_scale_K(K, downscale):
  import numpy as np

  K = np.asarray(K).copy()
  K[:2, :] *= downscale
  return K


def run(args):
  import estimater as estimater_mod

  np = estimater_mod.np
  cv2 = estimater_mod.cv2
  imageio = estimater_mod.imageio
  trimesh = estimater_mod.trimesh
  o3d = estimater_mod.o3d
  dr = estimater_mod.dr
  logging = estimater_mod.logging
  set_logging_format = estimater_mod.set_logging_format
  set_seed = estimater_mod.set_seed
  ScorePredictor = estimater_mod.ScorePredictor
  PoseRefinePredictor = estimater_mod.PoseRefinePredictor
  FoundationPose = estimater_mod.FoundationPose
  depth2xyzmap = estimater_mod.depth2xyzmap
  toOpen3dCloud = estimater_mod.toOpen3dCloud
  draw_posed_3d_box = estimater_mod.draw_posed_3d_box
  draw_xyz_axis = estimater_mod.draw_xyz_axis

  rgb_dir = Path(args.rgb_dir)
  depth_dir = Path(args.depth_dir)
  output_dir = Path(args.output_dir)

  if args.skip < 1:
    raise ValueError('--skip must be >= 1')
  if args.mesh_scale <= 0:
    raise ValueError('--mesh_scale must be positive')

  set_logging_format()
  set_seed(args.seed)

  rgb_files = sorted_rgb_paths(rgb_dir)
  if len(rgb_files) == 0:
    raise RuntimeError(f'No RGB images found in {rgb_dir}')

  if args.cam_k_file:
    K = load_cam_k_from_txt(args.cam_k_file)
  else:
    fx, fy, cx, cy = args.intrinsics
    K = cam_k_from_pinhole(fx, fy, cx, cy)

  loaded = trimesh.load(args.mesh_file, process=False)
  if isinstance(loaded, trimesh.Scene):
    if len(loaded.geometry) == 0:
      raise RuntimeError(f'No geometry in mesh file: {args.mesh_file}')
    mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
  else:
    mesh = loaded
  if not isinstance(mesh, trimesh.Trimesh):
    raise RuntimeError(f'Expected a triangle mesh; got {type(mesh)}')

  mesh.apply_scale(args.mesh_scale)

  debug = args.debug
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'track_vis').mkdir(parents=True, exist_ok=True)
  (output_dir / 'ob_in_cam').mkdir(parents=True, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  pose_estimator = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=str(output_dir),
    debug=debug,
    glctx=glctx,
  )
  logging.info('estimator initialization done')

  reg_i = int(args.register_frame)
  if reg_i < 0 or reg_i >= len(rgb_files):
    raise ValueError(f'register_frame {reg_i} out of range for {len(rgb_files)} RGB files')

  first_rgb_path = rgb_files[reg_i]
  color0 = imageio.imread(first_rgb_path)[..., :3]
  H0, W0 = color0.shape[:2]

  downscale = 1.0
  if args.shorter_side is not None:
    downscale = args.shorter_side / min(H0, W0)

  H = int(H0 * downscale)
  W = int(W0 * downscale)
  target_wh = (W, H)
  K_scaled = apply_scale_K(K, downscale)

  if args.mask_path:
    ob_mask_reg = load_binary_mask(Path(args.mask_path), target_wh).astype(np.uint8)
  else:
    x1, y1, x2, y2 = args.mask_bbox
    sx = W / float(W0)
    sy = H / float(H0)
    ob_mask_reg = mask_from_bbox(
      W,
      H,
      int(round(x1 * sx)),
      int(round(y1 * sy)),
      int(round(x2 * sx)),
      int(round(y2 * sy)),
    ).astype(np.uint8)

  indices = list(range(reg_i, len(rgb_files), args.skip))
  if len(indices) == 0:
    raise RuntimeError('No frames to process after register_frame.')

  for j, i in enumerate(indices):
    rgb_path = rgb_files[i]
    id_str = rgb_path.stem
    logging.info(f'frame {j}/{len(indices)} (index {i}) id:{id_str}')

    depth_path = depth_path_for_rgb(rgb_path, depth_dir)
    color = load_rgb(rgb_path, target_wh)
    depth = load_depth(depth_path, target_wh, args.depth_scale, args.zfar)

    if j == 0:
      pose = pose_estimator.register(K=K_scaled, rgb=color, depth=depth, ob_mask=ob_mask_reg, iteration=args.est_refine_iter)

      if debug >= 3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(str(output_dir / 'model_tf.obj'))
        xyz_map = depth2xyzmap(depth, K_scaled)
        valid = depth >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(str(output_dir / 'scene_complete.ply'), pcd)
    else:
      pose = pose_estimator.track_one(rgb=color, depth=depth, K=K_scaled, iteration=args.track_refine_iter)

    pose_path = output_dir / 'ob_in_cam' / f'{id_str}.txt'
    np.savetxt(pose_path, pose.reshape(4, 4))

    if debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(K_scaled, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K_scaled, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[..., ::-1])
      cv2.waitKey(1)

    if debug >= 2:
      imageio.imwrite(output_dir / 'track_vis' / f'{id_str}.png', vis)


if __name__ == '__main__':
  run(build_parser().parse_args())
