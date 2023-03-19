"""
Example launch commands:

python3 generate_toy_examples.py -n all -c all --visualize -colsub 5000 -rowsub 15 -frac "0.2 0.21"
python3 generate_toy_examples.py -n "Spider_Monkey" -c "dtu106" --visualize -colsub 5000 -rowsub 15 -frac "0.2 0.21"
python3 generate_toy_examples.py -n "Dog" -c "The Pumpkin" --visualize -colsub 5000 -rowsub 15 -frac "0.2 0.21"
"""

import os
import warnings

import cv2
import argparse
import numpy as np
import plotly
import plotly.graph_objects as go
from sklearn.preprocessing import minmax_scale
from pytorch3d.io import load_obj

from utils import geo_utils, plot_utils


warnings.filterwarnings("ignore")  # suppresses torch warnings about a missing texture files alongside obj moodels


CAMERA_SRC_CFG = {
    'data_path': '../datasets/Euclidean',
    'scenes': {
        'dtu106': {
            'image_resolution': {'h': 1197, 'w': 1595},
            '3d_area_bounds': {'min': {'x': -200, 'y': -200, 'z': 400}, 'max': {'x': 200, 'y': 200, 'z': 800}}
        },
        'dtu500': {  # looks bad on images
            'image_resolution': {'h': 2035, 'w': 3061},
            '3d_area_bounds': {'min': {'x': -16, 'y': -10, 'z': -0.2}, 'max': {'x': -12, 'y': -6, 'z': 0.2}}
        },
        'Folke Filbyter': {
            'image_resolution': {'h': 1292, 'w': 1926},
            '3d_area_bounds': {'min': {'x': -0.05, 'y': -0.05, 'z': 0.05}, 'max': {'x': 0.05, 'y': 0.05, 'z': 0.3}}
        },
        'Gustav Vasa': {
            'image_resolution': {'h': 1931, 'w': 1290},
            '3d_area_bounds': {'min': {'x': -6, 'y': -1, 'z': 6}, 'max': {'x': 2, 'y': 4, 'z': 14}}
        },
        'GustavIIAdolf': {
            'image_resolution': {'h': 1265, 'w': 1878},
            '3d_area_bounds': {'min': {'x': 6, 'y': -5, 'z': -15}, 'max': {'x': 13, 'y': 10, 'z': 0}}
        },
        'The Pumpkin': {
            'image_resolution': {'h': 1293, 'w': 1933},
            '3d_area_bounds': {'min': {'x': -15, 'y': 5, 'z': -20}, 'max': {'x': 0, 'y': 20, 'z': -5}}
        },
    }
}


class ToyGenerator:
    def __init__(self, data_path):
        self.data_path = data_path

    def _get_line_points(self):
        x = np.arange(1000)
        y = x * 5
        z = np.arange(1000) * 0.3
        return np.stack([x, y, z]).T

    def _get_obj_points(self, scene_name):
        verts, _, _ = load_obj(os.path.join(self.data_path, 'points', f'{scene_name}.obj'))
        return verts.numpy()

    def generate(self, scene_name: str, cameras_src_name: str, visualize: bool = True, target_fraction: float = None,
                 column_subsample: int = None, row_subsample: int = None):
        if scene_name == 'line':
            world_points = self._get_line_points()
        else:
            world_points = self._get_obj_points(scene_name)

        scene_data = self._get_scene_data(world_points, cameras_src_name, target_fraction,
                                          column_subsample, row_subsample)

        np.savez(os.path.join(self.data_path, f'{scene_name} [CAM {cameras_src_name}]_small.npz'), **scene_data)
        if visualize:
            self.visualize_data(scene_data, scene_name, cameras_src_name)

    def _get_scene_data(self, world_points: np.ndarray, cameras_src_name: str, target_fraction: tuple = None,
                        column_subsample: int = None, row_subsample: int = None) -> dict:
        scene_data = {}
        camera_cfg = CAMERA_SRC_CFG['scenes'][cameras_src_name]

        # Copy camera matricies
        cam_data = np.load(os.path.join(CAMERA_SRC_CFG['data_path'], f'{cameras_src_name}.npz'))
        for key in ['Ps_gt', 'Ns', 'K_gt', 'R_gt', 'T_gt']:
            scene_data[key] = cam_data[key]
        scene_data['image_resolution'] = camera_cfg['image_resolution']

        # Save original 3d points
        world_points = self._cloud_to_bounds(world_points, camera_cfg['3d_area_bounds'])
        scene_data['3d_point_gt'] = world_points.T

        # Save point tracks in pixels
        uvz = scene_data['Ps_gt'] @ np.concatenate([world_points, np.ones([len(world_points), 1])], axis=1).T  # [m,3,n]
        uvz /= uvz[:, [2], :]
        uvz[:, 0, :] = uvz[:, 0, :].clip(0, camera_cfg['image_resolution']['w'] - 1)
        uvz[:, 1, :] = uvz[:, 1, :].clip(0, camera_cfg['image_resolution']['h'] - 1)
        uvz = uvz[:, :2]
        m, _, n = uvz.shape
        M = uvz.reshape(m * 2, n)

        M = geo_utils.remove_empty_tracks_cams(M)

        if column_subsample != None:
            sampled_cols = np.sort(np.random.choice(n, size=min(column_subsample, n), replace=False))
            M = M[:, sampled_cols]
        if row_subsample != None:
            sampled_rows = np.sort(np.random.choice(m, size=min(row_subsample, m), replace=False))
            M = M[np.sort(np.concatenate([sampled_rows * 2, sampled_rows * 2 + 1]))]
            for key in ['Ps_gt', 'Ns', 'K_gt', 'R_gt', 'T_gt']:
                scene_data[key] = scene_data[key][sampled_rows]

        if target_fraction is not None:
            M = geo_utils.sparsify_M(M, target_fraction)

        scene_data['M'] = M
        return scene_data

    @staticmethod
    def _cloud_to_bounds(world_points, bounds):
        world_points[:, 0] = minmax_scale(world_points[:, 0], feature_range=(bounds['min']['x'], bounds['max']['x']))
        world_points[:, 1] = minmax_scale(world_points[:, 1], feature_range=(bounds['min']['y'], bounds['max']['y']))
        world_points[:, 2] = minmax_scale(world_points[:, 2], feature_range=(bounds['min']['z'], bounds['max']['z']))
        return world_points

    def visualize_data(self, scene_data: dict, scene_name: str, cameras_src_name: str):
        self._visualize_cloud(scene_data, scene_name, cameras_src_name)
        self._visualize_images(scene_data, scene_name, cameras_src_name)

    def _visualize_cloud(self, scene_data: dict, scene_name: str, cameras_src_name: str):
        Rs_gt, ts_gt = scene_data['R_gt'], scene_data['T_gt']
        pts3D = scene_data['3d_point_gt']  # toy data special key (not available in real data)

        data = []
        data.append(plot_utils.get_3D_quiver_trace(ts_gt, Rs_gt[:, :3, 2], color='#86CE00', name='cam_gt'))
        data.append(plot_utils.get_3D_scater_trace(ts_gt.T, color='#86CE00', name='cam_gt', size=1))
        data.append(plot_utils.get_3D_scater_trace(pts3D, '#3366CC', '3D_points_gt', size=2))

        fig = go.Figure(data=data)
        fig.update_layout(title=f'{scene_name} [CAM {cameras_src_name}]', showlegend=True)

        os.makedirs(os.path.join(self.data_path, 'plots'), exist_ok=True)
        path = os.path.join(self.data_path, 'plots', f'{scene_name} [CAM {cameras_src_name}].html')
        plotly.offline.plot(fig, filename=path, auto_open=False)

    def _visualize_images(self, scene_data: dict, scene_name: str, cameras_src_name: str):
        os.makedirs(os.path.join(self.data_path, 'images', f'{scene_name} [CAM {cameras_src_name}]'), exist_ok=True)
        M = scene_data['M']
        xs = geo_utils.M_to_xs(M)
        for i, points in enumerate(xs):
            h, w = scene_data['image_resolution']['h'], scene_data['image_resolution']['w']
            image = np.ones((h, w, 3))
            valid_points = points[points.sum(axis=1) > 0].T.astype(int)
            valid_points[1] = h - valid_points[1] - 1
            image[valid_points[1], valid_points[0]] = 0.
            image *= 255
            cv2.imwrite(os.path.join(self.data_path, 'images', f'{scene_name} [CAM {cameras_src_name}]', f'{i}.jpg'),
                        image.astype(np.uint8))


if __name__ == '__main__':
    scene_choices = ['line', 'dolphin', 'American_Paint_Horse_Nuetral', 'Cat', 'Dog', 'Spider_Monkey']
    cam_choices = ['dtu106', 'Folke Filbyter', 'Gustav Vasa', 'GustavIIAdolf', 'The Pumpkin']
    # other scenes with nice camera position
    # 'dtu500', 'Jonas Ahlstromer', 'Lund University Sphinx', 'Tsar Nikolai I', 'Urban II'

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str,
                        help='Scene name, use "all" to iterate over all scenes',
                        choices=scene_choices + ['all'])
    parser.add_argument('--cam_src', '-c', type=str,
                        help='Scene name to take cameras from, use "all" to iterate over all camera sources',
                        choices=cam_choices + ['all'])
    parser.add_argument('--visualize', action='store_true', help='Add to plot the cloud')
    parser.add_argument('--target_fraction', '-frac', default=None, type=str,
                        help='Sampling range for sparsification (how much should stay) given as a space-separated '
                             'string e.g. "0.07 0.4"')
    parser.add_argument('--column_subsample', '-colsub', default=None, type=int,
                        help='Num columns (tracks) in the resulting data')
    parser.add_argument('--row_subsample', '-rowsub', default=None, type=int,
                        help='Num rows (cameras) in the resulting data')
    args = parser.parse_args()

    if args.target_fraction is not None:
        args.target_fraction = list(map(float, args.target_fraction.strip().split()))
    scene_names = scene_choices if args.name == 'all' else [args.name]
    camera_srcs = cam_choices if args.cam_src == 'all' else [args.cam_src]
    generator = ToyGenerator('../datasets/Euclidean_toy')
    for scene_name in scene_names:
        for camera_src in camera_srcs:
            print(f'Generating {scene_name} with cameras from {camera_src}...')
            generator.generate(scene_name, camera_src, visualize=args.visualize, target_fraction=args.target_fraction,
                               column_subsample=args.column_subsample, row_subsample=args.row_subsample)
