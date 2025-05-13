import copy

import numpy as np

import tensorflow as tf

from deodr import (
    Camera,
    Scene3D,
)
from deodr.triangulated_mesh import TriMesh

class Render:
    def __init__(self, max_depth=0, depth_scale=1):
        self.mesh = None
        self.depthScale = depth_scale
        self.scene = Scene3D(0)
        self.set_max_depth(max_depth)
		
    def set_mesh(self, vertices, faces):
        self.mesh = TriMesh(
            faces, vertices
        )  # we do a copy to avoid negative stride not support by Tensorflow
		
        object_center = vertices.mean(axis=0)
        object_radius = np.max(np.std(vertices, axis=0))
        self.camera_center = object_center + np.array([-0.5, 0, 5]) * object_radius
		
        self.scene.set_mesh(self.mesh)

    def set_max_depth(self, max_depth):
        self.scene.max_depth = max_depth
        self.scene.set_background_color([max_depth])

    def set_depth_scale(self, depth_scale):
        self.depthScale = depth_scale

    def set_image(self, width, height, intrinsic, extrinsic):
        self.width = width
        self.height = height
		
        # focal = None
        # if focal is None:
            # focal = 2 * self.width

        # rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # trans = -rot.T.dot(self.camera_center)
        # intrinsic = np.array(
            # [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]]
        # )
        # extrinsic = np.column_stack((rot, trans))
		
        self.camera = Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
            distortion=None,
        )
        
    def render(self):
        depth_scale = 1 * self.depthScale
        depth = self.scene.render_depth(
            self.camera,
            depth_scale=depth_scale,
        )
        #depth = tf.clip_by_value(depth, 0, self.scene.max_depth)
        return depth
