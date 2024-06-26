# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle
from util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding

import models
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras
import pdb

logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, DIFFUSER: Dict, DENOISER: Dict):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.pose_encoding_type = pose_encoding_type

        self.image_feature_extractor = instantiate(IMAGE_FEATURE_EXTRACTOR, _recursive_=False)
        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (270x524606 and 702x512)
        # resolve the above error by introducing a nn.Linear layer before the image_feature_extractor and self.diffuser
        # self.link_to_diffuser = nn.Linear(270, 702)

        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        # attempt to freeze pointnet and linear to conserve memory once trained enough
        # for param in self.image_feature_extractor.parameters():
        #     param.requires_grad = False

        # for param in self.diffuser.model._first.parameters():
        #     param.requires_grad = False

        self.target_dim = denoiser.target_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # set to identity! let's see!
        # if the layer is 2 dimensional
        if isinstance(m, nn.Linear):
            nn.init.eye_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # for convolutional layer, apply dirac delta function and tensors with 3 to 5 dimensions only
        elif isinstance(m, nn.Conv2d):
            nn.init.dirac_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.dirac_(m.weight)
            
        # if isinstance(m, nn.Linear):
        #     nn.init.trunc_normal_(m.weight, std=0.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def diffuser_wrapper(self,pose_encoding, z):
        return self.diffuser(pose_encoding, z=z)
    
    def forward(
        self,
        image: torch.Tensor,
        init_poses = None,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
        training=True,
        test_init=False,
        batch_repeat=-1,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """
        shapelist = list(image.shape)
        batch_num = shapelist[0]
        frame_num = shapelist[1]

        reshaped_image = image.reshape(batch_num * frame_num, *shapelist[2:])
        z = self.image_feature_extractor(reshaped_image).reshape(batch_num, frame_num, -1)
        if training:
            pose_encoding = camera_to_pose_encoding(gt_cameras, pose_encoding_type=self.pose_encoding_type)

            if batch_repeat > 0:
                pose_encoding = pose_encoding.reshape(batch_num * batch_repeat, -1, self.target_dim)
                z = z.repeat(batch_repeat, 1, 1)
            else:
                pose_encoding = pose_encoding.reshape(batch_num, -1, self.target_dim)

            # pose_encoding_tf = self.link_to_diffuser(pose_encoding)
            # diffusion_results = torch.utils.checkpoint.checkpoint(self.diffuser_wrapper, pose_encoding, z)
            diffusion_results = self.diffuser(pose_encoding, z=z)
            diffusion_results["pred_cameras"] = pose_encoding_to_camera(
                diffusion_results["x_0_pred"], pose_encoding_type=self.pose_encoding_type
            )

            return diffusion_results
        elif test_init:
            B, N, _ = z.shape

            target_shape = [B, N, 1]

            # sampling
            (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                shape=target_shape, init_poses=init_poses,  z=z, cond_fn=cond_fn, cond_start_step=cond_start_step
            )

            # write the tensor to a csv file
            # transform to 2d
            pose_encoding_to_write = pose_encoding.squeeze(0)
            pose_encoding_to_write = pose_encoding_to_write.cpu().detach().numpy()
            np.savetxt("pose_encoding.csv", pose_encoding_to_write, delimiter=" ")

            # convert the encoded representation to PyTorch3D cameras
            
            pred_cameras = pose_encoding_to_camera(pose_encoding, pose_encoding_type=self.pose_encoding_type)

            diffusion_results = {"pred_cameras": pred_cameras, "z": z}

            return diffusion_results
        else:
            B, N, _ = z.shape
            target_shape = [B, N, self.target_dim]

            # sampling
            (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                shape=target_shape, z=z, init_poses=init_poses,  cond_fn=cond_fn, cond_start_step=cond_start_step
            )

            # convert the encoded representation to PyTorch3D cameras
            pred_cameras = pose_encoding_to_camera(pose_encoding, pose_encoding_type=self.pose_encoding_type)

            diffusion_results = {"pred_cameras": pred_cameras, "z": z}

            return diffusion_results
