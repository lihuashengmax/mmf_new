# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation
import numpy as np


def point_sample(img_meta,
                 img_features,
                 points,#n 3
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True,
                 img_id=0,
                 out_size_factor_img=4.0):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # # apply transformation based on info in img_meta
    # points = apply_3d_transformation(
    #     points, coord_type, img_meta, reverse=True)

    # # project points to camera coordinate
    # pts_2d_with_depth = points_cam2img(points, proj_mat, with_depth=True)
    # pts_depth = pts_2d_with_depth[:, -1]
    # pts_2d = pts_2d_with_depth[:, :2]

    # valid_depth_idx = (pts_depth > 0).reshape(-1,)

    # # img transformation: scale -> crop -> flip
    # # the image is resized by img_scale_factor
    # img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    # img_coors -= img_crop_offset

    # # grid sample, the valid grid range should be in [-1,1]
    # coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    # if img_flip:
    #     # by default we take it as horizontal flip
    #     # use img_shape before padding for flip
    #     orig_h, orig_w = img_shape
    #     coor_x = orig_w - coor_x

    lidar2img_rt = torch.from_numpy(img_meta['lidar2img'][img_id]).to(points.device).to(torch.float32)#4 4
    points_4d = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)# (n, 4)
    points_2d = torch.matmul(points_4d, lidar2img_rt.transpose(-1, -2))  #(n, 4) * (4, 4) = (n, 4) 注意，必须要转置
    # points_2d[..., 2] = torch.clamp(points_2d[..., 2], min=1e-5)
    points_2d[..., :2] = points_2d[..., :2] / points_2d[..., 2:3] / out_size_factor_img

    # assert 'valid_shape' in img_meta
    if 'valid_shape' in img_meta:
        valid_shape = img_meta['valid_shape']
        valid_img_w = np.array(valid_shape[img_id, 0] / out_size_factor_img) #正对应图像深层特征 1
        valid_img_h = np.array(valid_shape[img_id, 1] / out_size_factor_img) #正对应图像深层特征 1
    else:
        valid_img_w = np.array(img_features.shape[-1])
        valid_img_h = np.array(img_features.shape[-2])

    valid_img_w = torch.from_numpy(valid_img_w).to(points_2d.device)
    valid_img_h = torch.from_numpy(valid_img_h).to(points_2d.device)

    center_xs = points_2d[..., 0]  # (n) 正对应图像深层特征 coor_x
    center_ys = points_2d[..., 1]  # (n) 正对应图像深层特征 coor_y

    valid_depth_idx = (points_2d[..., 2] > 0).reshape(-1,) #n

    valid_idx = (center_xs >= 0) & (center_xs < valid_img_w) & (center_ys >= 0) & \
                    (center_ys < valid_img_h) & valid_depth_idx  # n

    h, w = img_features.shape[-2], img_features.shape[-1]
    coor_y = (center_ys / h * 2 - 1).unsqueeze(-1)#n 1
    coor_x = (center_xs / w * 2 - 1).unsqueeze(-1)#n 1

    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    # valid_idx = valid_depth_idx & valid_y_idx & valid_x_idx

    # import cv2
    # import matplotlib.pyplot as plt
    # import numpy as np

    # cmap = plt.cm.get_cmap('hsv', 256)
    # cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    # tmp_img = cv2.imread(img_meta['filename'][img_id])
    # resize_h, resize_w, _ = img_meta['img_shape']
    # tmp_img = cv2.resize(tmp_img, (resize_w, resize_h), cv2.INTER_LINEAR)

    # pts_2d_with_depth[:, :2] *= img_scale_factor
    # pts_2d_with_depth = pts_2d_with_depth[valid_idx].cpu().numpy()
    # for i in range(pts_2d_with_depth.shape[0]):
    #     depth = pts_2d_with_depth[i, 2]
    #     m = int(640.0/depth)
    #     if m >=len(cmap):
    #         m = -1
    #     color = cmap[m,:]
    #     cv2.circle(tmp_img, (int(np.round(pts_2d_with_depth[i,0])),
    #         int(np.round(pts_2d_with_depth[i,1]))),
    #         1, color=tuple(color), thickness=-1)
    # Nuscenes
    # cv2.imwrite('nus_code/figs/%s' % img_meta['filename'][img_id].split('/')[-1], tmp_img)
    # Waymo
    # img_num = int(img_meta['filename'][img_id][-13])
    # camera_mapping = ['front', 'front_left', 'front_right', 'side_left', 'side_right']
    # cv2.imwrite('waymo_code/figs/%s_%s.png' % (img_meta['filename'][img_id].\
    #                     split('/')[-1][:-4], camera_mapping[img_num]), tmp_img)

    return point_features.squeeze().t(), valid_idx


@FUSION_LAYERS.register_module()
class MultiVoxelFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True,
                 out_size_factor_img=4):
        super(MultiVoxelFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.out_size_factor_img = out_size_factor_img
        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels * 2, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, voxel_coors, voxel_feats, \
                        img_metas, voxel_size, point_cloud_range):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features. NOTE this should be a list of list
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, voxel_coors, img_metas, \
                            voxel_size, point_cloud_range)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(voxel_feats)

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([img_pre_fuse, pts_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, voxel_coors, img_metas, \
                        voxel_size, point_cloud_range):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        num_camera = img_ins[0].shape[0] // len(img_metas)
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            voxel_coors_per_img = voxel_coors[voxel_coors[:, 0] == i]
            x = (voxel_coors_per_img[:, 3] + 0.5) * voxel_size[0] + point_cloud_range[0]
            y = (voxel_coors_per_img[:, 2] + 0.5) * voxel_size[1] + point_cloud_range[1]
            z = (voxel_coors_per_img[:, 1] + 0.5) * voxel_size[2] + point_cloud_range[2]
            x = x.unsqueeze(-1); y = y.unsqueeze(-1); z = z.unsqueeze(-1)
            decoded_voxel_coors = torch.cat([x, y, z], dim=-1)
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i * num_camera: \
                            (i + 1) * num_camera], decoded_voxel_coors, img_metas[i], level))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta, level):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, num_camera, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        # ##########这三个参数没用
        # img_scale_factor = (
        #     pts.new_tensor(img_meta['scale_factor'][:2])
        #     if 'scale_factor' in img_meta.keys() else 1)
        # img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        # img_crop_offset = (
        #     pts.new_tensor(img_meta['img_crop_offset'])
        #     if 'img_crop_offset' in img_meta.keys() else 0)
        # ##########这三个参数没用
        # proj_mat_list = get_proj_mat_by_coord_type(img_meta, self.coord_type)#这个参数没用
        num_camera = img_feats.shape[0]#有用 为6
        img_pts_list = []; valid_idx_list = []
        for camera_id in range(num_camera):
            img_pts, valid_idx = point_sample(
                img_meta=img_meta,#与img_id结合使用
                img_features=img_feats[camera_id].unsqueeze(0),#1 c h w
                points=pts,#n 3
                proj_mat=None,
                coord_type=None,
                img_scale_factor=None,
                img_crop_offset=None,
                img_flip=None,
                img_pad_shape=None,
                img_shape=None,
                aligned=self.aligned,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                img_id=camera_id,
                out_size_factor_img=self.out_size_factor_img*(level+1)#backbone与neck的下采样倍数
            )
            img_pts_list.append(img_pts)
            valid_idx_list.append(valid_idx)

        # align pts_feats from each camera
        assign_mask = img_pts.new_zeros((pts.shape[0]), dtype=torch.bool)
        final_img_pts = img_pts.new_zeros((img_pts.shape))
        for camera_id in range(num_camera):
            valid_idx = valid_idx_list[camera_id]
            assign_idx = (~assign_mask) & valid_idx
            assign_mask |= assign_idx
            final_img_pts[assign_idx] += img_pts_list[camera_id][assign_idx]
        # assert torch.sum(assign_mask) == assign_mask.shape[0], (torch.sum(assign_mask), assign_mask.shape[0])
        return final_img_pts
