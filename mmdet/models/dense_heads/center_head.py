from logging import warning
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmcv.ops import batched_nms

from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from ..utils import gaussian_radius, gen_gaussian_target
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class CenterHead(BaseDenseHead):
    """Head of CenterNet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. Because
            HourglassNet-104 outputs the final feature and intermediate
            supervision feature and HourglassNet-52 only outputs the final
            feature. Default: 2.
        train_cfg (dict | None): Training config. Useless in CenterHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterHead. Default: None.
        loss_heatmap (dict | None): Config of center heatmap loss. Default:
            GaussianFocalLoss.
        loss_offset (dict | None): Config of center offset loss. Default:
            SmoothL1Loss.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        num_feat_levels=2,
        train_cfg=None,
        test_cfg=None,
        loss_heatmap=dict(
            type="GaussianFocalLoss", alpha=2.0, gamma=4.0, loss_weight=1
        ),
        loss_offset=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1),
    ):
        super(CenterHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.center_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = (
            build_loss(loss_heatmap) if loss_heatmap is not None else None
        )
        self.loss_offset = build_loss(loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CenterHead."""
        return nn.Sequential(
            ConvModule(in_channels, feat_channels, 3, padding=1),
            ConvModule(feat_channels, out_channels, 1, norm_cfg=None, act_cfg=None),
        )

    def _init_center_kpt_layers(self):
        """Initialize center keypoint layers.

        Including center heatmap branch and center offset branch.
        """
        self.center_heat = nn.ModuleList()
        self.center_off = nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.center_heat.append(
                self._make_layers(
                    out_channels=self.num_classes, in_channels=self.in_channels
                )
            )
            self.center_off.append(
                self._make_layers(
                    out_channels=self.center_offset_channels,
                    in_channels=self.in_channels,
                )
            )

    def _init_layers(self):
        """Initialize layers for CenterHead.
        """
        self._init_center_kpt_layers()

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.num_feat_levels):
            # The initialization of parameters are different between nn.Conv2d
            # and ConvModule. Our experiments show that using the original
            # initialization of nn.Conv2d increases the final mAP by about 0.2%
            self.center_heat[i][-1].conv.reset_parameters()
            self.center_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.center_off[i][-1].conv.reset_parameters()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of center heatmaps, offset heatmaps.
                - center_heats (list[Tensor]): Center heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - center_offs (list[Tensor]): Center offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  center_offset_channels.
        """
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.

        Returns:
            tuple[Tensor]: A tuple of CenterHead's output for current feature
            level. Containing the following Tensors:

                - center_heat (Tensor): Predicted center heatmap.
                - center_off (Tensor): Predicted center offset heatmap.
        """
        center_heat = self.center_heat[lvl_ind](x)
        center_off = self.center_off[lvl_ind](x)
        result_list = [center_heat, center_off]
        return result_list

    def get_targets(
        self, gt_bboxes, gt_labels, feat_shape, img_shape,
    ):
        """Generate center targets.

        Including center heatmap, center offset.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].

        Returns:
            dict: Ground truth of center heatmap, center offset.
            Containing the following keys:

                - center_heatmap (Tensor): Ground truth center heatmap.
                - center_offset (Tensor): Ground truth center offset.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)

        gt_center_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width]
        )
        gt_center_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])

        for batch_id in range(batch_size):
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                # Use coords in the feature level to generate ground truth
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                # Int coords on feature map/ground truth tensor
                # left_idx = int(min(scale_left, width - 1))
                # right_idx = int(min(scale_right, width - 1))
                # top_idx = int(min(scale_top, height - 1))
                # bottom_idx = int(min(scale_bottom, height - 1))
                center_x_idx = int(min(scale_center_x, width - 1))
                center_y_idx = int(min(scale_center_y, height - 1))

                # Generate gaussian heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius(
                    (scale_box_height, scale_box_width), min_overlap=0.3
                )
                radius = max(0, int(radius))
                gt_center_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_center_heatmap[batch_id, label],
                    [center_x_idx, center_y_idx],
                    radius,
                )

                # Generate center offset
                # left_offset = scale_left - left_idx
                # top_offset = scale_top - top_idx
                # right_offset = scale_right - right_idx
                # bottom_offset = scale_bottom - bottom_idx
                center_x_offset = scale_center_x - center_x_idx
                center_y_offset = scale_center_y - center_y_idx

                gt_center_offset[
                    batch_id, 0, center_y_idx, center_x_idx
                ] = center_x_offset
                gt_center_offset[
                    batch_id, 1, center_y_idx, center_x_idx
                ] = center_y_offset

        target_result = dict(
            center_heatmap=gt_center_heatmap, center_offset=gt_center_offset,
        )

        return target_result

    def loss(
        self,
        center_heats,
        center_offs,
        gt_bboxes,
        gt_labels,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Compute losses of the head.

        Args:
            center_heats (list[Tensor]): Center heatmaps for each level
                with shape (N, num_classes, H, W).
            center_offs (list[Tensor]): Center offsets for each level
                with shape (N, center_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Center keypoint losses of all
                  feature levels.
                - off_loss (list[Tensor]): Center offset losses of all feature
                  levels.
        """
        targets = self.get_targets(
            gt_bboxes, gt_labels, center_heats[-1].shape, img_metas[0]["pad_shape"],
        )
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        det_losses, off_losses = multi_apply(
            self.loss_single, center_heats, center_offs, mlvl_targets,
        )
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses)
        return loss_dict

    def loss_single(self, center_hmp, center_off, targets):
        """Compute losses for single level.

        Args:
            center_hmp (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            center_off (Tensor): Center offset for current level with
                shape (N, center_offset_channels, H, W).
            targets (dict): Center target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's differnet branches
            containing the following losses:

                - det_loss (Tensor): Center keypoint loss.
                - off_loss (Tensor): Center offset loss.
        """
        gt_center_hmp = targets["center_heatmap"]
        gt_center_off = targets["center_offset"]

        # Detection loss
        center_det_loss = self.loss_heatmap(
            center_hmp.sigmoid(),
            gt_center_hmp,
            avg_factor=max(1, gt_center_hmp.eq(1).sum()),
        )
        det_loss = center_det_loss

        # Offset loss
        # We only compute the offset loss at the real center position.
        # The value of real center would be 1 in heatmap ground truth.
        # The mask is computed in class agnostic mode and its shape is
        # batch * 1 * width * height.
        center_off_mask = (
            gt_center_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_center_hmp)
        )
        center_off_loss = self.loss_offset(
            center_off,
            gt_center_off,
            center_off_mask,
            avg_factor=max(1, center_off_mask.sum()),
        )

        off_loss = center_off_loss

        return det_loss, off_loss

    def get_bboxes(
        self, center_heats, center_offs, img_metas, rescale=False, with_nms=True,
    ):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heats (list[Tensor]): Center heatmaps for each level
                with shape (N, num_classes, H, W).
            center_offs (list[Tensor]): Center offsets for each level
                with shape (N, center_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert center_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heats[-1][img_id : img_id + 1, :],
                    center_offs[-1][img_id : img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms,
                )
            )

        return result_list

    def _get_bboxes_single(
        self, center_heat, center_off, img_meta, rescale=False, with_nms=True,
    ):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            center_heat (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            center_off (Tensor): Center offset for current level with
                shape (N, center_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            center_heat=center_heat.sigmoid(),
            center_off=center_off,
            img_meta=img_meta,
            k=self.test_cfg.center_topk,
            kernel=self.test_cfg.local_maximum_kernel,
        )

        if rescale:
            batch_bboxes /= batch_bboxes.new_tensor(img_meta["scale_factor"])

        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view([-1, 1])
        clses = batch_clses.view([-1, 1])

        idx = scores.argsort(dim=0, descending=True)
        bboxes = bboxes[idx].view([-1, 4])
        scores = scores[idx].view(-1)
        clses = clses[idx].view(-1)

        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = detections[:, -1] > -0.1
        detections = detections[keepinds]
        labels = clses[keepinds]

        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels, self.test_cfg)

        return detections, labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        if "nms_cfg" in cfg:
            warning.warn(
                "nms_cfg in test_cfg will be deprecated. " "Please rename it as nms"
            )
        if "nms" not in cfg:
            cfg.nms = cfg.nms_cfg

        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels, cfg.nms)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[: cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1, 1, dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernel.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.

        Returns:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode_heatmap(
        self, center_heat, center_off, img_meta=None, k=100, kernel=3, num_dets=100,
    ):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            center_heat (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            center_off (Tensor): Center offset for current level with
                shape (N, center_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            k (int): Get top k center keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            num_dets (int): Num of raw boxes before doing nms.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """
        batch, _, height, width = center_heat.size()
        inp_h, inp_w, _ = img_meta["pad_shape"]

        # perform nms on heatmaps
        center_heat = self._local_maximum(center_heat, kernel=kernel)

        center_scores, center_inds, center_clses, center_ys, center_xs = self._topk(
            center_heat, k=k
        )

        # We use repeat instead of expand here because expand is a
        # shallow-copy function. Thus it could cause unexpected testing result
        # sometimes. Using expand will decrease about 10% mAP during testing
        # compared to repeat.
        center_xs = center_xs.view(batch, k)
        center_ys = center_ys.view(batch, k)

        center_off = self._transpose_and_gather_feat(center_off, center_inds)
        center_off = center_off.view(batch, k, 2)

        center_xs = center_xs + center_off[..., 0]
        center_ys = center_ys + center_off[..., 1]

        # all possible boxes based on top k centers (ignoring class)
        center_xs *= inp_w / width
        center_ys *= inp_h / height

        x_off = img_meta["border"][2]
        y_off = img_meta["border"][0]

        center_xs -= x_off
        center_ys -= y_off

        center_xs *= center_xs.gt(0.0).type_as(center_xs)
        center_ys *= center_ys.gt(0.0).type_as(center_ys)

        BBOX_CONST = 25.0
        bboxes = torch.stack(
            (
                center_xs - BBOX_CONST,
                center_ys - BBOX_CONST,
                center_xs + BBOX_CONST,
                center_ys + BBOX_CONST,
            ),
            dim=-1,
        )

        center_scores = center_scores.view(batch, k)

        scores = center_scores  # scores for all possible boxes

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = self._gather_feat(bboxes, inds)

        clses = center_clses.contiguous().view(batch, -1, 1)
        clses = self._gather_feat(clses, inds).float()

        return bboxes, scores, clses
