# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS, ROTATED_LOSSES
from .rotate_standard_roi_head import RotatedStandardRoIHead
import torch.nn.functional as F
import torch.distributed as dist

def is_dist():
    return dist.is_available() and dist.is_initialized()

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if is_dist():
        dist.all_reduce(tensor, op=op)
    return tensor

@ROTATED_HEADS.register_module()
class CPD_OrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc',
                 distill_start_training_epoch = 4,
                 loss_distill=None):
        
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        
        self.start_distill_train = distill_start_training_epoch

        self.kappa = 10.0
        self.proto_momentum = 0.9


        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            version=version)
        
        out_channels = bbox_roi_extractor.out_channels
        output_size = bbox_roi_extractor.roi_layer.out_size
        self.roi_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=output_size, stride=1, padding=0),
        )

        self.gt_mlp = nn.Linear(out_channels, out_channels)
        self.roi_mlp = nn.Linear(out_channels, out_channels)

        num_classes = bbox_head.num_classes
        self.register_buffer('dc', torch.zeros(num_classes, out_channels))

        if loss_distill is not None:
            self.loss_distill = ROTATED_LOSSES.build(loss_distill)
        else:
            self.loss_distill = None   

    def init_weights(self):
        super().init_weights()

        for layer in self.roi_proj:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for mlp in [self.gt_mlp, self.roi_mlp]:
            nn.init.xavier_uniform_(mlp.weight)
            nn.init.constant_(mlp.bias, 0)

        if self.loss_distill is not None and hasattr(self.loss_distill, 'init_weights'):
            self.loss_distill.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self, 
                      epoch,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(epoch, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            losses.update(bbox_results['loss_distill'])
        return losses

    def distill_and_update(self, x, gt_rois, gt_labels) -> None:
        """Prototype Generation and EMA Update."""
        if len(gt_rois) == 0:
            return

        bbox_roi_extractor = self.bbox_roi_extractor
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], gt_rois)
        
        compress_feats = self.roi_proj(bbox_feats).view(bbox_feats.size(0), -1)
        z_i = self.gt_mlp(compress_feats)
        z = F.normalize(z_i, p=2, dim=1, eps=1e-8)

        num_classes = self.dc.shape[0]
        device = z.device

        dc_curr = self.dc.detach()
        dc_norm = F.normalize(dc_curr, p=2, dim=1, eps=1e-8)

        sim = self.kappa * torch.matmul(z, dc_norm.t())

        mask = torch.zeros(z.size(0), num_classes, device=device)
        mask[torch.arange(z.size(0)), gt_labels] = 1.0

        sim_for_max = sim.clone()
        safe_min = torch.finfo(sim.dtype).min
        sim_for_max[mask == 0] = safe_min
        
        max_val, _ = torch.max(sim_for_max, dim=0, keepdim=True)
        max_val = torch.clamp(max_val, min=0.0)

        # A_{i,c}
        exp_sim = torch.exp(sim - max_val) * mask
        denom = exp_sim.sum(dim=0, keepdim=True) + 1e-6
        exp_sim = exp_sim / denom
        d_star = torch.matmul(exp_sim.t(), z_i)

        if is_dist():
            all_reduce(d_star, op='sum')
            all_reduce(denom, op='sum')

        with torch.no_grad():
            valid_mask = (denom.squeeze(0) > 1e-5).float().unsqueeze(1)
            self.dc.copy_(
                (self.proto_momentum * self.dc + (1 - self.proto_momentum) * d_star) * valid_mask +
                self.dc * (1 - valid_mask)
            )


    def _bbox_forward_train(self, epoch, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            epoch (int): Current epoch.
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        
        gt_rois = rbbox2roi(gt_bboxes)
        gt_labels_ = torch.concat(gt_labels, dim = 0)
        
        self.distill_and_update(x, gt_rois, gt_labels_)
        if epoch >= self.start_distill_train and self.loss_distill is not None:
            labels = torch.concat([res.pos_gt_labels for res in sampling_results], dim = 0)

            pos_mask = torch.cat([
                torch.cat([
                    torch.ones(res.pos_gt_bboxes.shape[0] if res.num_gts != 0 else 0, dtype=torch.bool, device=rois.device),
                    torch.zeros(res.neg_bboxes.shape[0], dtype=torch.bool, device=rois.device)
                ]) for res in sampling_results
            ])
            
            if pos_mask.any():
                pos_bbox_feats = bbox_results['bbox_feats'][pos_mask]
                pos_compress_feats = self.roi_proj(pos_bbox_feats).flatten(1)
                pos_compress_feats = self.roi_mlp(pos_compress_feats)
                distill_loss = self.loss_distill(pos_compress_feats, self.dc, labels)
                bbox_results.update(loss_distill={'loss_distill':distill_loss})

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
