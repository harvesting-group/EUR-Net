# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors,TaskAlignedAssignerwithRipeness
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

class RipeLoss(nn.Module):
    """Criterion ripe for computing training losses during training."""

    def __init__(self):
        """Initialize the ripe module with regularization maximum and DFL settings."""
        super().__init__()

    def forward(self, gt_ripe, target_ripe, fg_mask):
        """IoU loss."""
        lamma = 1.0
        num = gt_ripe[fg_mask].numel()
        error = torch.abs(gt_ripe[fg_mask] - target_ripe[fg_mask])
        loss_iou = error.sum() * lamma / num
        return loss_iou


class GaussianNegativeLogLikelihood(nn.Module): # 修正了拼写 (Gaussion -> Gaussian, Likehood -> Likelihood)
    def __init__(self, eps=1e-5):
        super().__init__()
        # 建议直接使用 sum，然后在 forward 中安全地除以 num
        self.gnll_loss = nn.GaussianNLLLoss(reduction='sum', eps=eps)

    def forward(self, predicted_ripe, predicted_var, target_ripe, fg_mask):
        # gamma = 0.9 #0.5 imgsz=1280
        gamma = 0.1  # 可以根据需要调整 gamma 的值 #0.1 imgsz=640
        
        # 提取有效（前景）的预测和标签
        pred_mu_fg = predicted_ripe[fg_mask]
        pred_var_fg = predicted_var[fg_mask]
        target_fg = target_ripe[fg_mask]
        
        num = target_fg.numel()

        if num == 0:
            dummy_loss = (predicted_ripe.sum() + predicted_var.sum()) * 0.0
            return dummy_loss

        # 计算总 Loss (因为初始化时用了 reduction='sum')
        ripe_loss_sum = self.gnll_loss(pred_mu_fg, target_fg, pred_var_fg)
        
        # 计算平均 Loss 并乘以权重
        loss = (ripe_loss_sum / (num+1e-5)) * gamma
        
        return loss

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()



class v8RipenessLoss(v8DetectionLoss):
    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.model = model #new add
        # 正确初始化 args 属性
        if hasattr(model, 'args'):
            self.args = model.args
        
        #added for consistency loss
        self.ripe_cons = getattr(model.args, "ripe_cons", 0.05)
        self.cons_feat = getattr(model.args, "cons_feat", getattr(model.args, "feat_cons", 0.1))
        self.cons_eps = getattr(model.args, "cons_eps", 1e-6)
        self.cons_box = getattr(model.args, "cons_box", 0.2)
        self.consist_flag = getattr(model.args, "consist", False)
        self.light_single_aux = getattr(model.args, "light_single_aux", False)
        self.light_corr = getattr(model.args, "light_corr", 1.0)
        self.light_route = getattr(model.args, "light_route", 0.2)
        self.light_identity = getattr(model.args, "light_identity", 0.5)
        self.light_smooth = getattr(model.args, "light_smooth", 0.01)
        self.light_balance = getattr(model.args, "light_balance", 0.01)
        self.light_diverse = getattr(model.args, "light_diverse", 0.01)


        # self.moe_balance = getattr(model.args, "moe_balance", 0.05) #0.02 #2
        # self.moe_task = getattr(model.args, "moe_task", 0.02)
        # self.moe_entropy = getattr(model.args, "moe_entropy", 0.002)
        # self.moe_scale = getattr(model.args, "moe_scale", 0.01)
        # self.moe_view = getattr(model.args, "moe_view", 0.01)
        # self.moe_task_margin = getattr(model.args, "moe_task_margin", 0.2)

        m = model.model[-1]  
        self.overlap = model.args.overlap_mask
        self.ripe_loss = RipeLoss()
        self.nc = m.nc
        self.assigner_ripe = TaskAlignedAssignerwithRipeness(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.GaussianNLL = GaussianNegativeLogLikelihood()
        self.light_stem = next(
            (module for module in model.model.modules() if module.__class__.__name__ == "LightMoEStem"),
            None,
        )

    @staticmethod
    def _luminance(img):
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    @staticmethod
    def _gradient_mag(img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, :, :, 1:] = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy[:, :, 1:, :] = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        return (dx.square() + dy.square() + 1e-6).sqrt()

    def _build_single_route_targets(self, img):
        """Generate single-image pseudo labels for dark/over/normal routing supervision."""
        lum = self._luminance(img)
        sat = (img > 0.98).float().mean(dim=1, keepdim=True)
        detail = self._gradient_mag(img)
        detail = detail / (detail.amax(dim=(2, 3), keepdim=True) + 1e-6)

        dark_score = F.relu(0.35 - lum) * (1.0 + 0.5 * (1.0 - detail))
        over_score = F.relu(lum - 0.75) + sat
        normal_score = torch.exp(-((lum - 0.5) ** 2) / 0.05) * (1.0 - sat).clamp_min(0.0) * (0.5 + 0.5 * detail)

        target = torch.cat((dark_score, over_score, normal_score), dim=1)
        target = target + 1e-4
        return target / target.sum(dim=1, keepdim=True)

    @staticmethod
    def _charbonnier(pred, target, eps=1e-3):
        return torch.sqrt((pred - target).square() + eps**2).mean()

    def _single_image_light_losses(self, input_img):
        """Compute single-image auxiliary supervision for LightMoE experts without a multi-view teacher."""
        zero = torch.tensor(0.0, device=self.device)
        if not self.light_stem or self.light_stem.last_corrected is None:
            return [zero] * 6

        corrected = self.light_stem.last_corrected
        route_probs = self.light_stem.last_route_probs
        confidence = self.light_stem.last_confidence
        expert_feats = self.light_stem.last_expert_feats

        route_target = self._build_single_route_targets(input_img)
        normal_weight = route_target[:, 2:3]
        dark_weight = route_target[:, 0:1]
        over_weight = route_target[:, 1:2]

        corrected_l = self._luminance(corrected)
        input_grad = self._gradient_mag(input_img)
        corrected_grad = self._gradient_mag(corrected)
        corrected_sat = (corrected > 0.98).float().mean(dim=1, keepdim=True)

        # Encourage recovery in dark/over-exposed regions while preserving local structure.
        dark_penalty = (F.relu(0.35 - corrected_l) * dark_weight).mean()
        over_penalty = ((F.relu(corrected_l - 0.75) + corrected_sat) * over_weight).mean()
        corr_loss = dark_penalty + over_penalty + 0.1 * self._charbonnier(corrected_grad, input_grad)

        route_loss = (
            route_target
            * (route_target.clamp_min(1e-6).log() - route_probs.clamp_min(1e-6).log())
        ).sum(dim=1).mean()

        identity_loss = ((corrected - input_img).abs() * normal_weight).sum() / (
            normal_weight.sum() * corrected.shape[1] + 1e-6
        )

        conf_dx = torch.abs(confidence[:, :, :, 1:] - confidence[:, :, :, :-1]).mean()
        conf_dy = torch.abs(confidence[:, :, 1:, :] - confidence[:, :, :-1, :]).mean()
        route_dx = torch.abs(route_probs[:, :, :, 1:] - route_probs[:, :, :, :-1]).mean()
        route_dy = torch.abs(route_probs[:, :, 1:, :] - route_probs[:, :, :-1, :]).mean()
        smooth_loss = conf_dx + conf_dy + 0.5 * (route_dx + route_dy)

        route_mean = route_probs.flatten(2).mean(dim=-1).mean(dim=0)
        uniform = route_mean.new_full(route_mean.shape, 1.0 / route_mean.numel())
        balance_loss = F.mse_loss(route_mean, uniform)

        diverse_loss = zero
        if expert_feats is not None and expert_feats.numel():
            pooled = expert_feats.flatten(3).mean(dim=-1)  # [B, E, C]
            pooled = F.normalize(pooled, dim=-1)
            pair_losses = []
            for i in range(pooled.shape[1]):
                for j in range(i + 1, pooled.shape[1]):
                    pair_losses.append((pooled[:, i] * pooled[:, j]).sum(dim=-1).abs().mean())
            if pair_losses:
                diverse_loss = torch.stack(pair_losses).mean()

        return corr_loss, route_loss, identity_loss, smooth_loss, balance_loss, diverse_loss

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""

        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            # out = torch.zeros(batch_size, counts.max(), 5, device=self.device) #5 =cls, cx, cy, width, height
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device) #5 =cls, cx, cy, width, height

            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:] # targets =[image_id, cls, bbox]
            out[..., 2:6] = xywh2xyxy(out[..., 2:6].mul_(scale_tensor))
        return out
    
    @staticmethod
    def _unpack_ripe_preds(pred):
        if isinstance(pred[0], list):
            if len(pred) == 6:
                return pred
            feats, pred_masks, proto, pred_ripe, pred_var = pred
            return feats, pred_masks, proto, pred_ripe, pred_var, None

        aux = pred[1]
        if len(aux) == 6:
            return aux
        feats, pred_masks, proto, pred_ripe, pred_var = aux
        return feats, pred_masks, proto, pred_ripe, pred_var, None

    @staticmethod
    def _js_divergence(p, q, eps=1e-6):
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        m = 0.5 * (p + q)
        return 0.5 * ((p * (p.log() - m.log())).sum(dim=-1) + (q * (q.log() - m.log())).sum(dim=-1))

    def _compute_single_view_loss_bundle(self, preds, batch, include_moe_aux=True):
        """Compute the standard ripe task losses for one RGB view."""
        loss = torch.zeros(5, device=self.device)
        # aux_loss = torch.zeros(3, device=self.device)
        current_epoch = batch.get("epoch", 0)
        feats, pred_masks, proto, pred_ripe, pred_var, moe_meta = self._unpack_ripe_preds(preds)
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()
        pred_ripe = pred_ripe.permute(0, 2, 1).contiguous()
        pred_var = pred_var.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["ripeness"], batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_ripe, gt_bboxes = targets.split((1, 1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help."
            ) from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_ripeness, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner_ripe(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_ripe,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )
            if current_epoch < 70:
                loss[4] = self.ripe_loss(target_ripeness, pred_ripe, fg_mask)
            else:
                loss[4] = self.GaussianNLL(pred_ripe, pred_var, target_ripeness, fg_mask)
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= self.hyp.ripe

        return loss, batch_size


    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        if self.light_single_aux and self.light_stem is not None:
            task_loss,  batch_size = self._compute_single_view_loss_bundle(preds, batch, include_moe_aux=False)
            input_img = batch["img"].to(self.device)
            corr_loss, route_loss, identity_loss, smooth_loss, balance_loss, diverse_loss = self._single_image_light_losses(
                input_img
            )

            light_aux = torch.stack(
                (
                    self.light_corr * corr_loss,
                    self.light_route * route_loss,
                    self.light_identity * identity_loss,
                    self.light_smooth * smooth_loss,
                    self.light_balance * balance_loss,
                    self.light_diverse * diverse_loss,
                )
            )
            total = task_loss.sum() + light_aux.sum()
            return total * batch_size, torch.cat((task_loss, light_aux)).detach()

        else:
            # loss, aux_loss, batch_size = self._compute_single_view_loss_bundle(preds, batch, include_moe_aux=True)
            # total = loss.sum() + aux_loss.sum()
            # return total * batch_size, torch.cat((loss, aux_loss)).detach()
            loss, batch_size = self._compute_single_view_loss_bundle(preds, batch, include_moe_aux=True)
            total = loss.sum() 
            return total * batch_size, loss.detach()


    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()
    
    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()




# class v8RipenessLoss(v8DetectionLoss):
#     def __init__(self, model):  # model must be de-paralleled
#         """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
#         super().__init__(model)
#         m = model.model[-1]  
#         self.overlap = model.args.overlap_mask
#         self.ripe_loss = RipeLoss()
#         self.nc = m.nc
#         self.assigner_ripe = TaskAlignedAssignerwithRipeness(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

#     def preprocess(self, targets, batch_size, scale_tensor):
#         """Preprocesses the target counts and matches with the input batch size to output a tensor."""

#         if targets.shape[0] == 0:
#             out = torch.zeros(batch_size, 0, 6, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             counts = counts.to(dtype=torch.int32)
#             # out = torch.zeros(batch_size, counts.max(), 5, device=self.device) #5 =cls, cx, cy, width, height
#             out = torch.zeros(batch_size, counts.max(), 6, device=self.device) #5 =cls, cx, cy, width, height

#             for j in range(batch_size):
#                 matches = i == j
#                 n = matches.sum()
#                 if n:
#                     out[j, :n] = targets[matches, 1:] # targets =[image_id, cls, bbox]
#             out[..., 2:6] = xywh2xyxy(out[..., 2:6].mul_(scale_tensor))
#         return out

#     def __call__(self, preds, batch):
#         """Calculate and return the loss for the YOLO model."""
#         loss = torch.zeros(5, device=self.device)  # box, cls, dfl ripe 
#         feats, pred_masks, proto, pred_ripe = preds if isinstance(preds[0], list) else preds[1]
#         batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#            (self.reg_max * 4, self.nc), 1)
        
#         # B, grids, ..
#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()
#         pred_masks = pred_masks.permute(0, 2, 1).contiguous()
#         pred_ripe = pred_ripe.permute(0, 2, 1).contiguous()
        
       
#         dtype = pred_scores.dtype
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # Targets
#         try:
#             # print(batch['cls'].view(-1, 1), batch['ripeness'], batch['bboxes'])
#             batch_idx = batch['batch_idx'].view(-1, 1)
#             targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['ripeness'], batch['bboxes']), 1)
#             targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#             gt_labels, gt_ripe, gt_bboxes = targets.split((1, 1, 4), 2)  # cls, xyxy

#             mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
#         except RuntimeError as e:
#             raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
#                             "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
#                             "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
#                             "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
#                             'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

#         _, target_ripeness, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner_ripe(
#             pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor, gt_labels, gt_ripe, gt_bboxes, mask_gt)
        
#         target_scores_sum = max(target_scores.sum(), 1)

#         # Cls loss
#         # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
#         loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         if fg_mask.sum():
#             # Bbox loss
#             loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
#                                               target_scores, target_scores_sum, fg_mask)
#             # Masks loss
#             masks = batch['masks'].to(self.device).float()
#             if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
#                 masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

#             loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
#                                                        pred_masks, imgsz, self.overlap)
            
#             # Ripeness loss
#             loss[4] = self.ripe_loss(target_ripeness, pred_ripe, fg_mask)

#         # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
#         else:
#             loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

#         loss[0] *= self.hyp.box  # box gain
#         loss[1] *= self.hyp.box  # seg gain
#         loss[2] *= self.hyp.cls  # cls gain
#         loss[3] *= self.hyp.dfl  # dfl gain
#         loss[4] *= self.hyp.ripe

#         return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
#     @staticmethod
#     def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
#                          area: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the instance segmentation loss for a single image.

#         Args:
#             gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
#             pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
#             proto (torch.Tensor): Prototype masks of shape (32, H, W).
#             xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
#             area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

#         Returns:
#             (torch.Tensor): The calculated mask loss for a single image.

#         Notes:
#             The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
#             predicted masks from the prototype masks and predicted mask coefficients.
#         """
#         pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
#         loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
#         return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()
    
#     def calculate_segmentation_loss(
#         self,
#         fg_mask: torch.Tensor,
#         masks: torch.Tensor,
#         target_gt_idx: torch.Tensor,
#         target_bboxes: torch.Tensor,
#         batch_idx: torch.Tensor,
#         proto: torch.Tensor,
#         pred_masks: torch.Tensor,
#         imgsz: torch.Tensor,
#         overlap: bool,
#     ) -> torch.Tensor:
#         """
#         Calculate the loss for instance segmentation.

#         Args:
#             fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
#             masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
#             target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
#             target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
#             batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
#             proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
#             pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
#             imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
#             overlap (bool): Whether the masks in `masks` tensor overlap.

#         Returns:
#             (torch.Tensor): The calculated loss for instance segmentation.

#         Notes:
#             The batch loss can be computed for improved speed at higher memory usage.
#             For example, pred_mask can be computed as follows:
#                 pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
#         """
#         _, _, mask_h, mask_w = proto.shape
#         loss = 0

#         # Normalize to 0-1
#         target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

#         # Areas of target bboxes
#         marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

#         # Normalize to mask size
#         mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

#         for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
#             fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
#             if fg_mask_i.any():
#                 mask_idx = target_gt_idx_i[fg_mask_i]
#                 if overlap:
#                     gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
#                     gt_mask = gt_mask.float()
#                 else:
#                     gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

#                 loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
#                                               marea_i[fg_mask_i])

#             # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
#             else:
#                 loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

#         return loss / fg_mask.sum()



class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
