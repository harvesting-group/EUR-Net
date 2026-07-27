# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import RipenessMetrics, box_iou, fitness, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class RipenessValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "ripe"
        self.metrics = RipenessMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        batch['ripeness'] = batch['ripeness'].to(self.device).float()
        if "instance_id" in batch:
            batch["instance_id"] = batch["instance_id"].to(self.device).long()
        if "group_id" in batch:
            batch["group_id"] = batch["group_id"].to(self.device).long()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        # more accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

        # [新增] 初始化成熟度 MAE 的统计容器
        self.maturity_mae_sum = 0.0
        self.maturity_mae_count = 0
        self.group_maturity = {}
        self.group_gt_exposure_count = {}
        self.uncertainty_nll_sum = 0.0
        self.uncertainty_covered_95 = 0
        self.uncertainty_interval_width_95_sum = 0.0
        self.uncertainty_abs_errors = []
        self.uncertainty_pred_stds = []


    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p = ops.non_max_suppression_ripeness(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        aux = preds[1]
        proto = aux[2] if isinstance(aux, (list, tuple)) else aux
        return p, proto

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """Prepares a batch for training or inference by processing images and targets."""
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:-2], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    @staticmethod
    def _group_id_from_file(im_file):
        """Build a stable group id by removing the trailing exposure token from the file stem."""
        stem = Path(im_file).stem
        parts = stem.rsplit("_", 1)
        group_key = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        return zlib.crc32(group_key.encode("utf-8")) & 0x7FFFFFFF

    @staticmethod
    def _instance_ids_from_bboxes(bboxes):
        """Build stable instance ids from GT boxes, sorted top-to-bottom then left-to-right."""
        n = len(bboxes)
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=bboxes.device)
        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        order = torch.argsort(centers[:, 1] * 100000.0 + centers[:, 0])
        instance_id = torch.empty(n, dtype=torch.long, device=bboxes.device)
        instance_id[order] = torch.arange(n, device=bboxes.device)
        return instance_id

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 18) % (
            "Class",
            "Images", "Instances",
            "Box(P", "R", "mAP50", "mAP50-95)",
            "Mask(P", "R", "mAP50", "mAP50-95)",
            "MAE(M)",
            "GMAE(M)",
            "GStd(M)",
            "UNLL",
            "UCorr",
            "PICP95",
            "MPIW95",
            "MeanSigma",
        )

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            gt_ripe = batch["ripeness"][batch["batch_idx"] == si] # 获取真实的成熟度
            gt_instance_id = batch.get("instance_id")
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id[batch["batch_idx"] == si].view(-1)
            gt_group_id = batch.get("group_id")
            if gt_group_id is not None:
                gt_group_id = gt_group_id[batch["batch_idx"] == si].view(-1)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            if gt_instance_id is None:
                gt_instance_id = self._instance_ids_from_bboxes(bbox)
            fallback_gid = self._group_id_from_file(batch["im_file"][si])
            if gt_group_id is None:
                gt_group_id = torch.full((nl,), fallback_gid, dtype=torch.long, device=self.device)
            image_gt_keys = {
                (int(gid.item()), int(inst_id.item())) for gid, inst_id in zip(gt_group_id, gt_instance_id)
            }
            for key in image_gt_keys:
                self.group_gt_exposure_count[key] = self.group_gt_exposure_count.get(key, 0) + 1
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            gt_masks = pbatch.pop("masks")
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            
            if npr==0:
                continue
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"], mask_matches = self._process_batch(
                    predn,
                    bbox,
                    cls,
                    pred_masks,
                    gt_masks,
                    self.args.overlap_mask,
                    masks=True,
                    return_matches=True,
                )
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

                # MAE/GMAE/GStd reuse the exact segmentation matches at Mask IoU=0.50.
                matches_at_50 = mask_matches[0]
                matched_gt_idx = matches_at_50[:, 0]
                matched_pred_idx = matches_at_50[:, 1]

                if matched_gt_idx.numel():
                    pred_ripe = predn[matched_pred_idx, -2].view(-1)
                    pred_var = predn[matched_pred_idx, -1].view(-1).clamp_min(1e-6)
                    matched_gt_ripe = gt_ripe[matched_gt_idx].view(-1)

                    abs_error = (pred_ripe - matched_gt_ripe).abs()
                    pred_std = pred_var.sqrt()
                    self.maturity_mae_sum += abs_error.sum().item()
                    self.maturity_mae_count += matched_gt_idx.numel()
                    self.uncertainty_nll_sum += (
                        0.5 * (torch.log(2 * torch.pi * pred_var) + abs_error.square() / pred_var)
                    ).sum().item()
                    self.uncertainty_covered_95 += (abs_error <= 1.96 * pred_std).sum().item()
                    self.uncertainty_interval_width_95_sum += (2 * 1.96 * pred_std).sum().item()
                    self.uncertainty_abs_errors.extend(abs_error.detach().cpu().tolist())
                    self.uncertainty_pred_stds.extend(pred_std.detach().cpu().tolist())

                    matched_instance_id = gt_instance_id[matched_gt_idx]
                    matched_group_id = gt_group_id[matched_gt_idx]
                    seen_image_instances = set()
                    for pr, gr, gid, inst_id in zip(
                        pred_ripe.detach(), matched_gt_ripe.detach(), matched_group_id, matched_instance_id
                    ):
                        key = (int(gid.item()), int(inst_id.item()))
                        if key in seen_image_instances:
                            continue
                        seen_image_instances.add(key)
                        item = self.group_maturity.setdefault(key, {"pred": [], "gt": []})
                        item["pred"].append(float(pr.item()))
                        item["gt"].append(float(gr.item()))
                # ==========================================================


                # # ==========================================================
                # # [新增] 成熟度 MSE 计算逻辑
                # # ==========================================================
                # # 1. 计算预测框和真实框的 IoU，形状为 (num_gts, num_preds)
                # iou_box = box_iou(bbox, predn[:, :4]) 
                
                # # 2. 为每个预测框找到最匹配的真实框
                # best_ious_box, best_gt_idx_box = iou_box.max(dim=0) 
                
                # # 3. 定义匹配成功条件：IoU大于0.5 且 类别预测正确
                # valid_matches_box = (best_ious_box > 0.5) & (predn[:, 5] == cls[best_gt_idx_box])
                
                # if valid_matches_box.any():
                #     # <--- 注意：这里假设 predn 的最后一列（-1）是预测的成熟度，请根据你的模型输出维度调整！
                #     # 如果你的成熟度紧跟在类别置信度后面，可能是 predn[:, 6]
                #     pred_ripe_box = predn[valid_matches_box, -1]
                    
                #     matched_gt_ripe_box = gt_ripe[best_gt_idx_box[valid_matches_box]]
                    
                #     # 计算 MSE 并累加
                #     # mse = F.mse_loss(pred_ripe.view(-1), matched_gt_ripe.view(-1), reduction='sum')
                #     mae_box =  F.l1_loss(pred_ripe_box.view(-1), matched_gt_ripe_box.view(-1), reduction='sum')
                #     self.maturity_mae_sum_box += mae_box.item()
                #     self.maturity_mae_count_box += valid_matches_box.sum().item()
                # # ==========================================================

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    # ==========================================================
    # [新增] 重载 get_stats 以返回 MSE 并写入日志
    # ==========================================================
    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = super().get_stats() # 这里会返回 mAP 等基础字典
        
        avg_mae = (
            self.maturity_mae_sum / self.maturity_mae_count if self.maturity_mae_count > 0 else float("nan")
        )
        avg_group_mae = self.group_average_mae()
        avg_group_std = self.group_prediction_std()
        uncertainty_nll, uncertainty_corr, picp95, mpiw95, mean_sigma = self.uncertainty_metrics()

        self.metrics.update_ripeness(
            avg_mae, avg_group_mae, avg_group_std, uncertainty_nll, uncertainty_corr, picp95, mpiw95, mean_sigma
        )

        # 记录到字典中，这会使其自动显示在 TensorBoard 和 results.csv 中
        stats["metrics/maturity_mae_M"] = avg_mae
        stats["metrics/maturity_gmae_M"] = avg_group_mae
        stats["metrics/maturity_gstd_M"] = avg_group_std
        stats["metrics/maturity_group_mae_M"] = avg_group_mae
        stats["metrics/maturity_group_std_M"] = avg_group_std
        stats["metrics/maturity_group_count_M"] = len(self.group_maturity)
        stats["metrics/maturity_uncertainty_nll"] = uncertainty_nll
        stats["metrics/maturity_uncertainty_corr"] = uncertainty_corr
        stats["metrics/maturity_picp95"] = picp95
        stats["metrics/maturity_mpiw95"] = mpiw95
        stats["metrics/maturity_mean_sigma"] = mean_sigma
        stats["fitness"] = fitness(self.metrics.mean_results(), mae=avg_mae, gmae=avg_group_mae, gstd=avg_group_std)
        return stats

    def uncertainty_metrics(self):
        """Return Gaussian NLL, error-uncertainty correlation, coverage, interval width, and mean sigma."""
        count = self.maturity_mae_count
        if not count:
            return (float("nan"),) * 5

        nll = self.uncertainty_nll_sum / count
        picp95 = self.uncertainty_covered_95 / count
        mpiw95 = self.uncertainty_interval_width_95_sum / count
        errors = np.asarray(self.uncertainty_abs_errors, dtype=np.float64)
        stds = np.asarray(self.uncertainty_pred_stds, dtype=np.float64)
        corr = float("nan")
        if count > 1 and errors.std() > 0 and stds.std() > 0:
            corr = float(np.corrcoef(errors, stds)[0, 1])
        mean_sigma = float(stds.mean()) if stds.size else float("nan")
        return float(nll), corr, float(picp95), float(mpiw95), mean_sigma

    def group_error_values(self):
        """Return per-group ripeness absolute errors after averaging predictions and labels in each group."""
        if not self.group_maturity:
            return []
        errors = []
        for item in self.group_maturity.values():
            if item["pred"] and item["gt"]:
                errors.append(abs(float(np.mean(item["pred"])) - float(np.mean(item["gt"]))))
        return errors

    # def group_average_mae(self):
    #     """Average MAE after grouping multi-exposure predictions by strawberry instance."""
    #     errors = self.group_error_values()
    #     return float(np.mean(errors)) if errors else float("nan")

    # # def group_prediction_std(self):
    #     """Mean prediction standard deviation for targets matched in every exposure."""
    #     if not self.group_maturity:
    #         return float("nan")
    #     stds = []
    #     for key, item in self.group_maturity.items():
    #         expected = self.group_gt_exposure_count.get(key, 0)
    #         if expected >= 2 and len(item["pred"]) == expected:
    #             stds.append(float(np.std(item["pred"])))
    #     return float(np.mean(stds)) if stds else float("nan")

    def group_average_mae(self):
        """Average MAE after grouping multi-exposure predictions by strawberry instance."""
        if not self.group_maturity:
            return float("nan")

        gmaes = []

        for key, item in self.group_maturity.items():
            expected = self.group_gt_exposure_count.get(key, 0)

            predictions = np.asarray(item["pred"], dtype=float)
            predictions = predictions[np.isfinite(predictions)]

            matched = len(predictions)

            # 至少两个曝光
            if expected >= 2 and matched >= 2:
                gt = float(np.mean(item["gt"]))
                mean_pred = float(np.mean(predictions))
                gmaes.append(abs(mean_pred - gt))

        return float(np.mean(gmaes)) if gmaes else float("nan")
    
    def group_prediction_std(self):
        """Mean prediction standard deviation for groups with at least two matched exposures."""
        if not self.group_maturity:
            return float("nan")

        stds = []

        for key, item in self.group_maturity.items():
            expected = self.group_gt_exposure_count.get(key, 0)

            predictions = np.asarray(item["pred"], dtype=float)
            predictions = predictions[np.isfinite(predictions)]

            matched = len(predictions)

            # 至少两个曝光
            if expected >= 2 and matched >= 2:
                stds.append(float(np.std(predictions, ddof=0)))

        return float(np.mean(stds)) if stds else float("nan")


    def print_results(self):
        """Prints validation set metrics per class + MSE."""
        # 1. 获取表头
        # pf = self.get_desc()  
        # LOGGER.info(pf)
        
        # 2. 计算全局平均 MSE
        avg_mae = (
            self.maturity_mae_sum / self.maturity_mae_count if self.maturity_mae_count > 0 else float("nan")
        )
        avg_group_mae = self.group_average_mae()
        avg_group_std = self.group_prediction_std()
        uncertainty_nll, uncertainty_corr, picp95, mpiw95, mean_sigma = self.uncertainty_metrics()
        self.metrics.update_ripeness(
            avg_mae, avg_group_mae, avg_group_std, uncertainty_nll, uncertainty_corr, picp95, mpiw95, mean_sigma
        )

        # 3. 获取常规的 mAP 结果
        metrics_vals = self.metrics.mean_results()
        
        # 4. 修复 nt_all 缺失问题
        # 尝试获取实例总数，如果 self.nt_all 不存在，则从 stats 中累加
        if hasattr(self, 'nt_all'):
            nt = self.nt_all
        else:
            # self.stats["target_cls"] 存储了每个 batch 的类别标签，列表长度之和即为实例总数
            nt = sum(len(x) for x in self.stats["target_cls"]) if self.stats["target_cls"] else 0

        # 5. 格式化打印
        # 注意：这里的 %11i * 2 对应的是 Images 和 Instances
        # *metrics_vals 对应 P, R, mAP 等 8 个指标
        # 最后 avg_mse 对应 MSE
        LOGGER.info(("%22s" + "%11i" * 2 + "%11.3g" * len(metrics_vals)) % (
            "all", self.seen, nt, *metrics_vals
        ))
    
    def _process_batch(
        self,
        detections,
        gt_bboxes,
        gt_cls,
        pred_masks=None,
        gt_masks=None,
        overlap=False,
        masks=False,
        return_matches=False,
    ):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor | None): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor | None): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.
            masks (bool): Flag indicating if the batch contains mask data.
            return_matches (bool): Whether to return matched label and detection indices at each IoU threshold.

        Returns:
            (torch.Tensor | tuple): Correct predictions, optionally with per-threshold matching indices.

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Example:
            ```python
            detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            gt_cls = torch.tensor([1, 0])
            correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
            ```
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou, return_matches=return_matches)
    
    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),  # not set to self.args.max_det due to slow plotting speed
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
