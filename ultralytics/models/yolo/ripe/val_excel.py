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
        # 固定使用训练结束自动验证所采用的 Mask 处理方式。
        # Excel 和散点图由验证器内部直接保存，不需要开启 save_txt/save，
        # 避免 save_txt=True 切换到 process_mask_native 后改变 Mask 指标。
        self.process = ops.process_mask
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

        # 保存与 MAE 完全相同的 Mask IoU=0.50 匹配结果。
        self.maturity_gt_count = 0
        self.match_records = []
        self._match_results_exported = False


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
    def _group_key_and_light_id(im_file):
        """Return the multi-exposure group key and trailing numeric exposure/view id."""
        stem = Path(im_file).stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return stem, -1

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
            self.maturity_gt_count += nl
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

                    signed_error = pred_ripe - matched_gt_ripe
                    abs_error = signed_error.abs()
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

                    # 保存逐实例匹配记录。这里直接复用 Mask IoU=0.50 的匹配索引，
                    # 因而 Excel 中的 MAE 与终端输出 MAE 使用完全相同的样本。
                    image_path = Path(batch["im_file"][si])
                    group_key, light_id = self._group_key_and_light_id(image_path)
                    matched_gt_cls = cls.view(-1)[matched_gt_idx]
                    matched_pred_cls = predn[matched_pred_idx, 5].view(-1)
                    matched_conf = predn[matched_pred_idx, 4].view(-1)
                    instance_nll = 0.5 * (
                        torch.log(2 * torch.pi * pred_var) + signed_error.square() / pred_var
                    )
                    lower_95 = pred_ripe - 1.96 * pred_std
                    upper_95 = pred_ripe + 1.96 * pred_std
                    covered_95 = abs_error <= 1.96 * pred_std

                    for row_idx in range(matched_gt_idx.numel()):
                        self.match_records.append(
                            {
                                "image_name": image_path.name,
                                "image_stem": image_path.stem,
                                "group_key": group_key,
                                "light_id": int(light_id),
                                "group_id": int(matched_group_id[row_idx].item()),
                                "instance_id": int(matched_instance_id[row_idx].item()),
                                "gt_index": int(matched_gt_idx[row_idx].item()),
                                "pred_index": int(matched_pred_idx[row_idx].item()),
                                "gt_class": int(matched_gt_cls[row_idx].item()),
                                "pred_class": int(matched_pred_cls[row_idx].item()),
                                "confidence": float(matched_conf[row_idx].item()),
                                "gt_ripeness": float(matched_gt_ripe[row_idx].item()),
                                "pred_ripeness": float(pred_ripe[row_idx].item()),
                                "signed_error": float(signed_error[row_idx].item()),
                                "absolute_error": float(abs_error[row_idx].item()),
                                "pred_variance": float(pred_var[row_idx].item()),
                                "pred_sigma": float(pred_std[row_idx].item()),
                                "nll": float(instance_nll[row_idx].item()),
                                "lower_95": float(lower_95[row_idx].item()),
                                "upper_95": float(upper_95[row_idx].item()),
                                "covered_95": int(covered_95[row_idx].item()),
                            }
                        )

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
        """Set final metrics and export ripeness matches and scatter plot once."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self._export_match_results_and_scatter()

    def _build_grouped_match_table(self, matched_df):
        """Build one row per group_id + instance_id for multi-exposure consistency analysis."""
        import pandas as pd

        if matched_df.empty:
            return pd.DataFrame()

        grouped_rows = []
        group_columns = ["group_id", "instance_id"]

        for (group_id, instance_id), group in matched_df.groupby(group_columns, sort=True):
            group = group.sort_values(["light_id", "image_name", "gt_index"]).reset_index(drop=True)
            predictions = group["pred_ripeness"].to_numpy(dtype=np.float64)
            gt_value = float(group["gt_ripeness"].mean())
            mean_pred = float(np.mean(predictions))
            expected_views = int(self.group_gt_exposure_count.get((int(group_id), int(instance_id)), len(group)))
            matched_views = int(len(group))

            row = {
                "group_key": str(group["group_key"].iloc[0]),
                "group_id": int(group_id),
                "instance_id": int(instance_id),
                "gt_ripeness": gt_value,
                "expected_views": expected_views,
                "matched_views": matched_views,
                "complete_group": int(expected_views > 0 and matched_views == expected_views),
                "mean_pred_ripeness": mean_pred,
                "group_signed_error": mean_pred - gt_value,
                "group_absolute_error": abs(mean_pred - gt_value),
                "group_prediction_std": float(np.std(predictions, ddof=0)) if matched_views else float("nan"),
            }

            # 将每个曝光/视图横向写入同一行，便于查看同一草莓在不同曝光下的结果。
            for view_no, (_, item) in enumerate(group.iterrows(), start=1):
                row[f"view{view_no}_light_id"] = int(item["light_id"])
                row[f"view{view_no}_image_name"] = str(item["image_name"])
                row[f"view{view_no}_pred_ripeness"] = float(item["pred_ripeness"])
                row[f"view{view_no}_signed_error"] = float(item["signed_error"])
                row[f"view{view_no}_confidence"] = float(item["confidence"])

            grouped_rows.append(row)

        return pd.DataFrame(grouped_rows)

    @staticmethod
    def _autosize_excel_worksheets(writer):
        """Freeze headers, add filters, and set readable Excel column widths."""
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.columns:
                column_letter = cells[0].column_letter
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in cells)
                worksheet.column_dimensions[column_letter].width = min(max_length + 2, 36)

    def _draw_ripeness_error_scatter(self, matched_df, output_path):
        """Draw GT-ripeness versus signed-error scatter plot from exact validation matches."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt_values = matched_df["gt_ripeness"].to_numpy(dtype=np.float64)
        errors = matched_df["signed_error"].to_numpy(dtype=np.float64)
        absolute_errors = np.abs(errors)

        matched_count = int(len(errors))
        total_gt = int(self.maturity_gt_count)
        coverage = matched_count / total_gt if total_gt else float("nan")
        mae = float(np.mean(absolute_errors))
        bias = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        p95 = float(np.percentile(absolute_errors, 95))

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif"],
                "font.size": 14,
                "axes.labelsize": 17,
                "axes.titlesize": 18,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "legend.fontsize": 12,
            }
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(gt_values, errors, s=18, alpha=0.55, linewidths=0, label="Matched instances")
        # ax.axhline(0.0, linestyle="--", linewidth=1.3, label="Zero error")
        # ax.axhline(bias, linestyle="-.", linewidth=1.2, label=f"Mean error (Bias) = {bias:.4f}")
        ax.axhline(p95, linestyle=":", linewidth=1.2, label=f"±P95 = {p95:.4f}")
        ax.axhline(-p95, linestyle=":", linewidth=1.2)

        # 根据实际误差自动设置纵轴，避免大范围空白，同时不截断任何样本。
        max_abs_error = float(np.max(absolute_errors)) if matched_count else 0.0
        y_limit = min(1.1, max(0.05, max_abs_error * 1.08, p95 * 1.35))
        ax.set_xlim(0.0, 1.1)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel("Ground-truth ripeness")
        ax.set_ylabel("Ripeness deviation (Prediction − Ground truth)")
        ax.set_title("Ripeness Error Scatter Plot")
        ax.grid(alpha=0.25, linestyle="--")

        statistics_text = (
            f"Matched instances (N): {matched_count}\n"
            f"Total GT instances: {total_gt}\n"
            f"Coverage: {coverage:.2%}\n"
            f"MAE: {mae:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"Bias: {bias:.4f}\n"
            f"P95(|error|): {p95:.4f}"
        )
        ax.text(
            0.025,
            0.975,
            statistics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "black", "alpha": 0.92},
        )
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _export_match_results_and_scatter(self):
        """Save exact matched ripeness results to Excel and draw a scatter plot."""
        if self._match_results_exported:
            return
        self._match_results_exported = True

        if not self.match_records:
            LOGGER.warning("No Mask-IoU=0.50 ripeness matches were found; Excel and scatter plot were not created.")
            return

        try:
            import pandas as pd
        except ImportError as exc:
            LOGGER.warning(
                "Unable to export ripeness Excel. Install dependencies with: "
                "pip install pandas openpyxl matplotlib"
            )
            return

        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        excel_path = save_dir / "ripeness_match_results.xlsx"
        scatter_path = save_dir / "ripeness_error_scatter.png"

        matched_df = pd.DataFrame(self.match_records)
        matched_df = matched_df.sort_values(
            ["group_key", "instance_id", "light_id", "image_name", "gt_index"]
        ).reset_index(drop=True)
        matched_df.insert(0, "match_id", np.arange(1, len(matched_df) + 1, dtype=np.int64))

        grouped_df = self._build_grouped_match_table(matched_df)

        errors = matched_df["signed_error"].to_numpy(dtype=np.float64)
        absolute_errors = np.abs(errors)
        matched_count = int(len(matched_df))
        total_gt = int(self.maturity_gt_count)
        coverage = matched_count / total_gt if total_gt else float("nan")

        summary_items = {
            "matched_instances": matched_count,
            "total_gt_instances": total_gt,
            "match_coverage": coverage,
            "mean_gt_ripeness": float(matched_df["gt_ripeness"].mean()),
            "mean_pred_ripeness": float(matched_df["pred_ripeness"].mean()),
            "mean_signed_error_bias": float(np.mean(errors)),
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "signed_error_std": float(np.std(errors, ddof=0)),
            "median_absolute_error": float(np.median(absolute_errors)),
            "absolute_error_percentile_95": float(np.percentile(absolute_errors, 95)),
            "minimum_signed_error": float(np.min(errors)),
            "maximum_signed_error": float(np.max(errors)),
            "maximum_absolute_error": float(np.max(absolute_errors)),
            "zero_error_count": int(np.sum(np.abs(errors) <= 1e-8)),
            "grouped_instance_count": int(len(grouped_df)),
            "complete_group_count": int(grouped_df["complete_group"].sum()) if not grouped_df.empty else 0,
            "group_mae_from_table": float(grouped_df["group_absolute_error"].mean()) if not grouped_df.empty else float("nan"),
            "mean_group_prediction_std": (
                float(grouped_df["group_prediction_std"].mean()) if not grouped_df.empty else float("nan")
            ),
        }
        summary_df = pd.DataFrame({"metric": list(summary_items.keys()), "value": list(summary_items.values())})

        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                matched_df.to_excel(writer, sheet_name="MatchedResults", index=False)
                grouped_df.to_excel(writer, sheet_name="GroupedResults", index=False)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                self._autosize_excel_worksheets(writer)

            self._draw_ripeness_error_scatter(matched_df, scatter_path)
        except Exception as exc:
            LOGGER.warning(f"Failed to export ripeness matches or scatter plot: {exc}")
            return

        LOGGER.info(f"Ripeness matched results saved to {excel_path}")
        LOGGER.info(f"Ripeness error scatter plot saved to {scatter_path}")

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

        # 某些 Ultralytics 版本中 finalize_metrics/print_results 的调用顺序不同，
        # 再次调用作为兜底；内部标志会避免重复写文件。
        self._export_match_results_and_scatter()
    
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
