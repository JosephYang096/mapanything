import os
import sys
import warnings
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append("VGGT")

from Common.utils.parser_files import VideoImageReader
from Common.reconstruction.geometry_reconstruction import GeometryReconstruction
from Common.utils.metrics import calculate_metric, MetricWriter, ReconstructionMetricWriter
from Common.utils.visualization_utils import colorize_depth
from Common.lidar_post_process.calculate_volume_from_pointcloud import compute_bounding_box
from Common.lidar_post_process.calculate_volume_from_pointcloud import min_fit_obb_with_orthogonal_planes
from Common.utils.keys import ModelType, InputMode, CalibrateMode, TargetFields


def normalize_image_scale(image, target_size):
    width, length = image.shape[:2]
    if width > length:
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    if (image.shape[1] * 1.0 / image.shape[0]) == (target_size[0] * 1.0 / target_size[1]):
        return image
    else:
        return image


class PackageEstimationPipeline:
    def __init__(self, cfg, input_mode=InputMode.ImageVideo, target_fields=[TargetFields.NotReferenced]):
        self.output_dir = f"{cfg.output_dir}/{cfg.input_dir.split('/')[-2]}_{cfg.model_type}_{cfg.calibration_mode}_{cfg.calibration_factor[0]}_n{cfg.num_sample_frames}_i{cfg.model_input_frame}_s{cfg.video_sample_stride}_{cfg.experiment_name}/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.input_mode = input_mode
        self.video_reader = VideoImageReader(cfg.input_dir, cfg.num_sample_frames, load_masks=True, stride=cfg.video_sample_stride)
        if cfg.model_type != ModelType.TOF:
            self.geometry_reconstruct = GeometryReconstruction(cfg.model_type, cfg.ckpt_path, self.device)
        self.metric_writer = MetricWriter()
        self.small_package_metric_writer = MetricWriter()
        self.large_package_metric_writer = MetricWriter()
        self.recon_metric_writer = ReconstructionMetricWriter()

    def run_reconstruction(self, start_idx, images_path_list, output_dir, save_maps=True):
        """无TOF数据时，仅依赖模型推理获取深度/点云/尺度因子。"""
        predictions = self.geometry_reconstruct.inference(images_path_list)
        depths = predictions['metric_depths']
        points = predictions['points']
        scale_factor = predictions['scale_factor']
        depths_conf = predictions['depths_conf']
        # 无TOF数据，跳过MapAnything GT scale和intrinsic指标
        num_pred_frames = len(depths)  # VGGTMultiFrame只预测frame 0, num_pred_frames=1; 其他模型num_pred_frames=len(images_path_list)
        gt_scales_mapanything = [{'median_ratio': float('nan'), 'least_squares': float('nan')}] * num_pred_frames
        recon_metrics_list = []
        for i in range(num_pred_frames):
            frame_name = os.path.basename(images_path_list[i])
            cur_recon = {"image_name": frame_name}
            recon_metrics_list.append(cur_recon)
        res_list = []
        all_points = []
        all_colors = []
        for i in range(num_pred_frames):
            image_path = images_path_list[i]
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            # 从数据目录的 mask 文件读取（由 run_once 保存），替代 SAM3 推理
            mask_path = image_path.replace("_image.png", "_mask.png")
            if os.path.exists(mask_path):
                mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                # mask 可能是灰度或 RGBA，取第一个通道或 alpha 通道
                if mask_raw.ndim == 2:
                    mask = mask_raw > 0
                elif mask_raw.ndim == 3 and mask_raw.shape[2] == 4:
                    mask = mask_raw[:, :, 3] > 0  # alpha 通道
                else:
                    mask = mask_raw[:, :, 0] > 0
            else:
                mask = np.ones((image.shape[0], image.shape[1])) > 0
            point = points[i]
            depth = depths[i]
            depth_conf = depths_conf[i] > 0.1
            depth = np.squeeze(depth, axis=-1) if depth.shape[-1] == 1 else depth
            height, width = depth.shape[:2]
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            depth_conf = np.logical_and(depth_conf, mask)
            all_points.append(point[depth_conf])
            all_colors.append(image[depth_conf] / 255)
            if save_maps:
                overlay_img = image.copy()
                blue_overlay = np.zeros_like(image)
                blue_overlay[mask] = [0, 0, 255]
                blended = cv2.addWeighted(overlay_img, 0.7, blue_overlay, 0.3, 0)
                cv2.imwrite(output_dir + f'/{start_idx + i}_mask_overlay.png', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                cv2.imwrite(output_dir + f'/{start_idx + i}_depth_vis.png', cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            # raw_pcd = o3d.geometry.PointCloud()
            # raw_pcd.points = o3d.utility.Vector3dVector(point[depth_conf])
            # raw_pcd.colors = o3d.utility.Vector3dVector(image[depth_conf] / 255)
            # o3d.io.write_point_cloud(output_dir + f'/{start_idx + i}_raw_points.ply', raw_pcd, write_ascii=True)
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(all_points)
        all_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(output_dir + f'/{start_idx}_points.ply', all_pcd, write_ascii=True)
        bounding_box_ply = output_dir + f'/{start_idx}_points_bbox.ply'
        points_6d = np.hstack([all_points, all_colors * 255])
        # obb_scale_corrected, found_planes_sc = min_fit_obb_with_orthogonal_planes(
        #     all_points, output_dir=output_dir, label="linear_corrected")
        # if found_planes_sc:
        #     dims_scale_corrected = sorted(obb_scale_corrected.extent, reverse=True)
        #     box_length, box_width, box_height = dims_scale_corrected
        # else:
        box_length, box_width, box_height, _ = compute_bounding_box(
            points_6d, bounding_box_ply)
        res_list.append([box_length, box_width, box_height])
        return res_list, scale_factor, recon_metrics_list, gt_scales_mapanything

    @staticmethod
    def print_current_result(cur_result, item_name):
        print("-" * 100)
        print("current result: ", item_name)
        print(f"pred_l: {cur_result['length_pred']}, pred_w: {cur_result['width_pred']}, pred_h: {cur_result['height_pred']}")
        print(f"true_l: {cur_result['length_true']}, true_w: {cur_result['width_true']}, true_h: {cur_result['height_true']}")
        print(f"scale_factor_l: {cur_result['scale_factor_l']}, scale_factor_w: {cur_result['scale_factor_w']}, scale_factor_h: {cur_result['scale_factor_h']}")
        print(f"predicted_scale_factor: {cur_result['predicted_scale_factor']}")
        print(f"length_abs_err: {cur_result['length_abs_err']}, width_abs_err: {cur_result['width_abs_err']}, height_abs_err: {cur_result['height_abs_err']}")
        print(f"length_rel_err: {cur_result['length_rel_err']}, width_rel_err: {cur_result['width_rel_err']}, height_rel_err: {cur_result['height_rel_err']}")
        print(f"max_dim_rel_err: {cur_result['max_dim_rel_err']}, mean_dim_rel_err: {cur_result['mean_dim_rel_err']}, volume_rel_err: {cur_result['volume_rel_err']}")
        print(f"err_10_acc: {cur_result['err_10_acc']}", f"err_15_acc: {cur_result['err_15_acc']}", f"err_20_acc: {cur_result['err_20_acc']}")

    @staticmethod
    def print_recon_result(recon_result, item_name):
        print("-" * 100)
        print("reconstruction result: ", item_name)
        # scale_factor 指标 (MapAnything)
        if "ma_scale_factor_abs_err" in recon_result:
            print(f"ma_scale_factor_abs_err: {recon_result['ma_scale_factor_abs_err']:.4f}, ma_scale_factor_rel_err: {recon_result['ma_scale_factor_rel_err']:.4f}%, ma_scale_factor_ratio: {recon_result['ma_scale_factor_ratio']:.4f}, ma_gt_scale_factor: {recon_result['ma_gt_scale_factor']:.4f}")
        if "ma_ls_gt_scale_factor" in recon_result:
            print(f"ma_ls_gt_scale_factor: {recon_result['ma_ls_gt_scale_factor']:.4f}")
        # intrinsic (fx, fy) 指标
        if "fx_abs_err" in recon_result:
            print(f"fx_abs_err: {recon_result['fx_abs_err']:.4f}, fx_rel_err: {recon_result['fx_rel_err']:.4f}%, fy_abs_err: {recon_result['fy_abs_err']:.4f}%, fy_rel_err: {recon_result['fy_rel_err']:.4f}%")

    def run_once(self,):
        num_files = len(self.video_reader) if self.cfg.num_test_cases == -1 else min(len(self.video_reader), self.cfg.num_test_cases)
        all_pred_scales = []   # collect predicted scale_factors across all test cases
        all_gt_scales_ma = []  # collect GT scale_factors (MapAnything median_ratio) across all test cases
        all_scale_errors = []     # scale_error = pred_scale - gt_scale per sample
        all_volume_errors = []    # volume_rel_err per sample
        all_sample_names_err = [] # sample names for the scatter plot
        for idx in tqdm(range(num_files)):
            # 从 VideoImageReader 的 case prefix 提取 GT 维度
            # case prefix 格式: "{object}_{target_field}_{dim1}_{dim2}_{dim3}"
            # 例: "3M标识包裹_无参照物_25_25_29", "DL_43.0_31.8_34.0_RGB"
            case_prefix = self.video_reader.file_names[idx]
            segments = case_prefix.split("_")
            numeric_segs = []
            for seg in segments:
                try:
                    numeric_segs.append(float(seg))
                except ValueError:
                    continue
            if len(numeric_segs) < 3:
                warnings.warn(f"Cannot extract 3 GT dimensions from case prefix: {case_prefix}")
                continue
            # gts = numeric_segs[-3:]
            gts = numeric_segs[:3]
            sorted_gt = sorted(gts, reverse=True)
            target_name = case_prefix
            out_path = self.output_dir + target_name
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            image_path_lists = []
            # VideoImageReader 返回 list[dict] (load_masks=True): {"image": np.ndarray, "mask": np.ndarray|None}
            all_items = self.video_reader(idx)
            for frame_idx, item in enumerate(all_items):
                if isinstance(item, dict):
                    image = item["image"]
                    mask = item.get("mask", None)
                else:
                    # 向后兼容: load_masks=False 时返回纯 np.ndarray
                    image = item
                    mask = None
                image = normalize_image_scale(image, self.cfg.input_image_size)
                out_img_file = out_path + f"/{frame_idx}_image.png"
                cv2.imwrite(out_img_file, image)
                image_path_lists.append(out_img_file)
                # 保存 mask 到磁盘，供 run_reconstruction 读取
                if mask is not None:
                    out_mask_file = out_path + f"/{frame_idx}_mask.png"
                    mask = normalize_image_scale(mask, self.cfg.input_image_size)
                    cv2.imwrite(out_mask_file, mask)
            if self.cfg.model_input_frame == -1:
                res_list, scale_factor, recon_metrics_list, gt_scales_ma = self.run_reconstruction(0, image_path_lists, out_path, self.cfg.save_maps)
            else:
                num_infer = len(image_path_lists) // self.cfg.model_input_frame
                last_num = len(image_path_lists) % self.cfg.model_input_frame
                res_list = []
                scale_factor = []
                recon_metrics_list = []
                gt_scales_ma = []
                for i in range(num_infer):
                    input_paths = image_path_lists[i * self.cfg.model_input_frame:(i + 1) * self.cfg.model_input_frame]
                    output, scale, recon_metrics, gt_sc_ma = self.run_reconstruction(i * self.cfg.model_input_frame,
                        input_paths, out_path, self.cfg.save_maps)
                    res_list.extend(output)
                    scale_factor.append(scale)
                    recon_metrics_list.extend(recon_metrics)
                    gt_scales_ma.extend(gt_sc_ma if isinstance(gt_sc_ma, list) else [gt_sc_ma])
                if last_num != 0:
                    input_paths = image_path_lists[-(last_num+1):]
                    output, scale, recon_metrics, gt_sc_ma = self.run_reconstruction(num_infer * self.cfg.model_input_frame,
                        input_paths, out_path, self.cfg.save_maps)
                    res_list.extend(output)
                    scale_factor.append(scale)
                    recon_metrics_list.extend(recon_metrics)
                    gt_scales_ma.extend(gt_sc_ma if isinstance(gt_sc_ma, list) else [gt_sc_ma])
                scale_factor = np.mean(scale_factor)
            if not self.cfg.save_maps:
                for image_path in image_path_lists:
                    os.remove(image_path)
            cur_volume_errors = []  # collect volume_rel_err for current test case
            # 按 sorted_gt 最小值分类：>= 20 为大包裹，< 20 为小包裹
            package_metric_writer = (
                self.large_package_metric_writer
                if sorted_gt[-1] >= 20
                else self.small_package_metric_writer
            )
            for i, preds in enumerate(res_list):
                if preds[0] == -1:
                    continue
                cur_result = calculate_metric(target_name + f"_{i}.png", sorted_gt, preds, scale_factor,
                                              self.cfg.calibration_mode, self.cfg.calibration_factor)
                self.print_current_result(cur_result, f"{out_img_file}_{i}")
                self.metric_writer.update_current_result(cur_result)
                package_metric_writer.update_current_result(cur_result)
                cur_volume_errors.append(cur_result['volume_rel_err'])
            # 更新重建评测指标 (rel_depth / scale_factor / intrinsic)
            for recon_result in recon_metrics_list:
                self.print_recon_result(recon_result, recon_result.get("image_name", ""))
                self.recon_metric_writer.update_current_result(recon_result)
            self.metric_writer.save_instance_result(out_path)
            self.metric_writer.update_all_predictions()
            self.metric_writer.reset_instance_predictions()
            package_metric_writer.update_all_predictions()
            package_metric_writer.reset_instance_predictions()
            self.recon_metric_writer.save_instance_result(out_path)
            self.recon_metric_writer.update_all_predictions()
            self.recon_metric_writer.reset_instance_predictions()
            # ── Collect pred/GT scale_factor for scatter plot ──────────
            pred_sf_val = float(scale_factor) if isinstance(scale_factor, (int, float, np.floating)) else float(np.mean(scale_factor))
            all_pred_scales.append(pred_sf_val)
            # Collect MapAnything GT scale_factor (median_ratio)
            if isinstance(gt_scales_ma, list) and len(gt_scales_ma) > 0:
                ma_vals = [s['median_ratio'] for s in gt_scales_ma if not np.isnan(s['median_ratio'])]
                if len(ma_vals) > 0:
                    all_gt_scales_ma.append(float(np.mean(ma_vals)))
                else:
                    all_gt_scales_ma.append(float('nan'))
            else:
                all_gt_scales_ma.append(float('nan'))

            # ── Print per-case GT vs Pred scale_factor ─────────────────
            gt_ma_val = all_gt_scales_ma[-1] if len(all_gt_scales_ma) > 0 else float('nan')
            print(f"\n{'='*60}")
            print(f"[{target_name}] Scale Factor Comparison:")
            print(f"  Predicted:        {pred_sf_val:.6f}")
            if not np.isnan(gt_ma_val):
                print(f"  GT (MapAnything): {gt_ma_val:.6f}   (abs_err: {abs(pred_sf_val - gt_ma_val):.6f}, rel_err: {abs(pred_sf_val - gt_ma_val)/max(abs(gt_ma_val),1e-6)*100:.2f}%)")
                # ── Compute and print scale_error ──────────────────────
                scale_error = pred_sf_val - gt_ma_val
                vol_err = float(np.mean(cur_volume_errors)) if len(cur_volume_errors) > 0 else float('nan')
                all_scale_errors.append(scale_error)
                all_volume_errors.append(vol_err)
                all_sample_names_err.append(target_name)
                print(f"  scale_error (pred - GT): {scale_error:.6f}")
                if not np.isnan(vol_err):
                    print(f"  volume_rel_err:          {vol_err:.2f}%")
            print(f"{'='*60}")

        print("\n所有包裹评测指标:")
        self.metric_writer.save_and_print_summary_result(self.output_dir)
        if self.small_package_metric_writer.all_cnt > 0:
            print("\n小包裹评测指标（sorted_gt 最小值 <20）:")
            small_output_dir = os.path.join(self.output_dir, "small_packages")
            os.makedirs(small_output_dir, exist_ok=True)
            self.small_package_metric_writer.save_and_print_summary_result(small_output_dir)
        else:
            print("\n小包裹评测指标（sorted_gt 最小值 <20）: 无有效样本")
        if self.large_package_metric_writer.all_cnt > 0:
            print("\n大包裹评测指标（sorted_gt 最小值 >= 20）:")
            large_output_dir = os.path.join(self.output_dir, "large_packages")
            os.makedirs(large_output_dir, exist_ok=True)
            self.large_package_metric_writer.save_and_print_summary_result(large_output_dir)
        else:
            print("\n大包裹评测指标（sorted_gt 最小值 >= 20）: 无有效样本")
        self.recon_metric_writer.save_and_print_summary_result(self.output_dir)

        # ── Scale factor statistics & scatter plot ─────────────────────
        if len(all_pred_scales) > 0 and len(all_gt_scales_ma) > 0:
            pred_arr = np.array(all_pred_scales)
            gt_ma_arr = np.array(all_gt_scales_ma)

            # 过滤NaN值
            valid_mask = ~np.isnan(gt_ma_arr)
            gt_ma_valid = gt_ma_arr[valid_mask]
            pred_valid = pred_arr[valid_mask]

            print("\n" + "=" * 60)
            print("Scale Factor Statistics:")
            print(f"  Predicted  — mean: {pred_arr.mean():.4f}, std: {pred_arr.std():.4f}, "
                  f"min: {pred_arr.min():.4f}, max: {pred_arr.max():.4f}")
            if len(gt_ma_valid) > 0:
                print(f"  GT (MapAnything)   — mean: {gt_ma_valid.mean():.4f}, std: {gt_ma_valid.std():.4f}, "
                      f"min: {gt_ma_valid.min():.4f}, max: {gt_ma_valid.max():.4f}")
                print(f"  Pred vs GT (MapAnything)   Abs error — mean: {np.abs(pred_valid - gt_ma_valid).mean():.4f}")
                print(f"  Pred vs GT (MapAnything)   Rel error — mean: {(np.abs(pred_valid - gt_ma_valid) / np.abs(gt_ma_valid).clip(min=1e-6)).mean() * 100:.2f}%")
            print("=" * 60)

            # ── Scatter plot 1: pred vs GT (MapAnything) 分布对比 ──────
            sample_indices = np.arange(len(pred_arr))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(sample_indices, pred_arr, c='blue', alpha=0.7, s=40, label='预测值', zorder=3)
            if len(gt_ma_valid) > 0:
                ax.scatter(sample_indices[valid_mask], gt_ma_valid, c='green', alpha=0.7, s=40, marker='^', label='GT (MapAnything)', zorder=4)
            ax.set_xlabel('样本索引', fontsize=12)
            ax.set_ylabel('缩放因子', fontsize=12)
            title_str = f'缩放因子分布  (预测 μ={pred_arr.mean():.3f}±{pred_arr.std():.3f}'
            if len(gt_ma_valid) > 0:
                title_str += f',  GT_ma μ={gt_ma_valid.mean():.3f}±{gt_ma_valid.std():.3f}'
            title_str += ')'
            ax.set_title(title_str, fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'scale_factor_scatter.png')
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Scale factor scatter plot saved to: {save_path}")

            # ── Scatter plot 2: Pred vs GT (MapAnything) 对比图 ────────
            if len(gt_ma_valid) > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(gt_ma_valid, pred_valid, c='green', alpha=0.6, s=40, label='预测 vs GT')
                min_val = min(gt_ma_valid.min(), pred_valid.min())
                max_val = max(gt_ma_valid.max(), pred_valid.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
                abs_err = np.abs(pred_valid - gt_ma_valid).mean()
                rel_err = (np.abs(pred_valid - gt_ma_valid) / np.abs(gt_ma_valid).clip(min=1e-6)).mean() * 100
                ax.set_xlabel('GT 缩放因子 (MapAnything)', fontsize=12)
                ax.set_ylabel('预测缩放因子', fontsize=12)
                ax.set_title(f'预测 vs GT (MapAnything)\n绝对误差: {abs_err:.4f}, 相对误差: {rel_err:.2f}%', fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save_path2 = os.path.join(self.output_dir, 'scale_factor_pred_vs_gt.png')
                fig.savefig(save_path2, dpi=150)
                plt.close(fig)
                print(f"Scale factor Pred vs GT plot saved to: {save_path2}")

            # ── Scatter plot 3: scale_error vs volume_error ─────────────
            if len(all_scale_errors) > 0 and len(all_volume_errors) > 0:
                se_arr = np.array(all_scale_errors)
                ve_arr = np.array(all_volume_errors)
                valid_se_ve = ~np.isnan(ve_arr)
                se_valid = se_arr[valid_se_ve]
                ve_valid = ve_arr[valid_se_ve]
                names_valid = [all_sample_names_err[i] for i in range(len(valid_se_ve)) if valid_se_ve[i]]

                if len(se_valid) > 0:
                    # Mark volume_error > 50% threshold
                    VOLUME_ERR_THRESHOLD = 50.0
                    ve_high_mask = ve_valid > VOLUME_ERR_THRESHOLD

                    # Outlier detection using IQR on both dimensions
                    def detect_outliers_iqr(data):
                        q1 = np.percentile(data, 25)
                        q3 = np.percentile(data, 75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        return (data < lower) | (data > upper)

                    se_outliers = detect_outliers_iqr(se_valid)
                    ve_outliers = detect_outliers_iqr(ve_valid)
                    outliers = se_outliers | ve_outliers

                    fig, ax = plt.subplots(figsize=(10, 8))
                    # Normal points (not outlier, volume_err <= 50%)
                    normal_mask = ~outliers & ~ve_high_mask
                    ax.scatter(se_valid[normal_mask], ve_valid[normal_mask],
                               c='steelblue', alpha=0.7, s=50, label='正常点', zorder=3, edgecolors='grey', linewidths=0.5)
                    # IQR outlier points (not volume_err > 50%)
                    iqr_only = outliers & ~ve_high_mask
                    if iqr_only.any():
                        ax.scatter(se_valid[iqr_only], ve_valid[iqr_only],
                                   c='red', alpha=0.9, s=80, marker='X', label=f'IQR离群点 ({iqr_only.sum()})', zorder=4, edgecolors='darkred', linewidths=0.8)
                    # Volume error > 50% points
                    if ve_high_mask.any():
                        ax.scatter(se_valid[ve_high_mask], ve_valid[ve_high_mask],
                                   c='orange', alpha=0.9, s=90, marker='D', label=f'体积误差>{VOLUME_ERR_THRESHOLD:.0f}% ({ve_high_mask.sum()})',
                                   zorder=5, edgecolors='darkorange', linewidths=0.8)
                        for idx in np.where(ve_high_mask)[0]:
                            ax.annotate(names_valid[idx], (se_valid[idx], ve_valid[idx]),
                                        fontsize=7, alpha=0.9, color='darkorange',
                                        xytext=(6, 6), textcoords='offset points',
                                        arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))
                    # Annotate IQR outliers
                    if iqr_only.any():
                        for idx in np.where(iqr_only)[0]:
                            ax.annotate(names_valid[idx], (se_valid[idx], ve_valid[idx]),
                                        fontsize=7, alpha=0.85, color='darkred',
                                        xytext=(6, 6), textcoords='offset points',
                                        arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

                    ax.set_xlabel('缩放误差 (预测缩放 - GT缩放)', fontsize=12)
                    ax.set_ylabel('体积相对误差 (%)', fontsize=12)
                    ax.set_title(f'缩放误差 vs 体积误差  (样本数={len(se_valid)}, 体积误差>{VOLUME_ERR_THRESHOLD:.0f}%={ve_high_mask.sum()})', fontsize=13)
                    ax.legend(fontsize=11, loc='upper left')
                    ax.grid(True, alpha=0.3)

                    # Add 50% threshold line
                    ax.axhline(y=VOLUME_ERR_THRESHOLD, color='orange', linestyle='--', alpha=0.6, linewidth=1.0, label=f'体积误差 {VOLUME_ERR_THRESHOLD:.0f}%')

                    # Add Pearson correlation
                    if len(se_valid) > 1:
                        corr = np.corrcoef(se_valid, ve_valid)[0, 1]
                        ax.text(0.95, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
                                fontsize=10, verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    # Add zero line for scale_error
                    ax.axvline(x=0, color='grey', linestyle='--', alpha=0.4, linewidth=0.8)

                    plt.tight_layout()
                    save_path3 = os.path.join(self.output_dir, 'scale_error_vs_volume_error.png')
                    fig.savefig(save_path3, dpi=150)
                    plt.close(fig)
                    print(f"Scale error vs Volume error scatter plot saved to: {save_path3}")

                    # Print outlier details
                    if outliers.any():
                        print(f"\n  Outlier samples (IQR, scale_error vs volume_error):")
                        for idx in np.where(outliers)[0]:
                            print(f"    {names_valid[idx]}: scale_error={se_valid[idx]:.6f}, volume_rel_err={ve_valid[idx]:.2f}%")

                    # Print volume_error > 50% sample names
                    if ve_high_mask.any():
                        print(f"\n  Samples with volume_rel_err > {VOLUME_ERR_THRESHOLD:.0f}%:")
                        for idx in np.where(ve_high_mask)[0]:
                            print(f"    {names_valid[idx]}: scale_error={se_valid[idx]:.6f}, volume_rel_err={ve_valid[idx]:.2f}%")
                    else:
                        print(f"\n  No samples with volume_rel_err > {VOLUME_ERR_THRESHOLD:.0f}%")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run package dimension estimation using feed-forward 3D reconstruction inference on videos.")
    parser.add_argument("--input_dir", default="/mnt/workspace/common/data/self_build/Depth_Est_Test/RGB_Only/2026-0514-一线-地面-ziyi/", help="Input directory.")
    parser.add_argument("--output_dir", default="/mnt/workspace/zhangfan/code/package_dimension_estimate/test_all_0807", help="Output directory.")
    parser.add_argument('--model_type', choices=['VGGT', 'VGG3T', 'DA3', 'MOGE2', 'TOF'], default=ModelType.VGGT_METRIC_MULTI_FRAME, help="Select the mode type for 3D reconstruction.")
    parser.add_argument("--ckpt_path", default="/mnt/workspace/zhangfan/models/vggt_metric/multi_frame/baseline_size_518_294/vggt_metric_frame12_stride1_0807/best_ckpt_epoch_29.pth", help="Checkpoint path for model.")
    parser.add_argument("--experiment_name", type=str, default="test_vis1", help="Name of experimental item.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num_test_cases", type=int, default=-1, help="Number of cases for testing.")
    parser.add_argument("--video_sample_stride", type=int, default=2, help="Frame stride for video sampling.")
    parser.add_argument("--model_input_frame", type=int, default=12, help="Number of input frames for model input.")
    parser.add_argument("--num_sample_frames", type=int, default=12, help="Number of video frames for testing.")
    parser.add_argument("--input_image_size", type=list, default=[640, 480], help="Input image size for test pipeline.")
    parser.add_argument("--calibration_mode", choices=['manual', 'tof', 'metric'], default=CalibrateMode.MANUAL, help="Calibration mode for the absolute scale.")
    parser.add_argument("--calibration_factor", type=list, default=[1.0, 1.0, 1.0], help="Calibration factor of the absolute scale.")
    parser.add_argument("--save_maps", type=bool, default=False, help="Whether to save maps for debugging.")
    args = parser.parse_args()
    pipeline = PackageEstimationPipeline(args, input_mode=InputMode.Video, target_fields=[TargetFields.NotReferenced])
    pipeline.run_once()
