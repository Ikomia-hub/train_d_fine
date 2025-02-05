"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

# Attempt to import mlflow
try:
    import mlflow
    assert hasattr(mlflow, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    mlflow = None

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

# Example: If you have your own logger
import logging
LOGGER = logging.getLogger(__name__)
PREFIX = "[MLFLOW]"


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs
):
    """
    Train one epoch and log metrics to both TensorBoard (if writer is provided)
    and MLflow (if mlflow is installed and active).
    """
    if mlflow and mlflow.active_run() is None:
        # This sets or creates an experiment named "D-Fine_Experiments"
        mlflow.set_experiment("D-Fine_Experiments")
        mlflow.start_run(run_name="DET_Training")

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)

    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    # If MLflow is installed, we can ensure there's an active run.
    # You can also explicitly start a run here if needed:
    if mlflow and mlflow.active_run() is None:
        mlflow.start_run(run_name="DET_Training")

    # Log to see what run is active (optional)
    if mlflow:
        active_run = mlflow.active_run()
        LOGGER.info(
            f"{PREFIX} Active run ID: {active_run.info.run_id if active_run else 'None'}")

    # We'll track the global step for logging
    dataset_size = len(data_loader)  # total steps in one epoch

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * dataset_size + i  # global step across epochs

        metas = dict(
            epoch=epoch,
            step=i,
            global_step=global_step,
            epoch_step=dataset_size
        )

        # --- Forward & Backward Pass ---
        if scaler is not None:
            # Automatic Mixed Precision
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            # Basic NaN check (in case training explodes)
            if (torch.isnan(outputs['pred_boxes']).any() or
                    torch.isinf(outputs['pred_boxes']).any()):
                print(outputs['pred_boxes'])
                # Example: Save a checkpoint if needed
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            # Compute loss outside of autocast (avoiding extra half-precision ops)
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            # Gradient clipping
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # No AMP
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # Exponential Moving Average update
        if ema is not None:
            ema.update(model)

        # Learning rate warmup scheduling
        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # --- Metric Logging ---
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # --- TensorBoard Logging ---
        if writer and dist_utils.is_main_process():
            if global_step % 10 == 0:  # log every 10 steps
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        # --- MLflow Logging ---
        if mlflow and dist_utils.is_main_process():
            # log metrics each step or less frequently if desired
            mlflow.log_metric(
                "train_loss", loss_value.item(), step=global_step)
            mlflow.log_metric(
                "lr", optimizer.param_groups[0]["lr"], step=global_step)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Final aggregated stats for the epoch
    epoch_stats = {k: meter.global_avg for k,
                   meter in metric_logger.meters.items()}

    # Optionally, log final epoch stats to MLflow
    if mlflow and dist_utils.is_main_process():
        mlflow.log_metrics(
            {f"epoch_{k}": v for k, v in epoch_stats.items()}, step=epoch)

    return epoch_stats


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device: torch.device,
    epoch: int = None
):
    """
    Evaluate the model on the given data_loader, update the coco_evaluator,
    and log detailed metrics to MLflow including all COCO mAPs and Average Precision/Recall metrics.
    """
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    # If MLflow is installed, ensure an active run
    if mlflow and mlflow.active_run() is None:
        mlflow.start_run(run_name="DET_Evaluation")

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res = {
            target['image_id'].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}

    # Detailed COCO evaluation metrics logging
    if coco_evaluator is not None and mlflow and dist_utils.is_main_process():
        for iou_type in iou_types:
            if iou_type not in coco_evaluator.coco_eval:
                continue

            eval_results = coco_evaluator.coco_eval[iou_type].stats.tolist()
            stats[f'coco_eval_{iou_type}'] = eval_results

            # Map COCO metrics to MLflow-compliant names, matching exact order from COCO output
            metric_descriptions = [
                ('AP_all_IoU_50_95', 'AP all IoU=0.50:0.95'),
                ('AP_all_IoU_50', 'AP all IoU=0.50'),
                ('AP_all_IoU_75', 'AP all IoU=0.75'),
                ('AP_small', 'AP small objects'),
                ('AP_medium', 'AP medium objects'),
                ('AP_large', 'AP large objects'),
                ('AR_1_all', 'AR maxDets=1'),
                ('AR_10_all', 'AR maxDets=10'),
                ('AR_100_all', 'AR maxDets=100'),
                ('AR_100_small', 'AR small objects'),
                ('AR_100_medium', 'AR medium objects'),
                ('AR_100_large', 'AR large objects'),
                ('AR_100_IoU_50', 'AR IoU=0.50'),
                ('AR_100_IoU_75', 'AR IoU=0.75')
            ]

            # Log each metric with its description
            for (metric_name, metric_desc), value in zip(metric_descriptions, eval_results):
                metric_key = f"{iou_type}_{metric_name}"
                if epoch is not None:
                    mlflow.log_metric(metric_key, value, step=epoch)
                else:
                    mlflow.log_metric(metric_key, value)

                # For negative values (like -1.000 for small objects when none present),
                # log an additional flag metric
                if value < 0:
                    flag_key = f"{iou_type}_{metric_name}_no_instances"
                    if epoch is not None:
                        mlflow.log_metric(flag_key, 1, step=epoch)
                    else:
                        mlflow.log_metric(flag_key, 1)

            # Log key summary metrics
            if epoch is not None:
                # Primary mAP (IoU=0.50:0.95)
                mlflow.log_metric(f"{iou_type}_mAP",
                                  eval_results[0], step=epoch)
                # mAP at IoU=0.50
                mlflow.log_metric(f"{iou_type}_mAP50",
                                  eval_results[1], step=epoch)
                # mAP at IoU=0.75
                mlflow.log_metric(f"{iou_type}_mAP75",
                                  eval_results[2], step=epoch)
            else:
                mlflow.log_metric(f"{iou_type}_mAP", eval_results[0])
                mlflow.log_metric(f"{iou_type}_mAP50", eval_results[1])
                mlflow.log_metric(f"{iou_type}_mAP75", eval_results[2])

    return stats, coco_evaluator
