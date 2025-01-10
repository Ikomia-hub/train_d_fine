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
    and log metrics to MLflow if installed.
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
        # Accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # Gather final stats
    stats = {}
    # If you have other metrics in metric_logger, collect them
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Log coco metrics
    if coco_evaluator is not None:
        if 'bbox' in iou_types and coco_evaluator.coco_eval.get('bbox'):
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types and coco_evaluator.coco_eval.get('segm'):
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # Optionally log these stats to MLflow
    if mlflow and dist_utils.is_main_process():
        if epoch is not None:
            # log with step as 'epoch'
            mlflow.log_metrics(
                {f"eval_{k}": v for k, v in stats.items() if isinstance(v, float)},
                step=epoch
            )
            # For lists like mAP array, you can either log them individually
            # or omit them if they are not scalar.
            # E.g. if stats['coco_eval_bbox'] is a list of floats, do:
            if 'coco_eval_bbox' in stats:
                for i, val in enumerate(stats['coco_eval_bbox']):
                    mlflow.log_metric(f"coco_eval_bbox_{i}", val, step=epoch)
        else:
            # If epoch is None, log at step=0 or some fallback
            mlflow.log_metrics(
                {f"eval_{k}": v for k, v in stats.items() if isinstance(v, float)})

    return stats, coco_evaluator
