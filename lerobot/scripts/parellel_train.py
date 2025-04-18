#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
# from torch.amp import GradScaler  # 제거: Accelerate를 사용하므로 GradScaler가 필요없음
from torch.optim import Optimizer

# 추가: HuggingFace Accelerate 라이브러리 임포트
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    lr_scheduler=None,
    use_amp: bool = False,
    accelerator=None,  # Accelerate 객체를 전달받음
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    # Accelerate의 autocast를 이용해 AMP 모드로 forward 수행 (use_amp가 True일 경우)
    context_manager = accelerator.autocast() if use_amp and accelerator is not None else nullcontext()
    with context_manager:
        loss, output_dict = policy.forward(batch)
        # TODO: policy.unnormalize_outputs(output_dict) 필요시 구현
    # Accelerate가 backward 및 AMP gradient scaling을 자동으로 처리함
    accelerator.backward(loss)
    # Accelerate의 gradient clipping 헬퍼를 사용
    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    # 옵티마이저 스텝 및 gradient 초기화
    optimizer.step()
    optimizer.zero_grad()
    # 학습률 스케줄러 업데이트 (배치마다 업데이트)
    if lr_scheduler is not None:
        lr_scheduler.step()
    # # 내부 업데이트 (예: EMA) 수행 (정의되어 있다면)
    # if has_method(policy, "update"):
    #     policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    
    # Accelerate 객체 초기화 (여러 GPU 사용을 위해)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # accelerator = Accelerator(find_unused_parameters=True)
    
    # 메인 프로세스에서만 WandB Logger를 초기화 (분산 학습 시 중복 로그 방지)
    if cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if accelerator.is_main_process:
            logging.info("Logs will be saved locally.")
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Accelerate가 관리하는 device 사용
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # 평가용 환경 생성 (필요한 경우)
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler 제거 (Accelerate 사용)

    step = 0  # policy 업데이트 횟수

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if accelerator.is_main_process:
        logging.info("Output dir: " + f"{cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # 데이터로더 생성 (offline training)
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # Accelerate를 이용해 policy, optimizer, 데이터로더를 분산/혼합 정밀도 환경에 맞게 준비
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # (필요시) 배치를 accelerator.device로 이동; accelerator.prepare()가 이를 처리하지 않는 경우 대비
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # update_policy 호출 시 accelerator 객체를 전달하여 분산/AMP 처리를 수행함
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            accelerator=accelerator,
        )

        step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # 메인 프로세스에서만 로깅, 체크포인트 저장, 평가 수행 (중복 방지)
        if accelerator.is_main_process:
            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                origin_policy = accelerator.unwrap_model(policy)
                save_checkpoint(checkpoint_dir, step, cfg, origin_policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with (
                    torch.no_grad(),
                    accelerator.autocast() if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if accelerator.is_main_process and eval_env:
        eval_env.close()
    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
