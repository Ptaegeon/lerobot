#!/usr/bin/env python
# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

This version integrates a consistency model based on the KarrasDenoiser,
which is used to improve inference efficiency and training stability.
"""

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

# Importing additional modules for consistency model
import torch as th
import copy

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.consistency_diffusion.configuration_consistency_diffusion import ConsistencyDiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)


class ConsistencyDiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = ConsistencyDiffusionConfig
    name = "consistency_diffusion"

    def __init__(
        self,
        config: ConsistencyDiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration instance.
            dataset_stats: Statistics for normalization (if not provided here, expected later via load_state_dict).
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Queues store the latest observations and actions.
        self._queues = None

        self.diffusion = ConsistencyDiffusionModel(config)
        self.K = config.steps
        self.s0 = config.initial_discret_steps
        self.s1 = config.target_discret_steps
        self.mu0 = config.ema_decay
        
        self.reset()

    def N_schedule(self, k:int) -> int:
        frac = k / self.K
        inside = frac * ((self.s1 + 1)**2 - self.s0**2) + self.s0**2
        n_val = math.sqrt(inside) - 1.0
        return max(1, math.ceil(n_val) + 1)

    def mu_schedule(self, k: int) -> float:
        N_k = self.N_schedule(k)
        return math.exp(self.s0 * math.log(self.mu0) / N_k)

        
    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Call this on env.reset()."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

        self.train_steps = 0
        
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select a single action given environment observations.

        - Caches the latest n_obs_steps observations.
        - Uses the diffusion model to generate a long trajectory (of length horizon).
        - Only n_action_steps consecutive actions (starting from the current step) are used for execution.
        
        (For detailed schematic, see the original comments.)
        """
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy to avoid modifying original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Update queues with new observations.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # Stack the latest observations to form a batch input.
            batch_stack = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch_stack)
            # Unnormalize outputs if needed.
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Forward pass for computing training/validation loss."""
        self.train_steps += 1
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        
        self.N_k = self.N_schedule(self.train_steps)                    # Scheduled N(k) 
        loss_dict = self.diffusion.consistency_losses(batch, num_scales=self.N_k)
        loss = loss_dict['loss'].mean()
        
        if self.train_steps % 5 == 0:
            current_mu = self.mu_schedule(self.train_steps)      # Scheduled mu(k)
            self.step_ema(current_mu)
            

        
        return loss, None
    
    def step_ema(self, ema_decay):
        if self.train_steps < self.config.step_start_ema:
            return
        
        for current_params, ma_params in zip(self.diffusion.unet.parameters(), self.diffusion.target_unet.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = old_weight * ema_decay + (1 - ema_decay) * up_weight

class ConsistencyDiffusionModel(nn.Module):
    def __init__(self, config: ConsistencyDiffusionConfig):
        super().__init__()
        self.config = config

        # Build observation encoders.
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        self.target_unet = copy.deepcopy(self.unet)
        # self.noise_scheduler = _make_noise_scheduler(
        #     config.noise_scheduler_type,
        #     num_train_timesteps=config.num_train_timesteps,
        #     beta_start=config.beta_start,
        #     beta_end=config.beta_end,
        #     beta_schedule=config.beta_schedule,
        #     clip_sample=config.clip_sample,
        #     clip_sample_range=config.clip_sample_range,
        #     prediction_type=config.prediction_type,
        # )

        # if config.num_inference_steps is None:
        #     self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        # else:
        #     self.num_inference_steps = config.num_inference_steps

        # ================================================================================
        # NEW: Consistency Model Integration
        # ------------------------------------------------------------------------------
        # Instantiate the KarrasDenoiser for consistency-based training and sampling.
        # The following hyperparameters are set to default values; they can be added to the config if desired.
        self.consistency_denoiser = KarrasDenoiser(
            action_dim=config.action_feature.shape[0],
            horizon=config.horizon,
            # device=get_device_from_parameters(self),
            device=config.device,
            sigma_data=config.sigma_data,
            sigma_max=config.sigma_max,
            sigma_min=config.sigma_min,
            rho=config.rho,
            weight_schedule=config.weight_schedule,
            steps=config.initial_discret_steps,
            ts=config.ts,
            sampler=config.sampler,
            clip_denoised=config.clip_denoised,
        )
        # ================================================================================
    


    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and state into a single conditioning vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = [batch[OBS_ROBOT]]
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Expects batch to contain:
          - "observation.state": (B, n_obs_steps, state_dim)
          - "observation.images" and/or "observation.environment_state"
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        
        # [NOTE: Replace diffusion sampling with consistency sampling using KarrasDenoiser.sample]
        # actions = self.conditional_sample(batch_size, global_cond=global_cond)
        
        # [NOTE: Using consistency model sampling instead of diffusion conditional_sample]
        actions = self.sample(self.unet, global_cond, num=batch_size)

        # Select n_action_steps from the generated horizon.
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    # ================================================================================
    # NEW: Consistency Model Methods
    # ------------------------------------------------------------------------------
    def sample(self, model, state, num=10) -> Tensor:
        """
        Consistency sampling using the KarrasDenoiser.
        This method will generate actions from the given model and state using the consistency model sampler.
        """
        x = self.consistency_denoiser.sample(model, state, num)
        return x

    def consistency_losses(
        # self, model, x_start: Tensor, num_scales: int, target_model, state: Tensor, noise: Tensor | None = None
        self, batch: dict, num_scales: int = 40,
    ) -> dict:
        """
        Compute consistency losses using the KarrasDenoiser implementation.
        """
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps
        
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)
        
        x_start = batch['action']
        state = global_cond
        scale = None
        
        return self.consistency_denoiser.consistency_losses(self.unet, x_start, num_scales, state=state, target_model=self.target_unet)
    
    def set_scale(self):
        scales = np.ceil(np.sqrt((step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)+ start_scales**2)- 1).astype(np.int32)
        scales = np.maximum(scales, 1)
        c = -np.log(start_ema) * start_scales
        target_ema = np.exp(-c / scales)
        scales = scales + 1
        return None
    # ================================================================================
    # END NEW: Consistency Model Methods
    # ================================================================================


class SpatialSoftmax(nn.Module):
    """
    Spatial soft-argmax to extract keypoints from feature maps.
    """
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector, including optional cropping."""
    def __init__(self, config: ConsistencyDiffusionConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings (1D) as used in Transformer models."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """1D Conv block: Conv1d --> GroupNorm --> Mish activation."""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """1D Convolutional U-Net with FiLM modulation for conditioning."""
    def __init__(self, config: ConsistencyDiffusionConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        timesteps_embed = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """1D ResNet block with FiLM conditioning."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


# ================================================================================
# NEW: Consistency Model Implementation using KarrasDenoiser
# ------------------------------------------------------------------------------

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class KarrasDenoiser:
    """
    Consistency-based denoiser adapted from Karras et al. (2022).
    This class implements denoising, sampling, and consistency loss computation.
    """
    def __init__(
        self,
        action_dim,
        horizon,
        device,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        ts=None,
        sampler="onestep", 
        clip_denoised=True,
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.num_timesteps = steps

        self.device = device
        
        self.sampler = sampler
        self.steps = steps
        self.ts = [0, 20, 40] if ts is None else ts

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (((sigma - self.sigma_min) ** 2) + self.sigma_data**2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs Karras et al. (2022) noise schedule."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        state=None,
        target_model=None,
        noise=None,
    ):
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            # Returns the denoised output from the model.
            return self.denoise(model, x, t, state)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            # Simple Euler method update.
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            return samples

        indices = th.randint(0, num_scales - 1, (x_start.shape[0],), device=x_start.device)

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state)
        x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state).detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs)

        recon_diffs = (distiller - x_start) ** 2
        recon_loss = mean_flat(recon_diffs) * weights

        terms = {
            # "loss": recon_loss + consistency_loss,
            "loss": recon_loss,
            "consistency_loss": consistency_loss,
            "recon_loss": recon_loss,
        }
        return terms

    def denoise(self, model, x_t, sigmas, state):
        # Compute scaling factors for boundary condition.
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        # Rescale sigma (time) for model input.
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, state)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return model_output, denoised

    def sample(self, model, state, num=10):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(model, state, num=num)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(model, state)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        return x_0
    
    def sample_onestep(self, model, state, num=1000):
        if state is not None:
            x_T = th.randn((state.shape[0], self.horizon, self.action_dim), device=self.device) * self.sigma_max
            x_T = x_T.to(self.device)
            s_in = x_T.new_ones([x_T.shape[0]])
            return self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
        else:
            x_T = th.randn((num, self.horizon, self.action_dim), device=self.device) * self.sigma_max
            s_in = x_T.new_ones([x_T.shape[0]])
            return self.denoise(model, x_T, self.sigmas[0] * s_in, None)[1]
    
    def sample_multistep(self, model, state, num=1000):
        if state is not None:
            x_T = th.randn((state.shape[0], self.horizon, self.action_dim), device=self.device) * self.sigma_max
            t_max_rho = self.sigma_max ** (1 / self.rho)
            t_min_rho = self.sigma_min ** (1 / self.rho)
            s_in = x_T.new_ones([x_T.shape[0]])
            x = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
            for i in range(len(self.ts)):
                t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
                t = np.clip(t, self.sigma_min, self.sigma_max)
                x = x + th.randn_like(x) * np.sqrt(t**2 - self.sigma_min**2)
                x = self.denoise(model, x, t * s_in, state)[1]
            return x
        else:
            x_T = th.randn((num, self.horizon, self.action_dim), device=self.device) * self.sigma_max
            t_max_rho = self.sigma_max ** (1 / self.rho)
            t_min_rho = self.sigma_min ** (1 / self.rho)
            s_in = x_T.new_ones([x_T.shape[0]])
            x = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
            for i in range(len(self.ts)):
                t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
                t = np.clip(t, self.sigma_min, self.sigma_max)
                x = x + th.randn_like(x) * np.sqrt(t**2 - self.sigma_min**2)
                x = self.denoise(model, x, t * s_in, state)[1]
            return x

# ================================================================================
# END NEW: Consistency Model Implementation
# ================================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

#-----------------------------------------------------------------------------#
#------------------------------ neural networks-------------------------------#
#-----------------------------------------------------------------------------#

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])