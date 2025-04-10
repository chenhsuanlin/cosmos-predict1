# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict1.diffusion.training.models.extend_model import FSDPExtendDiffusionModel
from cosmos_predict1.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import PLACEHOLDER
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


num_frames = 57
dl3dv_dataseet = L(DL3DVDataset)(
    dataset_dir="datasets/dl3dv-10k",
    num_frames=num_frames,
    video_size=(720, 1280),
)
dataloader_train = L(DataLoader)(
    dataset=dl3dv_dataseet,
    sampler=L(get_sampler)(dataset=dl3dv_dataseet),
    batch_size=1,
    drop_last=True,
)


video2world_camera_7b_example = LazyDict(
    dict(
        defaults=[
            {"override /conditioner": "video_cond"},
            {"override /data_train": "data_dl3dv_57frames"},
            "_self_",
        ],
        model=dict(
            fsdp=dict(
                sharding_group_size=128,
            ),
            ema=dict(
                enabled=True,
                num=1,
            ),
            conditioner=dict(
                video_cond_bool=dict(
                    add_pose_condition=True,
                    condition_location="first_random_n",
                    cfg_unconditional_type="zero_condition_region_condition_mask",
                    apply_corruption_to_condition_region="noise_with_sigma_fixed",
                    first_random_n_num_condition_t_max=1,
                    first_random_n_num_condition_t_min=1,
                    dropout_rate=0.0,
                    augment_sigma_sample_p_mean=-3.0,
                    augment_sigma_sample_p_std=2.0,
                    augment_sigma_sample_multiplier=1.0,
                ),
                text=dict(
                    dropout_rate=0.2,
                ),
            ),
            net=L(VideoExtendGeneralDIT)(
                in_channels=23,
                extra_per_block_abs_pos_emb=True,
                pos_emb_learnable=True,
                extra_per_block_abs_pos_emb_type="learnable",
            ),
            vae=dict(
                video_vae=dict(
                    pixel_chunk_duration=57,
                ),
            ),
            adjust_video_noise=True,
            latent_shape=[
                16,
                8,
                88,
                160,
            ],
            context_parallel_size=1,
        ),
        trainer=dict(
            max_iter=2000,
        ),
        dataloader_train=dataloader_train,
        checkpoint=dict(
            save_iter=200,
            save_to_object_store=dict(
                enabled=False,
            ),
            load_from_object_store=dict(
                enabled=False,
            ),
            strict_resume=False,
            load_path="checkpoints/Cosmos-Predict1-7B-Video2World_camera/model.pt",
            load_training_state=False,
        ),
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_camera_7b_example",
        ),
        # using the video extend model for training
        model_obj=L(FSDPExtendDiffusionModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
    )
)


def register_experiments(cs):
    # Register the experiments
    for _item in [
        video2world_camera_7b_example,
    ]:
        experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
        _item["job"]["name"] = experiment_name
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
