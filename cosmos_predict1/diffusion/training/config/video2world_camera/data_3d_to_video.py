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

from cosmos_predict1.utils.lazy_config import LazyCall as L
from projects.cosmos.diffusion.v1.datasets.joint_dataloader import IterativeJointDataLoader


# function to load real video data (dl3dv) for training
def get_dataloader_real_video(
    num_video_frames: int = 57,
    num_workers: int = 8,
    batch_size: int = None,
    # Other parameters
    image_size: int = 256,
    use_fps_control=False,
    object_store="s3",
):
    # Other config
    num_view = num_video_frames
    load_keys = [
        "ai_caption_long",
        "t5_text_embedings_long",
        "video",
        "camera",
    ]
    augmentation = {
        "camera": L(EstimateCameraSampling)(
            num_view=num_view,
            image_size_hw=(image_size, image_size),
            latent_compression_ratio_h=8,
            latent_compression_ratio_w=8,
        ),
    }
    if batch_size is None:
        batch_size = 1
    distributor, detshuffle = get_distributor(is_train=True)

    # Note: in video dataset, the dataset are constructed in projects/cosmos/diffusion/v1/datasets/dataset_provider.py get_video_dataset and get_image_dataset
    # The dataloader is constructed in projects/cosmos/diffusion/v1/config/base/data.py, get_video_loader and get_image_loader

    # following video_dataloader_native_fps to set up the decoder
    video_decoder_name = "video_decoder_w_controlled_fps"
    chunk_size = 0
    use_fps_control = use_fps_control
    min_fps_thres: int = 12
    max_fps_thres: int = 40
    sampling_reweighting: bool = False
    sampling_reweighting_factor: float = 2.0
    limit_fps_range: bool = False

    video_dataset = L(projects.cosmos.diffusion.v1.datasets.webdataset.Dataset)(
        config=L(schema_webdataset.DatasetConfig)(
            keys=load_keys,
            buffer_size=16,  # This shuffle happens on the raw sample level
            streaming_download=True,
            dataset_info=dataset_info[object_store],
            distributor=distributor,
            decoders=[
                imaginaire.datasets.webdataset.decoders.pickle.pkl_decoder,
                projects.cosmos.diffusion.v1.datasets.decoders.video_decoder.construct_video_decoder(
                    video_decoder_name=video_decoder_name,
                    sequence_length=num_video_frames,
                    chunk_size=chunk_size,
                    use_fps_control=use_fps_control,
                    min_fps_thres=min_fps_thres,
                    max_fps_thres=max_fps_thres,
                    sampling_reweighting=sampling_reweighting,
                    sampling_reweighting_factor=sampling_reweighting_factor,
                    limit_fps_range=limit_fps_range,
                ),
            ],
            augmentation=augmentation,  # Change it for RM if needed @Max
            remove_extension_from_keys=False,
            # sample_keys_full_list_path="index",
        ),
        decoder_handler=warn_and_continue,
        detshuffle=detshuffle,
    )
    video_dataloader = L(imaginaire.datasets.webdataset.dataloader.DataLoader)(
        dataset=video_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=None,
        sampler=None,
    )
    return video_dataloader


def register_training_and_val_3d_data(cs):
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="data_dl3dv_57frames",
        node=get_dataloader_real_video(use_fps_control=False, num_video_frames=57, object_store_video="s3"),
    )
