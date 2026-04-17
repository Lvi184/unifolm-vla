import os

os.environ["NO_GCE_CHECK"] = "true"
os.environ["TFDS_DISABLE_GCS"] = "1"

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

# 双保险：直接禁用 TFDS 的 GCS 逻辑
gcs_utils._is_gcs_disabled = True

from rlds_dataset import rlds_dataset

builder = rlds_dataset(
    data_dir="/root/gpufree-data/AgiBotWorldChallenge-2026/unifolm_hdf5/unifolm_rlds/"
)

download_config = tfds.download.DownloadConfig(
    try_download_gcs=False,
    max_examples_per_split=10,
)

builder.download_and_prepare(download_config=download_config)
print("done")