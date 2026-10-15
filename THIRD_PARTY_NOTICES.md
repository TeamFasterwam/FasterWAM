# Third-party notices

Required Notice: Copyright (c) 2026 The Faster-WAM Authors

The [PolyForm Noncommercial License 1.0.0](LICENSE) applies to Faster-WAM's
original contributions. It does not replace the licenses of third-party code.

Faster-WAM is built on the Fast-WAM codebase and includes code adapted from the
projects below. Each component remains subject to its original license.

| Component | Upstream | License | Location in this repository |
| --- | --- | --- | --- |
| Fast-WAM | https://github.com/yuantianyuan01/FastWAM | MIT ([text](licenses/MIT-FastWAM.txt)) | Code base for training, data processing, the Fast-WAM baseline and evaluation |
| Wan2.2 | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | `src/fasterwam/models/wan22/` (video DiT, VAE, T5 text encoder, scheduler) |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | `src/fasterwam/models/wan22/helpers/` (model download, loading and state-dict conversion) |
| LeRobot | https://github.com/huggingface/lerobot | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | `src/fasterwam/datasets/lerobot/constants.py`, `src/fasterwam/datasets/lerobot/lerobot/` |
| NVIDIA utilities | — | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | `src/fasterwam/utils/misc.py` |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | BSD-3-Clause ([text](licenses/BSD-3-Clause-PyTorch3D.txt)) | `src/fasterwam/datasets/lerobot/transforms/rotation.py` |
| OpenVLA LIBERO utilities | https://github.com/openvla/openvla | MIT ([text](licenses/MIT-OpenVLA.txt)) | `experiments/libero/libero_utils.py` |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin | MIT ([text](third_party/RoboTwin/LICENSE)) | `third_party/RoboTwin/` (see its [vendor notice](third_party/RoboTwin/README.vendor.md)) |

Files adapted from Apache-2.0 projects were modified for this repository.

Model weights are not distributed with the code. The Wan2.2-TI2V-5B backbone,
VAE and T5 text encoder are downloaded from their model repositories and are
subject to their own licenses. LIBERO, LIBERO-Plus and their assets, and the
RoboTwin simulator assets, are installed separately under their own licenses.
