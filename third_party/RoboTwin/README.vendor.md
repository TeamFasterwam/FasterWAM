# Vendored RoboTwin

This directory vendors code from the upstream RoboTwin repository:

- Upstream project: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT License

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboTwin remain subject to the MIT License in this directory.
- For this project, the locally maintained [policy adapter](../../experiments/robotwin/fasterwam_policy) lives outside the vendored benchmark.
- The adapter contains project-specific code and is not copied from the other upstream `policy/*` subprojects.
- Evaluation creates `RoboTwin/policy/fasterwam_policy` as a symlink to the adapter in the Faster-WAM repository.
- If code is later copied from any upstream subdirectory with an additional license notice, the corresponding license file and attribution must also be preserved.

Local modifications:

- RoboTwin is vendored under `third_party/RoboTwin` for easier integration with this project.
- The checkout retains the simulator, fixed benchmark tasks and configurations,
  asset installation tools, and the evaluation runner used by Faster-WAM.
- Unused upstream policy implementations, code generation, data collection and
  export, asset authoring, offline description generation, and socket-based
  policy serving tools are omitted. Use the upstream repository for those workflows.
- Evaluation still generates episode instructions from the retained task and
  object description JSON files through
  `description/utils/generate_episode_instructions.py`.
- `script/requirements.txt` is replaced by the simulator package versions used
  for the Faster-WAM results. `script/_install.sh` is removed; Faster-WAM's
  `scripts/setup_env.sh robotwin` installs these packages and CuRobo v0.7.8
  without the upstream mplib planner patch.
- `script/eval_policy.py` includes the Faster-WAM evaluation options used by the
  bundled runner. See [README_Robotwin.md](../../README_Robotwin.md).
