"""LIBERO environment that renders camera observations only when policies need them."""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


@dataclass(frozen=True)
class SelectiveRenderSchedule:
    """Choose which control steps require fresh camera observations."""

    initial_wait_steps: int
    replan_steps: int
    render_steps_within_replan: frozenset[int]

    def __post_init__(self) -> None:
        assert self.initial_wait_steps >= 0, "initial_wait_steps must be >= 0"
        assert self.replan_steps > 0, "replan_steps must be > 0"
        assert self.replan_steps in self.render_steps_within_replan, (
            "The final action in every replan window must render the next policy observation."
        )
        assert all(1 <= step <= self.replan_steps for step in self.render_steps_within_replan), (
            "render_steps_within_replan must be in [1, replan_steps]"
        )

    def should_render(self, control_step: int) -> bool:
        """Return whether the post-action observation at ``control_step`` needs fresh pixels."""
        assert control_step > 0, "control_step is one-indexed and must be > 0"
        if control_step <= self.initial_wait_steps:
            return control_step == self.initial_wait_steps

        rollout_step = control_step - self.initial_wait_steps
        step_within_replan = (rollout_step - 1) % self.replan_steps + 1
        return step_within_replan in self.render_steps_within_replan


class SelectiveRenderOffScreenEnv(OffScreenRenderEnv):
    """Run every physics step while caching pixels between selected render steps."""

    def __init__(
        self,
        *,
        initial_wait_steps: int,
        replan_steps: int,
        render_steps_within_replan: Collection[int],
        **kwargs: Any,
    ) -> None:
        render_steps = frozenset(int(step) for step in render_steps_within_replan)
        self.render_schedule = SelectiveRenderSchedule(
            initial_wait_steps=int(initial_wait_steps),
            replan_steps=int(replan_steps),
            render_steps_within_replan=render_steps,
        )
        self.control_step = 0
        self.rendered_control_steps = 0
        self.skipped_control_steps = 0
        self.last_step_rendered = True
        self._camera_observable_names: tuple[str, ...] = ()
        self._last_rendered_images: dict[str, np.ndarray] = {}
        super().__init__(**kwargs)
        self._refresh_camera_observable_names()

    def _refresh_camera_observable_names(self) -> None:
        observables = self.env._observables
        self._camera_observable_names = tuple(
            name for name, observable in observables.items() if observable.modality == "image"
        )
        assert self._camera_observable_names, "LIBERO environment has no image observables to render"

    def _set_camera_observables_enabled(self, enabled: bool) -> None:
        for name in self._camera_observable_names:
            observable = self.env._observables[name]
            # Observable.set_enabled() resets its sampling clock. Preserve the
            # clock so scheduled frames are sampled at the baseline instant.
            observable._enabled = enabled

    def _cache_rendered_images(self, observation: dict[str, Any]) -> None:
        missing = [name for name in self._camera_observable_names if name not in observation]
        assert not missing, f"Fresh LIBERO observation is missing camera fields: {missing}"
        self._last_rendered_images = {
            name: np.array(observation[name], copy=True) for name in self._camera_observable_names
        }

    def _merge_cached_images(self, observation: dict[str, Any]) -> dict[str, Any]:
        assert self._last_rendered_images, "No rendered observation is available for pixel caching"
        merged = observation.copy()
        for name, image in self._last_rendered_images.items():
            merged[name] = np.array(image, copy=True)
        return merged

    def _consume_skipped_plus_noise_transform(self) -> None:
        """Advance LIBERO-Plus sensor-noise RNG without rendering a fresh frame."""
        from PIL import Image
        from libero.libero.envs import env_wrapper as plus_env_wrapper

        noise_level = int(getattr(self, "noise", 0))
        assert 1 <= noise_level <= 50, f"Unsupported LIBERO-Plus noise level: {noise_level}"
        agent_view = np.array(self._last_rendered_images["agentview_image"], copy=True)
        if agent_view.dtype != np.uint8:
            agent_view = (agent_view * 255).astype(np.uint8)
        image = Image.fromarray(agent_view)
        noise_functions = (
            plus_env_wrapper.motion_blur,
            plus_env_wrapper.gaussian_blur,
            plus_env_wrapper.zoom_blur,
            plus_env_wrapper.fog,
            plus_env_wrapper.glass_blur,
        )
        noise_function = noise_functions[(noise_level - 1) // 10]
        noise_function(image, severity=(noise_level - 1) % 10 + 1)

    def _reset_render_state(self, observation: dict[str, Any]) -> None:
        self.control_step = 0
        self.rendered_control_steps = 0
        self.skipped_control_steps = 0
        self.last_step_rendered = True
        self._cache_rendered_images(observation)

    def reset(self) -> dict[str, Any]:
        """Reset the simulator with camera observables enabled."""
        if hasattr(self, "env"):
            self._refresh_camera_observable_names()
            self._set_camera_observables_enabled(True)
        observation = super().reset()
        self._refresh_camera_observable_names()
        self._set_camera_observables_enabled(True)
        self._reset_render_state(observation)
        return observation

    def set_init_state(self, init_state: np.ndarray) -> dict[str, Any]:
        """Set an official LIBERO initial state and cache its fresh camera observation."""
        self._refresh_camera_observable_names()
        self._set_camera_observables_enabled(True)
        observation = super().set_init_state(init_state)
        self._reset_render_state(observation)
        return observation

    def step(self, action: np.ndarray | list[float]) -> tuple[dict[str, Any], float, bool, dict]:
        """Advance physics and render only at scheduled observation boundaries."""
        self.control_step += 1
        should_render = self.render_schedule.should_render(self.control_step)

        self._set_camera_observables_enabled(should_render)
        noise_level = int(getattr(self, "noise", 0))
        if noise_level != 0 and not should_render:
            # LIBERO-Plus applies sensor noise inside ControlEnv.step and assumes
            # agentview_image exists. Advance only the wrapped simulator, then
            # apply the same transform to cached pixels and discard its output.
            # This preserves the baseline NumPy RNG trajectory without rendering.
            observation, reward, done, info = self.env.step(action)
            self._consume_skipped_plus_noise_transform()
        else:
            observation, reward, done, info = super().step(action)

        if should_render:
            self._cache_rendered_images(observation)
            self.rendered_control_steps += 1
        else:
            observation = self._merge_cached_images(observation)
            self.skipped_control_steps += 1

        self.last_step_rendered = should_render
        return observation, reward, done, info


def get_selective_render_libero_env(
    task: Any,
    resolution: int,
    seed: int,
    env_num: int = 1,
    *,
    initial_wait_steps: int,
    replan_steps: int,
    render_steps_within_replan: Collection[int],
) -> tuple[SelectiveRenderOffScreenEnv, str]:
    """Create a single-process LIBERO environment with selective camera rendering."""
    assert env_num == 1, "Selective LIBERO rendering currently supports env_num=1"
    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = SelectiveRenderOffScreenEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=int(resolution),
        camera_widths=int(resolution),
        initial_wait_steps=initial_wait_steps,
        replan_steps=replan_steps,
        render_steps_within_replan=render_steps_within_replan,
    )
    env.seed(seed)
    return env, task.language


def build_selective_render_env_factory(cfg: Any) -> Callable[..., tuple[Any, str]]:
    """Build a drop-in replacement for ``eval_libero_single.get_libero_env``."""
    initial_wait_steps = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    assert replan_steps <= action_horizon, (
        f"replan_steps={replan_steps} exceeds action_horizon={action_horizon}; "
        "selective rendering cannot infer the shorter action-queue boundary"
    )
    render_steps = {replan_steps}

    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
        render_steps.update(range(action_video_freq_ratio, replan_steps + 1, action_video_freq_ratio))

    return functools.partial(
        get_selective_render_libero_env,
        initial_wait_steps=initial_wait_steps,
        replan_steps=replan_steps,
        render_steps_within_replan=render_steps,
    )
