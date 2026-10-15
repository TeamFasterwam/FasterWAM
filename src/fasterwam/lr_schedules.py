"""Warmup-Stable-Decay (WSD) learning-rate schedules.

Two ways to use this module:

1. End-to-end WSD run (``lr_scheduler_type=wsd``): linear warmup, a constant
   plateau at the peak lr, then a short decay to the floor over the last
   ``decay_steps`` of the run. Built by :func:`build_wsd_schedule`.

2. Decay branch (``wsd_branch_decay_steps=N`` together with
   ``resume=<state_dir>``): a stable-phase checkpoint is restored (weights,
   optimizer, dataloader progress), its saved lr-scheduler state is ignored,
   and the lr decays from its current value to the floor over exactly ``N``
   optimizer steps. This turns one long stable run into several finished
   "trained for K steps" models at the cost of ``N`` steps each, instead of one
   full cosine run per training length.

:class:`WSDDecayLR` is the decay leg shared by both paths.
"""

from __future__ import annotations

import logging
import math

from torch.optim.lr_scheduler import ConstantLR, LinearLR, LRScheduler, SequentialLR

logger = logging.getLogger(__name__)

WSD_DECAY_SHAPES = ("linear", "1-sqrt", "cosine")


def wsd_decay_factor(step: int, decay_steps: int, shape: str) -> float:
    """Return the decay multiplier in [0, 1] at ``step`` of ``decay_steps``.

    1.0 at step 0, 0.0 at ``decay_steps`` and beyond. ``shape`` selects the
    path between them: ``linear``, ``1-sqrt`` (the WSD default from Hägele et
    al. 2024, which stays high longer), or ``cosine`` (half-period).
    """
    decay_steps = max(int(decay_steps), 1)
    t = min(max(float(step), 0.0), float(decay_steps)) / float(decay_steps)
    if shape == "linear":
        return 1.0 - t
    if shape == "1-sqrt":
        return 1.0 - math.sqrt(t)
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"Unsupported WSD decay shape: {shape!r}. Expected one of {WSD_DECAY_SHAPES}.")


class WSDDecayLR(LRScheduler):
    """Decay each param group from ``start_lrs`` to ``end_lr`` over ``decay_steps``.

    ``start_lrs`` defaults to the optimizer's ``initial_lr`` (the peak lr), which
    is what the end-to-end WSD schedule wants. A decay branch instead calls
    :meth:`rebase` after the checkpoint is restored so the decay starts from the
    lr the stable run was actually using.

    The saved state carries a marker key; :meth:`load_state_dict` ignores a
    state dict without it. That is what lets a branch run restore a full
    accelerate state directory written by a run with a different scheduler
    (SequentialLR of warmup/cosine or warmup/constant) without that foreign
    state being dumped into this object's ``__dict__``.
    """

    STATE_MARKER = "wsd_decay_lr"

    def __init__(
        self,
        optimizer,
        decay_steps: int,
        end_lr: float,
        shape: str = "linear",
        start_lrs: list[float] | None = None,
        last_epoch: int = -1,
    ):
        if int(decay_steps) < 1:
            raise ValueError(f"decay_steps must be >= 1, got {decay_steps}")
        if shape not in WSD_DECAY_SHAPES:
            raise ValueError(f"Unsupported WSD decay shape: {shape!r}. Expected one of {WSD_DECAY_SHAPES}.")
        self.decay_steps = int(decay_steps)
        self.end_lr = float(end_lr)
        self.shape = str(shape)
        self._explicit_start_lrs = [float(lr) for lr in start_lrs] if start_lrs is not None else None
        # Populated when load_state_dict() is handed a foreign scheduler state;
        # rebase() prefers it over the optimizer's current lr because it is the
        # lr the previous run last applied, independent of what the optimizer
        # checkpoint restored into param_groups.
        self.foreign_last_lr: list[float] | None = None
        super().__init__(optimizer, last_epoch)

    @property
    def start_lrs(self) -> list[float]:
        if self._explicit_start_lrs is not None:
            return list(self._explicit_start_lrs)
        return list(self.base_lrs)

    def get_lr(self):
        factor = wsd_decay_factor(self.last_epoch, self.decay_steps, self.shape)
        return [self.end_lr + (start - self.end_lr) * factor for start in self.start_lrs]

    # SequentialLR calls ``step(0)`` at a milestone, which prefers the closed
    # form when present. The schedule is already a closed form of last_epoch.
    def _get_closed_form_lr(self):
        return self.get_lr()

    def rebase(self, start_lrs: list[float], decay_steps: int | None = None) -> None:
        """Restart the decay at step 0 from ``start_lrs``.

        Applies the step-0 lr to the optimizer immediately, so the first
        optimizer step after a branch resume already runs at ``start_lrs``.
        """
        groups = self.optimizer.param_groups
        if len(start_lrs) != len(groups):
            raise ValueError(
                f"rebase expects one start lr per param group ({len(groups)}), got {len(start_lrs)}"
            )
        if decay_steps is not None:
            if int(decay_steps) < 1:
                raise ValueError(f"decay_steps must be >= 1, got {decay_steps}")
            self.decay_steps = int(decay_steps)
        self._explicit_start_lrs = [float(lr) for lr in start_lrs]
        self.last_epoch = 0
        values = self.get_lr()
        for group, lr in zip(groups, values):
            group["lr"] = lr
        self._last_lr = list(values)

    def state_dict(self):
        state = super().state_dict()
        state[self.STATE_MARKER] = True
        return state

    def load_state_dict(self, state_dict):
        if not state_dict.get(self.STATE_MARKER, False):
            foreign_last_lr = state_dict.get("_last_lr")
            self.foreign_last_lr = (
                [float(lr) for lr in foreign_last_lr] if foreign_last_lr is not None else None
            )
            logger.warning(
                "WSDDecayLR ignored a foreign lr-scheduler state (keys=%s, last_lr=%s); "
                "the decay schedule will be rebased after the checkpoint is restored.",
                sorted(k for k in state_dict.keys() if k != "_schedulers"),
                self.foreign_last_lr,
            )
            return
        state = dict(state_dict)
        state.pop(self.STATE_MARKER, None)
        super().load_state_dict(state)


def build_wsd_schedule(
    optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    end_lr: float,
    shape: str = "linear",
) -> LRScheduler:
    """Warmup -> constant peak -> decay, spanning ``total_steps`` optimizer steps.

    Milestones are absolute step counts: warmup occupies ``[0, warmup_steps)``,
    the plateau ``[warmup_steps, total_steps - decay_steps)``, and the decay the
    final ``decay_steps``. The peak lr is each param group's ``initial_lr``.
    """
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    decay_steps = int(decay_steps)
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if warmup_steps < 0 or decay_steps < 1:
        raise ValueError(
            f"warmup_steps must be >= 0 and decay_steps >= 1, got {warmup_steps} and {decay_steps}"
        )
    stable_steps = total_steps - warmup_steps - decay_steps
    if stable_steps < 1:
        raise ValueError(
            "WSD needs a non-empty stable phase: "
            f"total_steps={total_steps} warmup_steps={warmup_steps} decay_steps={decay_steps}"
        )

    stable = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
    decay = WSDDecayLR(optimizer, decay_steps=decay_steps, end_lr=end_lr, shape=shape)
    if warmup_steps <= 0:
        return SequentialLR(optimizer, schedulers=[stable, decay], milestones=[stable_steps])

    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, stable, decay],
        milestones=[warmup_steps, warmup_steps + stable_steps],
    )
