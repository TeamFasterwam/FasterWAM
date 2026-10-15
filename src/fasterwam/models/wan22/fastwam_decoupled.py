"""Decoupled video and action experts with independently configured layer counts.

The video pass produces per-layer keys and values. Action layers consume selected
or fused video features through the configured readout strategy, with visibility
restricted to the observed first frame. Training and cached inference share the
same attention interfaces."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from fasterwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


def _resolve_slot_seeds(
    seed: Optional[Union[int, Sequence[int]]],
    *,
    batch_size: int,
) -> Optional[list[int]]:
    """Normalize a batched ``seed`` argument into one seed per batch slot.

    Returns ``None`` when the caller passed no seed, which means "draw from the
    ambient RNG". A scalar broadcasts to every slot, reproducing the historical
    behaviour where a whole batch -- and, because the eval server rebuilds the
    generator on every request, a whole evaluation run -- shares one noise draw.
    A sequence assigns slot ``i`` its own seed.
    """
    if seed is None:
        return None
    if isinstance(seed, (int, torch.Tensor)) or not isinstance(seed, Sequence):
        return [int(seed)] * batch_size
    slot_seeds = [int(value) for value in seed]
    if len(slot_seeds) != batch_size:
        raise ValueError(
            f"`seed` sequence has length {len(slot_seeds)} but the batch has "
            f"{batch_size} slots; pass one seed per slot or a single scalar."
        )
    return slot_seeds


class FastWAMDecoupled(FastWAM):
    """Train and infer decoupled video/action experts.

    Action layers consume final-layer or fused video K/V with first-frame
    visibility. Both single-request and batched inference reuse a video cache."""

    # The shared-model server can batch requests through the public cache path.
    supports_batched_action_inference: bool = True

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build separate attention masks for the decoupled two-phase forward.

        Returns a tuple of two masks instead of a single joint mask:
        - ``video_mask``: ``[Sv, Sv]`` -- video self-attention (same as baseline).
        - ``action_mask``: ``[Sa, Sv + Sa]`` -- action rows only. First-frame
          video columns are True, remaining video columns False, all action
          columns True (paradigm C topology).

        Args:
            video_seq_len: Number of video tokens ``Sv``.
            action_seq_len: Number of action tokens ``Sa``.
            video_tokens_per_frame: Number of video tokens per frame. Used to
                determine which columns correspond to the first frame.
            device: Device for the mask tensors.

        Returns:
            Tuple of ``(video_mask, action_mask)``.
        """
        # Video self-attention mask: [Sv, Sv]
        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # Action attention mask: [Sa, Sv + Sa]
        # Columns: [video_first_frame | video_other_frames | action]
        action_mask = torch.zeros(
            (action_seq_len, video_seq_len + action_seq_len),
            dtype=torch.bool,
            device=device,
        )
        # Action can see first-frame video tokens.
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        action_mask[:, :first_frame_tokens] = True
        # Action can see all action tokens (full self-attention).
        action_mask[:, video_seq_len:] = True

        return video_mask, action_mask

    def training_loss(self, sample: dict, tiled: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss using the decoupled two-phase forward pass.

        Same loss structure as the baseline ``FastWAM.training_loss()`` but uses
        ``self.mot.forward_decoupled()`` instead of ``self.mot()`` and passes
        separate video/action masks.

        Args:
            sample: Training sample dict from the dataloader.
            tiled: Whether to use tiled VAE encoding.

        Returns:
            Tuple of ``(loss_total, loss_dict)`` where ``loss_dict`` has keys
            ``"loss_video"`` and ``"loss_action"``.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        freeze_video_backbone = bool(
            getattr(self, "freeze_video_backbone_for_training", False)
        )

        # Sample action noise and timestep (independent from video).
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        if freeze_video_backbone:
            first_frame_latents = inputs["first_frame_latents"]
            if first_frame_latents is None:
                raise ValueError(
                    "`freeze_video_backbone_for_training` requires first-frame "
                    "conditioning latents."
                )

            timestep_video = torch.zeros(
                (batch_size,),
                device=self.device,
                dtype=first_frame_latents.dtype,
            )
            with torch.no_grad():
                video_pre = self.video_expert.pre_dit(
                    x=first_frame_latents,
                    timestep=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                )
                video_mask = self.video_expert.build_video_to_video_mask(
                    video_seq_len=video_pre["tokens"].shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=video_pre["tokens"].device,
                )
                video_kv_per_layer = self.mot.prefill_video_kv(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload={
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    video_attention_mask=video_mask,
                )

            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )

            _, action_mask = self._build_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=action_pre["tokens"].device,
            )
            action_tokens_out = self.mot.forward_action_with_video_kv(
                video_kv_per_layer=video_kv_per_layer,
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload=self.action_expert.build_cross_attn_payload(
                    action_pre
                ),
                action_attention_mask=action_mask,
            )
            pred_action = self.action_expert.post_dit(action_tokens_out, action_pre)

            action_loss_token = F.mse_loss(
                pred_action.float(), target_action.float(), reduction="none"
            ).mean(dim=2)
            if action_is_pad is not None:
                valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)

            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
            )
            loss_action = (action_loss_per_sample * action_weight).mean()
            loss_total = self.loss_lambda_action * loss_action
            return loss_total, {
                "loss_video": 0.0,
                "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            }

        # Sample video noise and timestep.
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        # Replace first frame with clean latent (conditioning).
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Pre-dit: embed tokens, compute RoPE, time modulation, context.
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )


        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        # Build separate masks for the decoupled forward.
        video_mask, action_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )

        # Video forward captures all KV layers, then action mixes them per layer.
        # CRITICAL: Use forward_decoupled(), NOT self.mot() which raises NotImplementedError.
        tokens_out = self.mot.forward_decoupled(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_masks={
                "video": video_mask,
                "action": action_mask,
            },
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": self.action_expert.build_cross_attn_payload(action_pre),
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # Post-dit: project tokens back to prediction space.
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # Strip first-frame prediction (not a denoising target when fusing).
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        # Video loss: per-sample MSE weighted by scheduler training weight.
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # Action loss: per-token MSE with pad masking.
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)  # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_video_kv(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict action velocity using all video-layer K/V.

        Args:
            latents_action: Noisy action latents ``[B, Sa, action_dim]``.
            timestep_action: Action timestep ``[B]``.
            context: Text/proprio context ``[B, L, D]``.
            context_mask: Context attention mask ``[B, L]``.
            video_kv_per_layer: List of per-layer video K/V, each a dict
                ``{"k": [B, Sv, H*Dh], "v": [B, Sv, H*Dh]}``. The list length
                is ``video_num_layers`` for the raw RoPE-aware cache, or
                ``action_num_layers`` after selection/fusion has been prepared.
            action_attention_mask: Rectangular mask ``[Sa, Sv + Sa]``.

        Returns:
            Predicted action velocity ``[B, Sa, action_dim]``.
        """
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_kv(
            video_kv_per_layer=video_kv_per_layer,
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload=self.action_expert.build_cross_attn_payload(
                action_pre
            ),
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _predict_action_noise_with_final_kv(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        final_video_kv: dict[str, torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compat wrapper. Prefer ``_predict_action_noise_with_video_kv()`` instead."""
        if isinstance(final_video_kv, dict):
            video_kv_per_layer = [final_video_kv] * self.mot.video_num_layers
        else:
            video_kv_per_layer = final_video_kv
        return self._predict_action_noise_with_video_kv(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_per_layer=video_kv_per_layer,
            action_attention_mask=action_attention_mask,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Infer actions using the decoupled KV cache path.

        Prefills the video expert once to obtain all video-layer K/V, then
        denoises actions for
        ``num_inference_steps`` Euler steps using the cached K/V.

        Args:
            prompt: Text prompt (mutually exclusive with context/context_mask).
            input_image: Conditioning image ``[1, 3, H, W]`` or ``[3, H, W]``.
            action_horizon: Number of action steps to predict.
            proprio: Optional proprioceptive state ``[D]`` or ``[1, D]``.
            context: Precomputed text context ``[B, L, D]``.
            context_mask: Context attention mask ``[B, L]``.
            negative_prompt: Unused (kept for API compatibility).
            text_cfg_scale: Unused.
            num_inference_steps: Number of Euler denoising steps.
            sigma_shift: Optional override for inference sigma shift.
            seed: Random seed for action noise.
            rand_device: Device to use for random number generation.
            tiled: Whether to use tiled VAE encoding.

        Returns:
            Dict ``{"action": tensor}`` with action shape ``[action_horizon, action_dim]``.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        # --- Input validation (same as baseline) ---
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Initialize action noise ---
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        vae_ms = _sync_ms(t0)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        text_enc_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        proprio_ms = _sync_ms(t0)

        result = self._infer_action_core(
            first_frame_latents=first_frame_latents,
            latents_action=latents_action,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
        )
        # Public single-request API returns ``[H, D]``.
        result["action"] = result["action"][0]

        core_timing = result.get("timing_ms", {})
        result["timing_ms"] = {
            "text_enc_ms": text_enc_ms,
            "proprio_ms": proprio_ms,
            "vae_ms": vae_ms,
            "video_prefill_ms": core_timing.get("video_prefill_ms", 0.0),
            "denoise_ms": core_timing.get("denoise_ms", 0.0),
            "num_denoise_steps": num_inference_steps,
        }
        return result

    def _encode_input_images_latents_batch(
        self,
        input_image: torch.Tensor,
        *,
        tiled: bool = False,
    ) -> torch.Tensor:
        """Encode a batch of conditioning images to first-frame VAE latents.

        Args:
            input_image: ``[B, 3, H, W]`` in model input scale.
            tiled: Tiled VAE encoding is not supported for B>1.

        Returns:
            Latents with shape ``[B, C, T, H', W']`` (T is typically 1).
        """
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [B,3,H,W], got {tuple(input_image.shape)}"
            )
        batch_size = int(input_image.shape[0])
        if batch_size < 1:
            raise ValueError("`input_image` batch size must be >= 1.")
        if tiled and batch_size > 1:
            raise ValueError(
                "tiled=True is not supported for batched image encoding; "
                "use infer_action() per request instead."
            )
        if batch_size == 1:
            return self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )

        images = [
            input_image[index].to(device=self.device).unsqueeze(1)
            for index in range(batch_size)
        ]
        latents = self.vae.encode(
            images,
            device=self.device,
            tiled=False,
        )
        if isinstance(latents, list):
            latents = torch.stack(latents, dim=0)
        return latents

    @torch.no_grad()
    def infer_action_batch(
        self,
        input_image: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[Union[int, Sequence[int]]] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        """Batched action inference for the shared-model LIBERO server.

        Contract mirrors ``infer_action`` but accepts a leading batch axis on
        every tensor input and returns ``action`` with shape
        ``[B, action_horizon, action_dim]``.

        ``seed`` accepts either form, and both are slot-invariant -- a request's
        noise never depends on which slot it occupies in a larger batch:

        - **scalar**: every slot reseeds an independent generator with that same
          value, so slot ``i`` matches a solo ``infer_action`` call with the same
          seed. This means the whole batch shares one noise draw.
        - **sequence** of length ``B``: slot ``i`` reseeds from ``seed[i]``, so
          each request gets its own draw. Use this to sample noise per inference
          instead of freezing one draw for a whole evaluation run.

        Args:
            input_image: Conditioning images ``[B, 3, H, W]``.
            action_horizon: Number of action steps to predict.
            context: Precomputed text context ``[B, L, D]``.
            context_mask: Context attention mask ``[B, L]``.
            proprio: Optional proprioceptive state ``[B, D]``.
            num_inference_steps: Number of Euler denoising steps.
            sigma_shift: Optional override for inference sigma shift.
            seed: Action-noise seed, scalar or per-slot sequence (see above).
            rand_device: Device to use for random number generation.
            tiled: Whether to use tiled VAE encoding (B==1 only).
            **unused_kwargs: Ignored extras from the shared-model server.

        Returns:
            Dict ``{"action": tensor}`` with shape ``[B, action_horizon, action_dim]``.
        """
        del unused_kwargs
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action_batch` requires `video_attention_mask_mode='first_frame_causal'`."
            )
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [B,3,H,W], got {tuple(input_image.shape)}"
            )
        batch_size = int(input_image.shape[0])
        if batch_size < 1:
            raise ValueError("`input_image` batch size must be >= 1.")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if context is None or context_mask is None:
            raise ValueError(
                "`infer_action_batch` requires precomputed `context` and `context_mask`."
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], "
                f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if int(context.shape[0]) != batch_size or int(context_mask.shape[0]) != batch_size:
            raise ValueError(
                f"context batch {tuple(context.shape)} / mask batch "
                f"{tuple(context_mask.shape)} must match image batch {batch_size}."
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2:
                raise ValueError(f"`proprio` must be [B,D], got shape {tuple(proprio.shape)}")
            if int(proprio.shape[0]) != batch_size:
                raise ValueError(
                    f"`proprio` batch {proprio.shape[0]} must match image batch {batch_size}."
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # Per-slot reseeding keeps B=1 bitwise-equal to ``infer_action`` and makes
        # a request's noise independent of which slot it occupies in a larger batch.
        slot_seeds = _resolve_slot_seeds(seed, batch_size=batch_size)
        if slot_seeds is None:
            latents_action = torch.randn(
                (batch_size, action_horizon, self.action_expert.action_dim),
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            slot_noise = []
            for slot_seed in slot_seeds:
                generator = torch.Generator(device=rand_device).manual_seed(slot_seed)
                slot_noise.append(
                    torch.randn(
                        (1, action_horizon, self.action_expert.action_dim),
                        generator=generator,
                        device=rand_device,
                        dtype=torch.float32,
                    )
                )
            latents_action = torch.cat(slot_noise, dim=0).to(
                device=self.device, dtype=self.torch_dtype
            )

        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_images_latents_batch(
            input_image=input_image, tiled=tiled
        )
        vae_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        text_enc_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        proprio_ms = _sync_ms(t0)

        result = self._infer_action_core(
            first_frame_latents=first_frame_latents,
            latents_action=latents_action,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
        )

        core_timing = result.get("timing_ms", {})
        result["timing_ms"] = {
            "text_enc_ms": text_enc_ms,
            "proprio_ms": proprio_ms,
            "vae_ms": vae_ms,
            "video_prefill_ms": core_timing.get("video_prefill_ms", 0.0),
            "denoise_ms": core_timing.get("denoise_ms", 0.0),
            "num_denoise_steps": num_inference_steps,
        }
        return result

    def _assert_inference_masks_match_training_masks(
        self,
        *,
        video_self_mask: torch.Tensor,
        action_attention_mask: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> None:
        """Fail if decoupled inference masks drift from the training mask builder.

        Inference intentionally constructs the video prefill and action-query
        masks locally because those tensors are consumed at different stages of
        the KV-cache path. This guard keeps that local construction tied to the
        shared paradigm-C topology used by training.
        """
        expected_video_mask, expected_action_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        if (
            video_self_mask.shape != expected_video_mask.shape
            or not torch.equal(video_self_mask, expected_video_mask)
        ):
            raise RuntimeError(
                "Inference video mask diverged from _build_mot_attention_mask(). "
                "This would make training and inference use different paradigm-C "
                "attention topology."
            )
        if (
            action_attention_mask.shape != expected_action_mask.shape
            or not torch.equal(action_attention_mask, expected_action_mask)
        ):
            raise RuntimeError(
                "Inference action mask diverged from _build_mot_attention_mask(). "
                "This would make training and inference use different paradigm-C "
                "attention topology."
            )

    def _infer_action_core(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
    ) -> dict[str, Any]:
        """Core action inference logic: prefill video KV + Euler denoising loop.

        Factored out so that both ``infer_action()`` (encodes image itself) and
        ``infer()`` (already has encoded latents) can share this without
        re-encoding the conditioning image.
        """
        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        # Build video self-attention mask for prefill.
        video_self_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        # Prefill: run all video layers and keep each layer's KV.
        video_kv_per_layer = self.mot.prefill_video_kv(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_self_mask,
        )
        # Source selection / learned layer fusion is static for this observation.
        # Prepare it once and share the resulting cache across every denoising
        # step instead of recomputing fused_kv inside each action forward.
        video_kv_per_layer = self.mot.prepare_action_video_kv(video_kv_per_layer)
        video_prefill_ms = _sync_ms(t0)

        # Build action attention mask: [Sa, Sv + Sa].
        # For single-frame inference, Sv == tokens_per_frame, so all video
        # columns correspond to the first frame and are all-True.
        action_attention_mask = torch.zeros(
            (action_horizon, video_seq_len + action_horizon),
            dtype=torch.bool,
            device=video_pre["tokens"].device,
        )
        first_frame_tokens = min(tokens_per_frame, video_seq_len)
        action_attention_mask[:, :first_frame_tokens] = True
        action_attention_mask[:, video_seq_len:] = True
        self._assert_inference_masks_match_training_masks(
            video_self_mask=video_self_mask,
            action_attention_mask=action_attention_mask,
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        batch_size = int(latents_action.shape[0])
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_action = self._predict_action_noise_with_video_kv(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_per_layer=video_kv_per_layer,
                action_attention_mask=action_attention_mask,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        denoise_ms = _sync_ms(t0)

        # Keep the batch axis: callers that historically expected ``[H, D]``
        # (``infer_action`` / ``infer``) index ``[0]`` themselves.
        return {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
            "timing_ms": {
                "video_prefill_ms": video_prefill_ms,
                "denoise_ms": denoise_ms,
            },
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """Full inference: video denoising + action denoising (independent stages).

        This is the eval inference path called by ``trainer.evaluate()``. It must
        return ``{"video": list_of_PIL_frames, "action": action_tensor}`` to
        match the baseline interface.

        Stage 1: Denoise video using the standalone ``video_expert.forward()``
        through N Euler steps. Clamp the first frame at each step. Decode via VAE.

        Stage 2: Prefill all-layer video K/V from the clean first frame,
        then denoise actions using ``_predict_action_noise_with_video_kv()``.

        Args:
            prompt: Text prompt (mutually exclusive with context/context_mask).
            input_image: Conditioning image ``[1, 3, H, W]`` or ``[3, H, W]``.
            num_frames: Number of video frames to generate (must satisfy T%4==1).
            action: Optional GT action for video conditioning (unused in decoupled).
            action_horizon: Number of action steps to predict.
            proprio: Optional proprioceptive state.
            context: Precomputed text context.
            context_mask: Context attention mask.
            negative_prompt: Unused.
            text_cfg_scale: Unused.
            action_cfg_scale: Unused.
            num_inference_steps: Number of Euler denoising steps.
            sigma_shift: Optional override for inference sigma shift.
            seed: Random seed.
            rand_device: Device for random number generation.
            tiled: Whether to use tiled VAE encoding/decoding.

        Returns:
            Dict ``{"video": list_of_PIL_frames, "action": action_tensor}``.
        """
        self.eval()

        # --- Input validation ---
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        if checked_t != num_frames:
            raise ValueError(
                f"`num_frames` must satisfy T % 4 == 1, got {num_frames}"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # --- Initialize latents ---
        latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        if action_horizon is not None:
            latents_action = torch.randn(
                (1, action_horizon, self.action_expert.action_dim),
                generator=action_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            latents_action = None

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # --- Resolve text context ---
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # ====================================================================
        # Stage 1: Video denoising (standalone video expert, N Euler steps)
        # Uses video_expert.forward() which runs all 30 layers with its own
        # self-attention mask internally. This is the same approach as
        # FastWAMIDM.infer_joint() Stage 1.
        # ====================================================================
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            # Standalone video forward: pre_dit -> all blocks -> post_dit.
            pred_video = self.video_expert(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            # Clamp first frame to conditioning latent at each step.
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        # Decode video latents to PIL frames.
        video_frames = self._decode_latents(latents_video, tiled=tiled)

        # ====================================================================
        # Stage 2: Action denoising (if action_horizon is specified)
        # Prefill video KV from clean first frame, then denoise action.
        # ====================================================================
        action_out = None
        if latents_action is not None and action_horizon is not None:
            action_result = self._infer_action_core(
                first_frame_latents=first_frame_latents,
                latents_action=latents_action,
                context=context,
                context_mask=context_mask,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
            )
            # ``infer`` is a single-request API; squeeze the batch axis.
            action_out = action_result["action"][0]

        return {
            "video": video_frames,
            "action": action_out,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        num_video_frames: int = 0,
        action_horizon: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Raises NotImplementedError -- decoupled MoT does not support joint inference.

        Use ``infer()`` (video + action independently) or ``infer_action()`` (action only).
        """
        raise NotImplementedError(
            "Decoupled MoT does not support joint inference. Use infer() instead."
        )
