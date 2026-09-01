"""Qwen2.5-VL integration for physical Layer-0 S2Prune deletion."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLProcessor
from transformers.cache_utils import DynamicCache

try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:  # Transformers 4.49 uses the explicit fallback below.
    create_causal_mask = None
    create_sliding_window_causal_mask = None

from .allocation import SelectionResult, select_s2prune_tokens


@dataclass
class PreparedInput:
    """Multimodal prompt tensors before decoder execution."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    input_embeds: torch.Tensor
    image_grid_thw: torch.Tensor
    visual_tokens: torch.Tensor
    merge_size: int
    eos_token_id: int
    source_image: Image.Image


@dataclass
class PrunedPrefill:
    """Decoder state after S2Prune and all remaining prefill layers."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache: Any
    selection: SelectionResult
    full_sequence_length: int
    pruned_sequence_length: int


def _language_model(model):
    model_body = getattr(model, "model", None)
    language_model = getattr(model_body, "language_model", None)
    if language_model is not None:
        return language_model
    if model_body is not None and hasattr(model_body, "layers"):
        return model_body
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return language_model
    raise AttributeError("The model does not expose a decoder language model")


def _visual_tower(model):
    for owner in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        visual = getattr(owner, "visual", None) if owner is not None else None
        if visual is not None:
            return visual
    raise AttributeError("The model does not expose a visual tower")


def _rope_owner(model):
    for owner in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if owner is not None and hasattr(owner, "get_rope_index"):
            return owner
    raise AttributeError("The model does not expose get_rope_index")


def _new_dynamic_cache(config):
    try:
        return DynamicCache(config=config)
    except TypeError:  # Transformers 4.x
        return DynamicCache()


def _layer_hidden(layer_output):
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


def _layer_attention_mask(language_model, masks, layer_idx: int):
    layer_types = getattr(language_model.config, "layer_types", None)
    return masks["full_attention"] if layer_types is None else masks[layer_types[layer_idx]]


def _run_decoder_layer(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings,
    cache,
    cache_position: torch.Tensor,
):
    """Call a decoder layer across Transformers 4.x/5.x cache APIs.

    Transformers 4.49 names the argument ``past_key_value`` while newer Qwen
    implementations use ``past_key_values``. Passing the wrong spelling is
    silently ignored by decoder layers that accept arbitrary keyword arguments,
    leaving the generation cache empty.
    """

    parameters = inspect.signature(layer.forward).parameters
    if "past_key_value" in parameters:
        cache_argument = {"past_key_value": cache}
    elif "past_key_values" in parameters:
        cache_argument = {"past_key_values": cache}
    else:
        raise RuntimeError("Unsupported decoder-layer cache API")
    return layer(
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        position_ids=None,
        use_cache=True,
        cache_position=cache_position,
        **cache_argument,
    )


def _make_mask(language_model, hidden_states, attention_mask):
    if create_causal_mask is None:
        sequence_length = hidden_states.shape[1]
        minimum = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.triu(
            torch.full(
                (1, 1, sequence_length, sequence_length),
                minimum,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        if attention_mask is not None:
            key_padding = attention_mask[:, None, None, :].to(dtype=torch.bool)
            causal_mask = causal_mask.masked_fill(~key_padding, minimum)
        return {"full_attention": causal_mask, "sliding_attention": causal_mask}

    kwargs = {
        "config": language_model.config,
        "inputs_embeds": hidden_states,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": None,
    }
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if getattr(language_model, "has_sliding_layers", False):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    return masks


def _gather_cache(cache, layer_count: int, positions: torch.Tensor) -> None:
    """Gather the prefix KV cache at exactly the surviving sequence positions."""

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for layer_idx in range(min(layer_count, len(cache.key_cache))):
            keys = cache.key_cache[layer_idx]
            values = cache.value_cache[layer_idx]
            if keys is not None:
                cache.key_cache[layer_idx] = keys.index_select(-2, positions)
            if values is not None:
                cache.value_cache[layer_idx] = values.index_select(-2, positions)
        return

    for layer_idx in range(layer_count):
        layer_cache = cache.layers[layer_idx]
        if not getattr(layer_cache, "is_initialized", False):
            continue
        layer_cache.keys = layer_cache.keys.index_select(-2, positions)
        layer_cache.values = layer_cache.values.index_select(-2, positions)


def load_model(
    model_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    attention_implementation: str = "eager",
):
    """Load the exact Qwen2.5-VL model family used by the experiments."""

    config = AutoConfig.from_pretrained(model_path)
    if str(getattr(config, "model_type", "")) != "qwen2_5_vl":
        raise ValueError("S2Prune's released Qwen runner requires Qwen2.5-VL")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        attn_implementation=attention_implementation,
    ).eval().to(device)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, processor, tokenizer


def _build_position_ids(model, processor_output, input_ids, attention_mask, image_grid_thw):
    owner = _rope_owner(model)
    parameters = inspect.signature(owner.get_rope_index).parameters
    if "mm_token_type_ids" in parameters:
        token_types = processor_output.get("mm_token_type_ids")
        if token_types is None:
            token_types = torch.zeros_like(input_ids)
            token_types[input_ids == model.config.image_token_id] = 1
        else:
            token_types = token_types.to(model.device)
        position_ids, _ = owner.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=token_types,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
    else:
        position_ids, _ = owner.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
    return position_ids


@torch.inference_mode()
def prepare_input(
    model,
    processor,
    question: str,
    image: Image.Image,
    target_visual_tokens: int = 576,
) -> PreparedInput:
    """Create the fixed 576-token multimodal prompt used in the paper.

    The source image is directly bicubically resized to 672 x 672 for the model
    input. The unresized RGB image is retained for Laplacian allocation.
    """

    source_image = image.convert("RGB")
    question = question.replace("<image>", "").strip()
    if not question:
        question = "Please describe this image."

    merge_size = int(processor.image_processor.merge_size)
    patch_size = int(processor.image_processor.patch_size)
    side_grid = int(round((int(target_visual_tokens) * merge_size**2) ** 0.5))
    if side_grid * side_grid != int(target_visual_tokens) * merge_size**2:
        raise ValueError("target_visual_tokens must correspond to a square Qwen grid")
    side_pixels = side_grid * patch_size
    resized = source_image.resize(
        (side_pixels, side_pixels), resample=Image.Resampling.BICUBIC
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": resized},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = processor(text=[text], images=[resized], return_tensors="pt")
    input_ids = output["input_ids"].to(model.device)
    attention_mask = output["attention_mask"].to(model.device)
    pixel_values = output["pixel_values"].to(model.device)
    image_grid_thw = output["image_grid_thw"].to(model.device)

    actual_visual_tokens = int(image_grid_thw.prod().item() // merge_size**2)
    if actual_visual_tokens != int(target_visual_tokens):
        raise RuntimeError(
            f"Expected {target_visual_tokens} visual tokens, got {actual_visual_tokens}"
        )
    position_ids = _build_position_ids(
        model, output, input_ids, attention_mask, image_grid_thw
    )

    visual = _visual_tower(model)
    visual_dtype = (
        visual.get_dtype() if hasattr(visual, "get_dtype") else next(visual.parameters()).dtype
    )
    visual_tokens = visual(pixel_values.to(dtype=visual_dtype), image_grid_thw)
    if hasattr(visual_tokens, "pooler_output") and visual_tokens.pooler_output is not None:
        visual_tokens = visual_tokens.pooler_output
    elif hasattr(visual_tokens, "last_hidden_state") and visual_tokens.last_hidden_state is not None:
        visual_tokens = visual_tokens.last_hidden_state
    visual_tokens = visual_tokens.reshape(-1, visual_tokens.shape[-1])

    image_positions = (
        input_ids[0] == model.config.image_token_id
    ).nonzero(as_tuple=False).squeeze(-1)
    if int(image_positions.numel()) != actual_visual_tokens:
        raise RuntimeError("Image placeholders and visual features do not match")
    input_embeds = model.get_input_embeddings()(input_ids)
    input_embeds[0, image_positions] = visual_tokens.to(input_embeds.dtype)

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = processor.tokenizer.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    return PreparedInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        input_embeds=input_embeds,
        image_grid_thw=image_grid_thw,
        visual_tokens=visual_tokens,
        merge_size=merge_size,
        eos_token_id=int(eos_token_id),
        source_image=source_image,
    )


@torch.inference_mode()
def s2prune_prefill(
    model,
    prepared: PreparedInput,
    budget: int,
    grid_size: int | None = None,
) -> PrunedPrefill:
    """Run Layer 0 in full, physically prune, then run all suffix layers."""

    language_model = _language_model(model)
    hidden = prepared.input_embeds
    position_ids = prepared.position_ids
    attention_mask = prepared.attention_mask
    input_ids = prepared.input_ids
    visual_positions = (
        input_ids[0] == model.config.image_token_id
    ).nonzero(as_tuple=False).squeeze(-1)
    if int(visual_positions.numel()) != 576:
        raise RuntimeError(f"S2Prune expects 576 visual tokens, got {visual_positions.numel()}")

    cache = _new_dynamic_cache(language_model.config)
    full_cache_position = torch.arange(
        hidden.shape[1], device=hidden.device, dtype=torch.long
    )
    position_embeddings = language_model.rotary_emb(hidden, position_ids)
    prefix_masks = _make_mask(language_model, hidden, attention_mask)

    visual_before = hidden[:, visual_positions, :].clone()
    hidden = _layer_hidden(_run_decoder_layer(
        language_model.layers[0],
        hidden,
        attention_mask=_layer_attention_mask(language_model, prefix_masks, 0),
        position_embeddings=position_embeddings,
        cache=cache,
        cache_position=full_cache_position,
    ))
    erc_scores = torch.linalg.vector_norm(
        (hidden[:, visual_positions, :] - visual_before).float(), ord=2, dim=-1
    ).squeeze(0)

    _, patch_h, patch_w = (
        int(value) for value in prepared.image_grid_thw.detach().cpu().reshape(-1, 3)[0]
    )
    merge_size = prepared.merge_size
    if patch_h * patch_w // merge_size**2 != int(erc_scores.numel()):
        raise RuntimeError("PatchMerger shape does not match the visual ERC scores")
    grid_h, grid_w = patch_h // merge_size, patch_w // merge_size
    selection = select_s2prune_tokens(
        erc_scores=erc_scores,
        image=prepared.source_image,
        grid_h=grid_h,
        grid_w=grid_w,
        budget=int(budget),
        grid_size=grid_size,
    )

    selected_visual_positions = visual_positions.index_select(
        0, selection.selected_indices
    )
    nonvisual_positions = (
        input_ids[0] != model.config.image_token_id
    ).nonzero(as_tuple=False).squeeze(-1)
    keep_positions = torch.cat(
        (nonvisual_positions, selected_visual_positions)
    ).sort().values

    kept_ids = input_ids[0].index_select(0, keep_positions)
    kept_visual = keep_positions[kept_ids == model.config.image_token_id]
    kept_nonvisual = keep_positions[kept_ids != model.config.image_token_id]
    if not torch.equal(kept_visual, selected_visual_positions):
        raise RuntimeError("Visual token order changed during sequence deletion")
    if not torch.equal(kept_nonvisual, nonvisual_positions):
        raise RuntimeError("A text or special token was removed")

    hidden = hidden.index_select(1, keep_positions)
    attention_mask = attention_mask.index_select(1, keep_positions)
    position_ids = position_ids.index_select(2, keep_positions)
    _gather_cache(cache, 1, keep_positions)

    suffix_masks = _make_mask(language_model, hidden, attention_mask)
    suffix_position_embeddings = language_model.rotary_emb(hidden, position_ids)
    suffix_cache_position = torch.arange(
        hidden.shape[1], device=hidden.device, dtype=torch.long
    )
    for layer_idx in range(1, len(language_model.layers)):
        hidden = _layer_hidden(_run_decoder_layer(
            language_model.layers[layer_idx],
            hidden,
            attention_mask=_layer_attention_mask(
                language_model, suffix_masks, layer_idx
            ),
            position_embeddings=suffix_position_embeddings,
            cache=cache,
            cache_position=suffix_cache_position,
        ))

    return PrunedPrefill(
        hidden_states=hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        selection=selection,
        full_sequence_length=int(input_ids.shape[1]),
        pruned_sequence_length=int(hidden.shape[1]),
    )


def _decode_step_position_ids(text_end_position: int, step: int, device) -> torch.Tensor:
    position = int(text_end_position) + 1 + int(step)
    return torch.full((3, 1, 1), position, device=device, dtype=torch.long)


def _closed_answer_complete(decoded: str, dataset: str) -> bool:
    text = decoded.strip()
    if dataset in {"ScienceQA", "MMBench", "MMMU"}:
        return re.fullmatch(r"\(?[A-E]\)?[.)]?", text, flags=re.IGNORECASE) is not None
    if dataset in {"MME", "POPE"}:
        return re.fullmatch(r"(?:yes|no)[.!]?", text, flags=re.IGNORECASE) is not None
    return False


def _repeated_suffix_start(token_ids, repeats: int = 3, max_ngram: int = 64):
    for ngram_size in range(1, min(max_ngram, len(token_ids) // repeats) + 1):
        start = len(token_ids) - repeats * ngram_size
        unit = token_ids[start : start + ngram_size]
        if unit and all(
            token_ids[start + offset * ngram_size : start + (offset + 1) * ngram_size]
            == unit
            for offset in range(1, repeats)
        ):
            return start
    return None


@torch.inference_mode()
def greedy_decode(
    model,
    tokenizer,
    prefill: PrunedPrefill,
    eos_token_id: int,
    max_new_tokens: int,
    dataset: str,
) -> Tuple[str, int]:
    """Greedily decode from the manually shortened multimodal KV cache."""

    language_model = _language_model(model)
    logits = model.lm_head(language_model.norm(prefill.hidden_states))[:, -1, :]
    generated = []
    current_mask = prefill.attention_mask
    prefill_text_end = int(prefill.position_ids[0, 0, -1].item())
    stop_token_ids = {int(eos_token_id)}
    configured_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured_eos, (list, tuple, set)):
        stop_token_ids.update(int(token_id) for token_id in configured_eos)
    elif configured_eos is not None:
        stop_token_ids.add(int(configured_eos))

    for step in range(int(max_new_tokens)):
        token = torch.argmax(logits, dim=-1)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id in stop_token_ids:
            break
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if _closed_answer_complete(decoded, dataset):
            break
        repeated_start = _repeated_suffix_start(generated)
        if repeated_start is not None:
            generated = generated[:repeated_start]
            break

        token_embeds = model.get_input_embeddings()(token.view(1, 1))
        current_mask = torch.cat(
            (
                current_mask,
                torch.ones(
                    (1, 1),
                    device=current_mask.device,
                    dtype=current_mask.dtype,
                ),
            ),
            dim=1,
        )
        step_position = _decode_step_position_ids(
            prefill_text_end, step, prefill.hidden_states.device
        )
        cache_position = torch.tensor(
            [current_mask.shape[1] - 1],
            device=prefill.hidden_states.device,
            dtype=torch.long,
        )
        output = model(
            inputs_embeds=token_embeds,
            attention_mask=current_mask,
            position_ids=step_position,
            past_key_values=prefill.cache,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = output.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)
