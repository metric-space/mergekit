import logging
import os
from typing import Optional

import click
import safetensors.torch
import torch
import tqdm
from transformers import AutoTokenizer

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-activation-based-merge")
@click.argument("model_path", type=str)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option(
    "--dtype",
    type=str,
    default="float16",
    help="Data type to convert weights to",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda",
    help="Device to compute on (default: cuda)",
)
@click.option(
    "--noise-level",
    "-n",
    type=float,
    default=0.1,
    help="Noise level to add to the weights",
    )
@add_merge_options
def main(
    model_path: str,
    out_path: str,
    dtype: Optional[str],
    device: Optional[str],
    noise_level: Optional[float],
    merge_options,
):
    model = ModelReference.model_validate(model_path)

    dtype = dtype_from_name(dtype) if dtype else None

    cache = LoaderCache()
    cache.lazy_unpickle = merge_options.lazy_unpickle
    cache.hf_cache_dir = merge_options.transformers_cache

    for m in tqdm.tqdm([model], desc="Preparing models"):
        cache.get(m)

    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    model_config = model.config(trust_remote_code=merge_options.trust_remote_code)
    model_arch_info = get_architecture_info(
        model.config(trust_remote_code=merge_options.trust_remote_code)
    )

    loader_1 = cache.get(model)

    os.makedirs(out_path, exist_ok=True)

    merge_internal_cache = {}
    unmerge_internal_cache = {}

    for weight_info in model_arch_info.all_weights(config=model_config):

        original_w = loader_1.get_tensor(weight_info.name, device=device)

        merge_matrix, unmerge_matrix = None, None

        print(f"{weight_info} with shape {original_w.shape}")

        if weight_info.input_space:
            if weight_info.input_space not in unmerge_internal_cache:
                if weight_info.is_embed:
                    unmerge_matrix = torch.eye(original_w.shape[1], device=device)
                else:
                    unmerge_matrix = torch.eye(original_w.shape[0], device=device)
                unmerge_matrix = unmerge_matrix + torch.randn_like(unmerge_matrix).diag().diagflat() * noise_level

                # calculate inverse of unmerge_matrix
                merge_internal_cache[weight_info.input_space] = torch.inverse(unmerge_matrix).to(dtype=dtype)
                unmerge_matrix = unmerge_matrix.to(dtype=dtype)
                unmerge_internal_cache[weight_info.input_space] = unmerge_matrix

                print(f"for {weight_info.input_space} IS  U: {unmerge_matrix.shape} and it's inverse: {merge_internal_cache[weight_info.input_space].shape}")
            else:
                unmerge_matrix = unmerge_internal_cache[weight_info.input_space]

        if weight_info.output_space:
            if weight_info.output_space not in merge_internal_cache:
                if weight_info.is_embed:
                    merge_matrix = torch.eye(original_w.shape[1], device=device)
                else:
                    merge_matrix = torch.eye(original_w.shape[0], device=device)

                merge_matrix = merge_matrix + torch.randn_like(merge_matrix).diag().diagflat() * noise_level

                unmerge_internal_cache[weight_info.output_space] = torch.inverse(merge_matrix).to(dtype=dtype)
                merge_matrix = merge_matrix.to(dtype=dtype)
                merge_internal_cache[weight_info.output_space] = merge_matrix

                print(f"for {weight_info.output_space} OS  M: {merge_matrix.shape} and it's inverse: {unmerge_internal_cache[weight_info.output_space].shape}")
            else:
                merge_matrix = merge_internal_cache[weight_info.output_space]

        if dtype is not None:
            original_w = original_w.to(dtype=dtype)

        w = torch.clone(original_w)

        if (merge_matrix is None and unmerge_matrix is None):
            logging.warning(
                f"❌ Weight {weight_info.name} for model 1 and model 2 has no merge or unmerge matrix"
            )

        if merge_matrix is not None:
            if weight_info.is_embed:
                print(w.dtype)
                print(merge_matrix.dtype)
                w = (merge_matrix @ w.T).T  # this could also be  merge_matrix.T @ w  by matrix transpose laws
            else:
                w = merge_matrix @ w

        if unmerge_matrix is not None:
            w = w @ unmerge_matrix

        # check if weights have not mutated, if yes then  shoot warning
        if torch.allclose(original_w, w):
            logging.warning(
                f"❌ Weight {weight_info.name} for model 1 has NOT mutated during merge"
            )
        else:
            logging.warning(
                f"✅ Weight {weight_info.name} for model 1 has mutated during merge"
            )

        writer.save_tensor(weight_info.name, w)
    writer.finalize()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(out_path, safe_serialization=True)

    # write config
    model_out_config = model.config(trust_remote_code=merge_options.trust_remote_code)
    if dtype:
        model_out_config.torch_dtype = dtype
    model_out_config.save_pretrained(out_path)


main()
