import os
import sys
from collections import defaultdict

import click
import numpy as np
import safetensors.torch
import scipy
import torch

from mergekit.architecture import _template_substitution, get_architecture_info
from mergekit.common import ModelReference


def calc_correlation_matrix(feats):
    feats = feats.view(-1, feats.shape[-1])

    return torch.corrcoef(feats.T)


def match_tensors_permute(
    absval=False,
    correlation_matrix=None,
):
    """
    This function is adapted from ZipIt! (https://github.com/gstoica27/ZipIt)
    """

    Om = correlation_matrix.shape[0] // 2
    device = correlation_matrix.device

    mats = [torch.eye(Om, device=device)]

    corr_submatrix = correlation_matrix[:Om, Om:].cpu().numpy()
    if absval:
        corr_submatrix = np.absolute(corr_submatrix)
    _, col_ind = scipy.optimize.linear_sum_assignment(corr_submatrix, maximize=True)

    new_mat = torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)]
    mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)

    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge, col_ind


def match_tensors_permute_MHA(
    n_heads=32,
    absval=False,
    correlation_matrix=None,
):
    """
    Handles different head permutations in attention.
    Modified version of the function here: https://github.com/nverma1/merging-text-transformers/blob/main/matching_functions.py#L76
    """

    Om = correlation_matrix.shape[0] // 2
    device = correlation_matrix.device
    query_size = Om // n_heads

    mats = [torch.eye(Om, device=device)]
    head_perms = []

    costs = np.ones((n_heads, n_heads)) * -sys.maxsize

    col_inds_storage = defaultdict(lambda: defaultdict(int))

    for j in range(n_heads):
        for k in range(n_heads):
            head1_idx = [query_size * j, query_size * (j + 1)]
            head2_idx = [query_size * k, query_size * (k + 1)]

            corr_submatrix = (
                correlation_matrix[
                    head1_idx[0] : head1_idx[1],
                    (Om + head2_idx[0]) : (Om + head2_idx[1]),
                ]
                .cpu()
                .numpy()
            )
            if absval:
                corr_submatrix = np.absolute(corr_submatrix)

            # compute perm for head j & head k
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                corr_submatrix, maximize=True
            )

            # store cost (cost is maximized here)
            costs[j, k] = corr_submatrix[row_ind, col_ind].sum()

            # store perm so we don't have to recompute it later
            col_inds_storage[j][k] = col_ind

    outer_row_ind, outer_col_ind = scipy.optimize.linear_sum_assignment(
        costs, maximize=True
    )

    for j in range(n_heads):
        head_1 = outer_row_ind[j]
        head_2 = outer_col_ind[j]

        head_perm = col_inds_storage[head_1][head_2]
        head_perms.append(torch.tensor(head_perm + query_size * head_2))

    new_mat = torch.eye(Om, device=device)[
        torch.cat(head_perms).clone().detach().long().to(device)
    ]
    mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge


@click.command("mergekit-abm-extract-permutations")
@click.argument("model1-ft", type=str, required=True)
@click.argument("model2-ft", type=str, required=True)
@click.option("--model_path", type=str, required=True, help="Model information")
@click.option(
    "--out_path", required=True, type=str, help="Output path for metric tensors"
)
@click.option(
    "--absval/--no-absval",
    required=False,
    default=False,
    help="Use absolute value on correlation matrices/submatrices while calculating merge/unmerge matrices",
)
@click.option(
    "--store-correlation/--no-store-correlation",
    required=False,
    default=False,
    help="Store correlation matrices",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="Device to compute on (default: cpu)",
)
def main(model1_ft, model2_ft, model_path, out_path, absval, store_correlation, device):
    os.makedirs(out_path, exist_ok=True)

    model = ModelReference.model_validate(model_path)

    model_config = model.config()

    model_arch_info = get_architecture_info(model_config)

    _json = model_arch_info.definition

    residual_space = None
    kq_space = None
    v_space = None

    # extract the residual, attention related spaces
    for weight in _json.layer_templates.weights:
        if weight.is_kq:
            kq_space = weight.output_space
            residual_space = weight.input_space
            continue

        # assuming order is observed
        if (
            not weight.is_kq
            and weight.head_split
            and (weight.input_space == residual_space)
        ):
            v_space = weight.output_space
            continue

    num_layers = model_arch_info.num_layers(model_config)

    kq_spaces = []
    v_spaces = []
    for j in range(num_layers):
        kq_spaces.append(
            _template_substitution(kq_space, num_layers=num_layers, layer_idx=j)
        )
        v_spaces.append(
            _template_substitution(v_space, num_layers=num_layers, layer_idx=j)
        )

    model1_features = safetensors.torch.load_file(model1_ft, device=device)
    model2_features = safetensors.torch.load_file(model2_ft, device=device)

    model1_features.pop("attention_mask")
    model2_features.pop("attention_mask")

    for feature_space in model1_features.keys():
        concatenated_feature = torch.cat(
            (model1_features[feature_space], model2_features[feature_space]), dim=-1
        )

        correlation_matrix = calc_correlation_matrix(concatenated_feature)
        if store_correlation:
            safetensors.torch.save_file(
                {feature_space: correlation_matrix.contiguous()},
                f"{out_path}/correlations/{feature_space}_correlation.safetensor",
            )


        if feature_space in (kq_spaces + v_spaces):
            merge, unmerge = match_tensors_permute_MHA(
                correlation_matrix=correlation_matrix,
                n_heads=model_config.num_attention_heads,
                absval=absval,
            )

        else:
            merge, unmerge, col_indices = match_tensors_permute(
                correlation_matrix=correlation_matrix,
                absval=absval,
            )

            errors = {}
            for i in range(len(col_indices)):
                k_ = col_ind[indices]
                if  i != k_:
                    errors[i] = np.abs(k_ - i)
            if np.sum(list(errors.values())) > 0:
                print(f"Feature space: {feature_space} -> {np.mean(list(errors.values()))}")
            else:
                print(f"Feature space: {feature_space} -> No errors")


        safetensors.torch.save_file(
            {feature_space: merge.contiguous()},
            f"{out_path}/{feature_space}_merge.safetensor",
        )

        safetensors.torch.save_file(
            {feature_space: unmerge.contiguous()},
            f"{out_path}/{feature_space}_unmerge.safetensor",
        )

        del merge, unmerge, correlation_matrix, concatenated_feature


if __name__ == "__main__":
    main()
