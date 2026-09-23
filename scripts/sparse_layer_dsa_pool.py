# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Sparse-layer DSA KV pool for hybrid KDA models (GLM-5.3-Flash glm5_next).

Only 11 of the 45 decoder layers are DSA (deepseek_sparse_attention); the 34
KDA (linear attention) layers keep their state in the mamba pool and never
touch this pool (`_transfer_full_attention_id` raises on non-DSA layer ids).
The stock `DSATokenToKVPool` still materializes `layer_num == num_hidden_layers`
slots because DSA layer ids are non-contiguous ([3, 7, ..., 43]) and the pool
is indexed by `layer_id - start_layer` -- 76% of the per-token bytes are dead
weight (34 layers x (1024B latent + ~132B index) of ~51.8KB/token).

This subclass materializes only the DSA layers and translates model layer ids
to dense ids at every public entry point. All external access funnels through
the overridden methods; `k_buffer`/`index_key_cache` direct indexing outside
this class is confined to the DSV4 pools, which hybrid glm5_next never uses.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


class SparseLayerDSATokenToKVPool(DSATokenToKVPool):
    """DSA pool that only materializes the sparse subset of DSA layers.

    Public entry points take GLOBAL model layer ids (unchanged contract with
    the attention backend / indexer); internally they are mapped to dense ids
    via ``_dense_of``.
    """

    def __init__(
        self,
        *args,
        sparse_layer_ids: List[int],
        **kwargs,
    ):
        self.sparse_layer_ids = list(sparse_layer_ids)
        if not self.sparse_layer_ids:
            raise ValueError("SparseLayerDSATokenToKVPool needs >=1 DSA layer")
        self._dense_of = {gid: i for i, gid in enumerate(self.sparse_layer_ids)}
        # The configurator passes layer_num (=num_effective_layers) and a
        # skip_topk list indexed by GLOBAL layer id offset from start_layer.
        # Override the former with the sparse count; re-index the latter dense.
        start = kwargs.get("start_layer") or 0
        skip = kwargs.pop("skip_topk_layers", None)
        if skip is not None:
            kwargs["skip_topk_layers"] = [skip[gid - start] for gid in self.sparse_layer_ids]
        kwargs.pop("layer_num", None)
        super().__init__(*args, layer_num=len(self.sparse_layer_ids), **kwargs)

    # ---- id translation -------------------------------------------------

    def _dense(self, layer_id: int) -> int:
        try:
            return self._dense_of[layer_id]
        except KeyError:
            raise ValueError(
                f"layer_id {layer_id} is not a DSA layer "
                f"(sparse DSA set: {self.sparse_layer_ids})"
            ) from None

    def _dense_global(self, layer_id: int) -> int:
        # Parent methods index with `layer_id - self.start_layer`; hand them a
        # virtual global id whose offset from start_layer is the dense id.
        return self._dense(layer_id) + self.start_layer

    # ---- KV buffer entry points ------------------------------------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return super().get_key_buffer(self._dense_global(layer_id))

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return super().get_value_buffer(self._dense_global(layer_id))

    def set_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ):
        # `layer_id_override` is the GLOBAL model layer id in the caller's
        # frame; translate before delegating (parent indexes dense slots).
        gid = layer_id_override if layer_id_override is not None else layer.layer_id
        return super().set_mla_kv_buffer(
            layer, loc, cache_k_nope, cache_k_rope,
            layer_id_override=self._dense_global(gid),
        )

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ):
        # MLA-flavored signature (layer, loc_info, cache_k, cache_v,
        # layer_id_override): translate the GLOBAL layer id before delegating.
        gid = layer_id_override if layer_id_override is not None else layer.layer_id
        return super().set_kv_buffer(
            layer, loc_info, cache_k, cache_v,
            layer_id_override=self._dense_global(gid),
        )

    # ---- DSA indexer entry points -----------------------------------------

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return super().get_index_k_with_scale_buffer(self._dense_global(layer_id))

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        return super().get_index_k_continuous(
            self._dense_global(layer_id), seq_len, page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        return super().get_index_k_scale_buffer(
            self._dense_global(layer_id),
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        super().set_index_k_scale_buffer(
            self._dense_global(layer_id), loc, index_k, index_k_scale
        )

    def get_kv_layer_ids(self):
        # Global layer ids aligned with the materialized DSA buffers.
        return list(self.sparse_layer_ids)
