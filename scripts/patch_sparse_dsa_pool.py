#!/usr/bin/env python3
"""Patch kv_cache_configurator.py: route hybrid KDA models to the sparse-layer
DSA pool (11 materialized layers instead of 45) inside _build_dsa_kv_pool."""
import sys

PATH = "/data/models/mtp_experiment_0907/sglang/srt/mem_cache/kv_cache_configurator.py"

OLD = """        else:
            PoolCls = DSATokenToKVPool
"""
NEW = """        else:
            # Hybrid KDA models (glm5_next): only a sparse subset of layers are
            # DSA; materialize just those instead of all num_effective_layers
            # slots (34 dead KDA layers cost ~76% of pool bytes).
            from sglang.srt.mem_cache.sparse_layer_dsa_pool import (
                SparseLayerDSATokenToKVPool,
            )

            _mambaish_ids = (
                self.mambaish_config.full_attention_layer_ids
                if self.mambaish_config is not None
                else []
            )
            _sparse_ids = [
                i
                for i in _mambaish_ids
                if self.layer_info.start_layer <= i < self.layer_info.end_layer
            ]
            if 0 < len(_sparse_ids) < self.layer_info.num_effective_layers:
                PoolCls = SparseLayerDSATokenToKVPool
                pool_kwargs["sparse_layer_ids"] = _sparse_ids
            else:
                PoolCls = DSATokenToKVPool
"""

with open(PATH) as f:
    src = f.read()

count = src.count(OLD)
if count != 1:
    print(f"FAIL: anchor occurs {count} times (need exactly 1)", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW)
with open(PATH, "w") as f:
    f.write(src)

import ast
ast.parse(src)
print("PATCH OK, syntax valid")
