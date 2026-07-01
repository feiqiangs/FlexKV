# External backend adapters for FlexKV.
# Currently provides the mooncake-store distributed KV cache backend.
#
# NOTE: intentionally NO eager top-level import of ``mooncake_store_utils`` here.
# ``mooncake_store_utils`` pulls in torch / requests / flexkv.common.block
# (-> flexkv.c_ext), so importing the light-weight ``mooncake_store_keys``
# (needed by flexkv.common.config) must NOT drag in that heavy chain.
# The three public objects are exposed lazily via ``__getattr__`` below.
__all__ = ["MooncakeStoreConfig", "MooncakeStoreClient", "MooncakeStoreCacheEngine"]
def __getattr__(name):
    if name in __all__:
        from . import mooncake_store_utils as _m
        return getattr(_m, name)
    raise AttributeError(name)
