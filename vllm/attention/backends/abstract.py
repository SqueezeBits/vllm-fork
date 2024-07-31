from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import (Any, Dict, Generic, List, Optional, Set, Tuple, Type,
                    TypeVar)

import torch


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        raise NotImplementedError

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "AttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError


@dataclass
class AttentionMetadataPerStage:
    """Attention metadata for a specific stage. I.e., prefill or decode."""
    # Note that this class is required for chunked-prefill,
    # as each prefill and decode phase should have their own metadata.
    
    # Total number of prefill requests.
    num_prefills: torch.Tensor
    # Number of prefill tokens.
    num_prefill_tokens: torch.Tensor
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: torch.Tensor

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }

T = TypeVar("T", bound=AttentionMetadataPerStage)

@dataclass
class AttentionMetadata(Generic[T]):
    """Attention metadata for prefill and decode batched together."""
    # Total number of prefill requests.
    num_prefills: torch.Tensor
    # Number of prefill tokens.
    num_prefill_tokens: torch.Tensor
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: torch.Tensor
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    # The attention metadata for prefill requests in a batch.
    # None if there's no prefill requests in a batch.
    prefill_metadata: Optional[T]
    # The attention metadata for decode requests in a batch.
    # None if there's no decode requests in a batch.
    decode_metadata: Optional[T]

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }


T = TypeVar("T", bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        max_seq_len: Optional[int] = 4096,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        raise NotImplementedError
