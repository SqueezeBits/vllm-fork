"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.
Currently supporting TP and PP.
Workflow:
- In prefill instance, KV cache sender *buffers* the KV cache send requests
- In decode instance
    - KV cache receiver sends the hash of input tokens to sender
    - KV cache sender executes send request
    - KV cache receiver receives the KV cache
"""
from typing import List, Optional, Tuple, Union
from collections import defaultdict, deque
from threading import Lock
import contextlib
import time
import threading

import torch
from torch.distributed import Backend

import vllm.envs as envs
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.logger import init_logger
import vllm.distributed.parallel_state as ps
from vllm.sequence import IntermediateTensors

from vllm.hpu.utils import with_mark_steps

assert envs.VLLM_DISAGG_PREFILL_ROLE in [None, "prefill", "decode"], \
    "VLLM_DISAGG_PREFILL_ROLE can only be prefill or decode."

IS_DISTRIBUTED_KV_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE is not None)
IS_KV_PREFILL_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "prefill")
IS_KV_DECODE_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "decode")

# add a tag when sending/recving input hash
DISTRIBUTED_KV_GLOO_TAG = 24857323

logger = init_logger(__name__)

import logging

import habana_frameworks.torch as htorch

class RankFilter(logging.Filter):
    def filter(self, record):
        # Only log if rank is 4
        rank = 1
        with contextlib.suppress(Exception):
            rank = torch.distributed.get_rank()

        return rank % 4 == 0


for handler in logger.handlers:
    handler.addFilter(RankFilter())


class DistributedKVCoordinator(GroupCoordinator):
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_custom_allreduce: bool = False,
        use_message_queue_broadcaster: bool = False,
    ):
        super().__init__(
            group_ranks,
            local_rank,
            torch_distributed_backend,
            False, # use pynccl
            use_custom_allreduce,
            False, # use TPU communicator
            True, # use HPU communicator
            use_message_queue_broadcaster,
        )

        # use a threadpool to buffer send request in disaggregated prefill
        self.input_hash_to_kv_sending_requests = defaultdict(deque)
        self.kv_sending_thread = None
        self.input_hash_to_kv_sending_requests_lock = Lock()
        self.target_rank_for_send = self.ranks[(self.rank_in_group + 1) %
                                               self.world_size]
        self.target_rank_for_recv = self.ranks[(self.rank_in_group - 1) %
                                               self.world_size]

        torch.set_default_device(self.device)

    @with_mark_steps
    def kv_cache_send(self,
                      input_hash: int,
                      tensor: torch.Tensor,
                      dst: Optional[int] = None) -> None:
        """Push the KV cache send request into the send buffer"""
        """NOTE: `dst` is the local rank of the destination rank."""

        self.input_hash_to_kv_sending_requests[input_hash].append((
            self.send, tensor.clone(), dst))

    @with_mark_steps
    def kv_cache_recv(
            self,
            size: torch.Size,
            dtype: torch.dtype,
            src: Optional[int] = None
    ) -> torch.Tensor:
        """Receives a tensor from the src rank (blocking)."""
        """This API should be used together with `push`"""
        """NOTE: `src` is the local rank of the destination rank."""

        tensor = self.recv(size, dtype, src)
        return tensor


    def send_input_hash(self, input_hash: int) -> int:

        logger.debug('[rank%d]: Sending input hash %d to rank %d',
                     torch.distributed.get_rank(), input_hash,
                     self.target_rank_for_send)

        # KV cache send go through CPU, and the original `send` only use GPU.
        # So create a new group for sending input hash.
        input_hash_tensor = torch.tensor([input_hash], device="cpu").long()
        torch.distributed.send(input_hash_tensor,
                               self.target_rank_for_send,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        return_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.recv(return_tensor,
                               self.target_rank_for_recv,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        return return_tensor.item()

    def recv_input_hash(self) -> Optional[int]:
        '''
            Receive an input hash, and check if it is already cached
        '''

        input_hash_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.recv(input_hash_tensor,
                               self.target_rank_for_recv,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        input_hash = input_hash_tensor.item()

        # a new input hash comes in, see if it is already cached
        logger.debug('Successfully received input hash %d', input_hash)
        self.input_hash_to_kv_sending_requests_lock.acquire()
        if input_hash not in self.input_hash_to_kv_sending_requests:
            logger.warning(
            f"The KV cache of {input_hash} does not exist. "\
            f"Existing input hash: {list(self.input_hash_to_kv_sending_requests.keys())}")

            # 0 for fail
            x = torch.tensor([0], device="cpu").long()
            torch.distributed.send(x,
                                    self.target_rank_for_send,
                                    self.cpu_group,
                                    tag=DISTRIBUTED_KV_GLOO_TAG)
            return None
        else:
            logger.debug('Input hash %d exists, start sending', input_hash)

            # 1 for success
            x = torch.tensor([1], device="cpu").long()
            torch.distributed.send(x,
                                   self.target_rank_for_send,
                                   self.cpu_group,
                                   tag=DISTRIBUTED_KV_GLOO_TAG)
        return input_hash

    def kv_cache_send_loop(self):

        while True:
            htorch.core.mark_step()
            logger.debug(
                '[rank%d]: Waiting for input hash from rank %d, my keys are %s',
                torch.distributed.get_rank(),
                self.target_rank_for_recv,
                list(self.input_hash_to_kv_sending_requests.keys()),
            )
            # wait for a new input hash
            # this function will acquire the lock
            input_hash = self.recv_input_hash()

            if input_hash is None:
                self.input_hash_to_kv_sending_requests_lock.release()
                continue

            # execute corresponding kv cache sending jobs in request queue
            while True:
                request = self.input_hash_to_kv_sending_requests[input_hash].popleft()
        
                # An empty request: the KV cache of one request are all sent
                if len(request) == 0:
                    break

                send_func, payload, dst = request
                send_func(payload, dst)

            if len(self.input_hash_to_kv_sending_requests[input_hash]) == 0:
                logger.debug('Finish input hash %d, free GPU memory...',
                             input_hash)
                del self.input_hash_to_kv_sending_requests[input_hash]
            else:
                logger.debug(
                    'The buffer for input hash %d is not empty, meaning that '\
                    'there are two jobs with identical input.',
                    input_hash)

            htorch.core.mark_step()
            self.input_hash_to_kv_sending_requests_lock.release()

    @with_mark_steps
    def kv_cache_send_ready(self, input_hash: int):
        # append an empty list to separate requests
        # as there might be identical requests, that has the same input hash
        self.input_hash_to_kv_sending_requests[input_hash].append([])
        logger.debug(f'Buffered input hash {input_hash}')

        if self.kv_sending_thread is None:
            self.kv_sending_thread = threading.Thread(
                target=self.kv_cache_send_loop)
            self.kv_sending_thread.start()


    def kv_cache_recv_start(self, input_hash: int):
        # notify the kv cache sender with the input hash id
        return self.send_input_hash(input_hash)

    def block_if_buffer_full(self):

        # block vLLM if the KV cache sending buffer is full
        # TODO: allow using other policies to handle buffer full
        while True:
            self.input_hash_to_kv_sending_requests_lock.acquire()
            if len(self.input_hash_to_kv_sending_requests.keys()) > 40:
                self.input_hash_to_kv_sending_requests_lock.release()
                time.sleep(0.1)
            else:
                self.input_hash_to_kv_sending_requests_lock.release()
                break

@with_mark_steps
def send_kv_caches_and_hidden_states(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    hidden_states: torch.Tensor,
) -> None:
    # lists are not hashable while tuples are
    seq_lens = model_input.seq_lens
    slot_mappings = model_input.attn_metadata.slot_mapping
    input_hash_list = model_input.input_hash_tensor.tolist()

    # Assumption: current batch is all-prefill requests
    ps.get_disagg_group().input_hash_to_kv_sending_requests_lock.acquire()

    for idx in range(model_input.real_batch_size):
        seq_len = seq_lens[idx]
        slot_mapping = slot_mappings[idx]
        effective_slots = slot_mapping[:seq_len]
        input_hash = input_hash_list[idx]
        htorch.core.mark_step()

        for i in range(model_executable.model.start_layer, 
                       model_executable.model.end_layer):
            
            kv_cache = kv_caches[i]
            _, _, num_heads, head_size = kv_cache[0].shape

            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

            ps.get_disagg_group().kv_cache_send(
                input_hash, key_cache[effective_slots])
            ps.get_disagg_group().kv_cache_send(
                input_hash, value_cache[effective_slots])

        ps.get_disagg_group().kv_cache_send(
            input_hash, hidden_states)
        
        ps.get_disagg_group().kv_cache_send_ready(input_hash)

    ps.get_disagg_group().input_hash_to_kv_sending_requests_lock.release()

    ps.get_disagg_group().block_if_buffer_full()

    logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

@with_mark_steps
def recv_kv_caches_and_hidden_states(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor]
) -> Tuple[torch.Tensor, bool]:

    bypass_model_exec = True

    # lists are not hashable while tuples are
    seq_lens = model_input.seq_lens
    slot_mappings = model_input.attn_metadata.slot_mapping
    input_hash_list = model_input.input_hash_tensor

    # Assumption: current batch is all-prefill requests
    hidden_states_for_one_req = []

    # enumerate different requests
    # FIXME(Kuntai): This impl assumes that all requests are prefill.
    for idx in range(model_input.real_batch_size):
        seq_len = seq_lens[idx]
        slot_mapping = slot_mappings[idx]
        input_hash = input_hash_list[idx]

        # notify the prefill instance to start sending KVs associated with input_hash
        contain = ps.get_disagg_group().kv_cache_recv_start(input_hash)

        # fail to find input_hash in prefill instance
        # this can occur but idk why...
        if contain == 0:
            bypass_model_exec = False
            continue

        # receive KV cache from disaggregated prefill instance
        for i in range(model_executable.model.start_layer,
                       model_executable.model.end_layer):

            # get kv cache
            kv_cache = kv_caches[i - model_executable.model.start_layer]

            # get kv cache shape (after sliced by tp)
            _, _, num_heads, head_size = kv_cache[0].shape
            key = ps.get_disagg_group().kv_cache_recv(
                torch.Size([seq_len, num_heads, head_size]),
                kv_cache[0].dtype)
            value = ps.get_disagg_group().kv_cache_recv(
                torch.Size([seq_len, num_heads, head_size]),
                kv_cache[0].dtype)

            key_cache, value_cache = kv_cache[0], kv_cache[1]
            key_cache = key_cache.reshape(-1, num_heads, head_size)
            value_cache = value_cache.reshape(-1, num_heads, head_size)

            effective_slots = slot_mapping[:seq_len]
            key_cache[effective_slots] = key
            value_cache[effective_slots] = value
            
        hidden_states_for_one_req.append(
            ps.get_disagg_group().kv_cache_recv(torch.Size(
                [1, model_executable.config.hidden_size]),
                                                kv_cache[0].dtype))

    for i in range(model_input.batch_size_padded - model_input.real_batch_size):
        hidden_states_for_one_req.append(hidden_states_for_one_req[0])


    if not bypass_model_exec:
        # Some of the KV cache is not retrieved
        # so we need to recompute the hidden state
        return [], bypass_model_exec

    # concatenate hidden states from different requests
    hidden_states = torch.cat(hidden_states_for_one_req, dim=0)
    
    logger.debug("[rank%d]: KV recv DONE.", torch.distributed.get_rank())

    return hidden_states, bypass_model_exec