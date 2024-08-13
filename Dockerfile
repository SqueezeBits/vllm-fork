FROM vault.habana.ai/gaudi-docker/1.16.2/ubuntu22.04/habanalabs/pytorch-installer-2.2.2 as vllm-base

ENV MODEL Meta-Llama-3-8B-Instruct
ENV SEQLEN 2048
ENV MEMUTIL 0.9
ENV BS 256
ENV NUMHPU 1
ENV BLOCK_SIZE 16

RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-hpu.txt requirements-hpu.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm
COPY benchmarks benchmarks

RUN pip3 install -e .

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'

ENV VLLM_USAGE_SOURCE production-docker-image
#################### OPENAI API SERVER ####################

EXPOSE 8000
