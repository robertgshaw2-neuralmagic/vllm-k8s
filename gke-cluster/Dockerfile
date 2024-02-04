FROM vllm/vllm-openai

RUN apt update && apt install git -y && apt install git-lfs -y

RUN git lfs install

COPY download_model.py download_model.py

RUN python3 download_model.py --model-id "HuggingFaceH4/zephyr-7b-beta" --ignore-patterns "*.bin"

RUN ["/bin/bash"]