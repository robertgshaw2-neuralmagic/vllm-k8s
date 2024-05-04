FROM python:3.10-slim-bookworm

RUN apt update && apt install huggingface_hub[cli]

COPY download_model.sh /run/download_model.sh

ENTRYPOINT [ "/bin/bash" ]