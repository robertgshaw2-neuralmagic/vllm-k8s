FROM python:3.10-slim-bookworm

RUN apt update

RUN pip install huggingface_hub[cli]

COPY download_from_hub.sh /run/download_from_hub.sh

ENTRYPOINT [ "bash", "/run/download_from_hub.sh" ]