FROM python:3.11-slim-bookworm

RUN pip install huggingface_hub[cli]

COPY download_from_hub.sh /home/download_from_hub.sh 

ENTRYPOINT [ "bash", "/home/download_from_hub.sh" ] 
