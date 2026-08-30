ARG AI_TOOLKIT_IMAGE=ostris/aitoolkit@sha256:f06882439bcbc7073bad75f8dc4e6991ace83f320433b0ff409fd90cfc9c31a8
FROM ${AI_TOOLKIT_IMAGE}

ARG AI_TOOLKIT_REVISION=be995185f598c83abb990a088e9f634c4d36eb46

USER root
COPY requirements.txt /app/hartsy-ai-toolkit-worker/requirements.txt
RUN rm -rf /app/ai-toolkit \
    && git clone https://github.com/ostris/ai-toolkit.git /app/ai-toolkit \
    && cd /app/ai-toolkit \
    && git checkout "${AI_TOOLKIT_REVISION}" \
    && pip install --no-cache-dir --break-system-packages -r /app/ai-toolkit/requirements.txt -r /app/hartsy-ai-toolkit-worker/requirements.txt \
    && git rev-parse HEAD | grep -Fx "${AI_TOOLKIT_REVISION}"

COPY handler.py /app/hartsy-ai-toolkit-worker/handler.py
WORKDIR /app/hartsy-ai-toolkit-worker
ENV AI_TOOLKIT_ROOT=/app/ai-toolkit AI_TOOLKIT_REVISION=${AI_TOOLKIT_REVISION} PYTHONUNBUFFERED=1
CMD ["python", "-u", "handler.py"]
