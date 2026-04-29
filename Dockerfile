FROM mambaorg/micromamba:1.5.6-focal-cuda-12.1.1

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/env.yml

RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

WORKDIR /app

COPY ./model_core ./model_core
COPY ./model.py .
COPY ./s3_upload.py .
COPY ./generate_thumbnails.py .
COPY ./oceanlens_inference.py .
COPY ./run_oceanlens_inference.py .

CMD ["python", "run_oceanlens_inference.py"]
