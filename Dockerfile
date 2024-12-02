# First stage: build dependencies
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm AS builder

# Optional - install Lambda web adapter in case you want to run with with an AWS Lamba function URL
# COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Update apt
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Create a directory for the model
RUN mkdir -p /model /model/minilm /install

WORKDIR /src

COPY requirements_aws.txt .

RUN pip install torch==2.5.1+cpu --target=/install --index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --target=/install sentence-transformers==3.3.1 --no-deps \
&& pip install --no-cache-dir --target=/install -r requirements_aws.txt \
&& pip install --no-cache-dir --target=/install gradio==5.6.0

# Add /install to the PYTHONPATH
ENV PYTHONPATH="/install:${PYTHONPATH}"

# Download the embedding model during the build process. Create a directory for the model and download specific files using huggingface_hub
COPY download_model.py /src/download_model.py
RUN python /src/download_model.py

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Change ownership of /home/user directory
RUN chown -R user:user /home/user

# Make output folder
RUN mkdir -p /home/user/app/output && mkdir -p /home/user/.cache/huggingface/hub && chown -R user:user /home/user

# Copy models from the builder stage
COPY --from=builder /model/minilm /home/user/app/model/minilm

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=7860 \
	GRADIO_ANALYTICS_ENABLED=False \
	GRADIO_THEME=huggingface \
	AWS_STS_REGIONAL_ENDPOINT=regional \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["python", "app.py"]