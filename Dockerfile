# First stage: build dependencies
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Install Lambda web adapter in case you want to run with with an AWS Lamba function URL
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.3 /lambda-adapter /opt/extensions/lambda-adapter

# Install wget
RUN apt-get update && apt-get install -y wget

# Create a directory for the model
RUN mkdir /model

WORKDIR /src

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Gradio needs to be installed after due to conflict with spacy in requirements
RUN pip install --no-cache-dir gradio==4.32.2 

# Download the BGE embedding model during the build process
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs -y
RUN git lfs install
RUN git clone https://huggingface.co/BAAI/bge-small-en-v1.5 /model/bge
RUN rm -rf /model/bge/.git

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Change ownership of /home/user directory
RUN chown -R user:user /home/user

# Make output folder
RUN mkdir -p /home/user/app/output && chown -R user:user /home/user/app/output

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=7860 \
	GRADIO_THEME=huggingface \
	AWS_STS_REGIONAL_ENDPOINT=regional \
	#GRADIO_ROOT_PATH=/data-text-search \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app
#COPY . $HOME/app

CMD ["python", "app.py"]