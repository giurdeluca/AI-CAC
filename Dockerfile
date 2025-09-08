FROM continuumio/miniconda3:latest
RUN conda update conda
RUN conda install python=3.9.19

# to make all requirements install
# Add build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

## Create the environment:
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Set the working directory
WORKDIR /app

COPY . .
RUN mkdir -p data output

ENTRYPOINT ["python3", "main_inference.py"]
