# FastAPI-BitNet

This project uses a combination of [Uvicorn](https://www.uvicorn.org/), [FastAPI](https://fastapi.tiangolo.com/) (Python) and [Docker](https://www.docker.com/) to provide a reliable REST API for testing [Microsoft's BitNet](https://github.com/microsoft/BitNet) out locally!

It supports running the inference framework, running BitNet model benchmarks and calculating BitNet model perplexity values.

It's offers the same functionality as the [Electron-BitNet](https://github.com/grctest/Electron-BitNet) project, however it does so through a REST API which devs/researchers can use to automate testing/benchmarking of 1-bit BitNet models!

## Setup instructions

If running in dev mode, run Docker Desktop on windows to initialize docker in WSL2.

Launch WSL: `wsl`

Install Conda: https://anaconda.org/anaconda/conda

Initialize the python environment:
```
conda init
conda create -n bitnet python=3.11
conda activate bitnet
```

Install the Huggingface-CLI tool to download the models:
```
pip install -U "huggingface_hub[cli]"
```
 
Download Microsoft's official BitNet model:
```
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir app/models/BitNet-b1.58-2B-4T
```

Build the docker image:
```
docker build -t fastapi_bitnet .
```

Run the docker image:
```
docker run -d --name ai_container -p 8080:8080 fastapi_bitnet
```

Once it's running navigate to http://127.0.0.1:8080/docs

## Docker hub repository

You can fetch the dockerfile at: https://hub.docker.com/repository/docker/grctest/fastapi_bitnet/general

## How to add to VSCode!

Run the dockerfile locally using the command above, then navigate to the VSCode Copilot chat window and find the wrench icon "Configure Tools...".

In the tool configuration overview scroll to the bottom and select 'Add more tools...' then '+ Add MCP Server' then 'HTTP'.

Enter into the URL field `http://127.0.0.1:8080/mcp` then your copilot will be able to launch new bitnet server instances and chat with them.