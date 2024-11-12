# FastAPI-BitNet

Install Conda: https://anaconda.org/anaconda/conda

Initialize the python environment:
```
conda init
conda create -n bitnet python=3.9
conda activate bitnet
```

Install the Huggingface-CLI tool to download the models:
```
pip install -U "huggingface_hub[cli]"
```

Download one/many of the 1-bit models from Huggingface below:
```
huggingface-cli download 1bitLLM/bitnet_b1_58-large --local-dir app/models/bitnet_b1_58-large
huggingface-cli download 1bitLLM/bitnet_b1_58-3B --local-dir app/models/bitnet_b1_58-3B
huggingface-cli download HF1BitLLM/Llama3-8B-1.58-100B-tokens --local-dir app/models/Llama3-8B-1.58-100B-tokens
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

---

Note:

If seeking to use this in production, make sure to extend the docker image with additional [authentication security](https://github.com/mjhea0/awesome-fastapi?tab=readme-ov-file#auth) steps. In its current state it's intended for use locally.

Building the docker file image requires upwards of 40GB RAM for `Llama3-8B-1.58-100B-tokens`, if you have less than 64GB RAM you will probably run into issues.

The Dockerfile deletes the larger f32 files, so as to reduce the time to build the docker image file, you'll need to comment out the `find /code/models/....` lines if you want the larger f32 files included.