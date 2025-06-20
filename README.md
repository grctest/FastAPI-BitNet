# FastAPI-BitNet

This project provides a robust REST API built with FastAPI and Docker to manage and interact with `llama.cpp`-based BitNet model instances. It allows developers and researchers to programmatically control `llama-cli` processes for automated testing, benchmarking, and interactive chat sessions.

It serves as a backend replacement for the [Electron-BitNet](https://github.com/grctest/Electron-BitNet) project, offering enhanced performance, scalability, and persistent chat sessions.

## Key Features

*   **Session Management**: Start, stop, and check the status of multiple persistent `llama-cli` and `llama-server` session based chats.
*   **Batch Operations**: Initialize, shut down, and chat with multiple instances in a single API call.
*   **Interactive Chat**: Send prompts to running bitnet sessions and receive cleaned model responses.
*   **Model Benchmarking**: Programmatically run benchmarks and calculate perplexity on GGUF models.
*   **Resource Estimation**: Estimate maximum server capacity based on available system RAM and CPU threads.
*   **VS Code Integration**: Connects directly to GitHub Copilot Chat as a tool via the Model Context Protocol.
*   **Automatic API Docs**: Interactive API documentation powered by Swagger UI and ReDoc.

## Technology Stack

*   [FastAPI](https://github.com/fastapi/fastapi) for the core web framework.
*   [Uvicorn](https://www.uvicorn.org/) as the ASGI server.
*   [Docker](https://www.docker.com/) for containerization and easy deployment.
*   [Pydantic](https://docs.pydantic.dev/) for data validation and settings management.
*   [fastapi-mcp](https://github.com/tadata-org/fastapi_mcp) for VS Code Copilot tool integration.

---

## Getting Started

### Prerequisites

*   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
*   [Conda](https://www.anaconda.com/download) (or another Python environment manager)
*   Python 3.10+

### 1. Set Up the Python Environment

Create and activate a Conda environment:
```bash
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

---

## Running the Application

### Using Docker (Recommended)

This is the easiest and recommended way to run the application.

1.  **Build the Docker image:**
    ```bash
    docker build -t fastapi_bitnet .
    ```

2.  **Run the Docker container:**
    This command runs the container in detached mode (`-d`) and maps port 8080 on your host to port 8080 in the container.
    ```bash
    docker run -d --name ai_container -p 8080:8080 fastapi_bitnet
    ```

### Local Development

For development, you can run the application directly with Uvicorn, which enables auto-reloading.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## API Usage

Once the server is running, you can access the interactive API documentation:

*   **Swagger UI**: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)
*   **ReDoc**: [http://127.0.0.1:8080/redoc](http://127.0.0.1:8080/redoc)

---

## VS Code Integration

### As a Copilot Tool (MCP)

You can connect this API directly to VS Code's Copilot Chat to create and interact with models.

1.  Run the application using Docker or locally.
2.  In VS Code, open the Copilot Chat panel.
3.  Click the wrench icon ("Configure Tools...").
4.  Scroll to the bottom and select `+ Add MCP Server`, then choose `HTTP`.
5.  Enter the URL: `http://127.0.0.1:8080/mcp`

Copilot will now be able to use the API to launch and chat with BitNet instances.

### See Also - VSCode Extension!

For a more integrated experience, check out the companion VS Code extension:
*   **GitHub**: [https://github.com/grctest/BitNet-VSCode-Extension](https://github.com/grctest/BitNet-VSCode-Extension)
*   **Marketplace**: [https://marketplace.visualstudio.com/items?itemName=nftea-gallery.bitnet-vscode-extension](https://marketplace.visualstudio.com/items?itemName=nftea-gallery.bitnet-vscode-extension)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
