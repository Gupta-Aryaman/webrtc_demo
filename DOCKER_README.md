# Running WebRTC Demo with DL Streamer in Docker

This guide explains how to run the WebRTC application with DL Streamer, keeping the signaling server on the host machine and running only the Python client in the Docker container.

## Prerequisites

- Docker installed on your system
- Intel DL Streamer models available locally
- Python environment on the host machine for the signaling server

## Setup

1. Set the environment variables for your model paths:

```bash
export MODELS_PATH=/path/to/your/models
```

2. Make the run script executable:

```bash
chmod +x run_in_docker.sh
```

3. Install the required dependencies for the signaling server on the host machine:

```bash
pip install websockets fastapi uvicorn
```

## Running the Application

### Step 1: Start the Signaling Server on the Host

Start the signaling server on your host machine:

```bash
python signal_server.py
```

This will start the signaling server on port 8080 by default.

### Step 2: Run the Python Client in Docker

Use the provided script to run the Python client in the Docker container:

```bash
./run_in_docker.sh
```

You can customize the environment variables to change the default settings:

```bash
# Set custom model paths
export MODEL=/home/dlstreamer/models/custom_model.xml
export MODEL_PROC=/home/dlstreamer/models/custom_model_proc.json

# Set custom RTSP URL
export RTSP_URL=rtsp://custom-server:8554/stream

# Set custom signaling server URL (if not running on localhost:8080)
export SERVER_URL=ws://custom-host:8080/ws

./run_in_docker.sh
```

### Manual Docker Command

If you prefer to run the commands manually:

```bash
docker run -it --rm --network host \
  -v ${MODELS_PATH}:/home/dlstreamer/models \
  -v $(pwd):/home/dlstreamer/webrtc_demo \
  --env MODELS_PATH=/home/dlstreamer/models \
  --user root \
  --entrypoint /bin/bash \
  intel/dlstreamer:latest \
  -c "cd /home/dlstreamer/webrtc_demo && pip install -r requirements.txt && chmod -R 755 /home/dlstreamer && exec setpriv --reuid=$(id -u) --regid=$(id -g) --init-groups /bin/bash"

# Inside the container, run the Python client
python python_client.py --model /home/dlstreamer/models/your_model.xml --model_proc /home/dlstreamer/models/your_model_proc.json --rtsp_url rtsp://localhost:8554/input_stream --server_url ws://localhost:8080/ws
```

## Accessing the Application

Once both the signaling server and Python client are running, open your browser and navigate to:

```
http://localhost:8080
```

## Troubleshooting

If you encounter any issues:

1. Make sure the Docker container has network access (using `--network host`)
2. Verify that the model paths are correctly mounted in the container
3. Check that all required Python dependencies are installed
4. Ensure the RTSP source is available at the expected URL
5. Confirm that the signaling server is running and accessible from the Docker container

## Notes

- The signaling server uses port 8080 by default
- WebRTC connections are established peer-to-peer
- The DL Streamer pipeline processes video frames from an RTSP source
- The Python client in the Docker container needs to be able to connect to the signaling server on the host
