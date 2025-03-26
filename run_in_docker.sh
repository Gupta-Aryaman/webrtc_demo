#!/bin/bash

# Set default paths
MODELS_PATH=${MODELS_PATH:-"/path/to/your/models"}
ASSETS_PATH=${ASSETS_PATH:-"$(pwd)"}

# Default model paths - update these with your actual model paths
MODEL=${MODEL:-"/home/dlstreamer/models/model.xml"}
MODEL_PROC=${MODEL_PROC:-"/home/dlstreamer/models/model_proc.json"}

# Default RTSP URL
RTSP_URL=${RTSP_URL:-"rtsp://localhost:8554/input_stream"}

# Default signaling server URL
SERVER_URL=${SERVER_URL:-"ws://localhost:8080/ws"}

# Run the Python client in the Docker container
docker run -it --rm --network host \
  -v ${MODELS_PATH}:/home/dlstreamer/models \
  -v ${ASSETS_PATH}:/home/dlstreamer/webrtc_demo \
  --env MODELS_PATH=/home/dlstreamer/models \
  --user root \
  --entrypoint /bin/bash \
  intel/dlstreamer:latest \
  -c "cd /home/dlstreamer/webrtc_demo && pip install -r requirements.txt && chmod -R 755 /home/dlstreamer && exec setpriv --reuid=$(id -u) --regid=$(id -g) --init-groups /bin/bash -c 'python python_client.py --model ${MODEL} --model-proc ${MODEL_PROC} --rtsp-url ${RTSP_URL} --server-url ${SERVER_URL}'"

# Note: To run the client separately, use:
# docker exec -it <container_id> /bin/bash -c "cd /home/dlstreamer/webrtc_demo && python python_client.py --model /home/dlstreamer/models/your_model.xml --model-proc /home/dlstreamer/models/your_model_proc.json"
