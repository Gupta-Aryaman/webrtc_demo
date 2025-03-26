# signaling_server.py
import asyncio
import json
import logging
import sys
import subprocess
import os
import threading
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('signaling-server')

# Create FastAPI app
app = FastAPI(title="WebRTC Signaling Server")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# A simple set to store connected clients
connected_clients: List[WebSocket] = []
python_client_process: Optional[subprocess.Popen] = None

# Class to manage WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        client_id = id(websocket)
        logger.info(f"New client connected: {client_id}. Total clients: {len(self.active_connections)}")
        return client_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = id(websocket)
            logger.info(f"Client disconnected: {client_id}. Remaining clients: {len(self.active_connections)}")

    async def broadcast(self, message: str, sender: WebSocket):
        sender_id = id(sender)
        for client in self.active_connections:
            if client != sender:
                try:
                    await client.send_text(message)
                    data = json.loads(message)
                    logger.info(f"Forwarded {data['type']} message from client {sender_id} to client {id(client)}")
                except Exception as e:
                    logger.error(f"Error forwarding message to client {id(client)}: {e}")

manager = ConnectionManager()

# Function to monitor and log output from the Python client process
def log_python_client_output(process: subprocess.Popen):
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"Python client: {line.strip()}")
            else:
                break

# Endpoint to start the Python client
@app.get("/start_client")
async def start_client(
    background_tasks: BackgroundTasks, 
    source: str = "test_pattern",
    model: str = "",
    model_proc: str = ""
):
    # Instead of starting the Python client, just return instructions
    logger.info(f"Received request to start client with source: {source}, model: {model}, model_proc: {model_proc}")
    
    # Return instructions for running the client in Docker
    docker_cmd = f"docker run -it --rm --network host -v /path/to/models:/home/dlstreamer/models -v $(pwd):/home/dlstreamer/webrtc_demo --user root --entrypoint /bin/bash intel/dlstreamer:latest -c \"cd /home/dlstreamer/webrtc_demo && pip install -r requirements.txt && python python_client.py --source {source}"
    
    if model:
        docker_cmd += f" --model {model}"
    if model_proc:
        docker_cmd += f" --model-proc {model_proc}"
    
    docker_cmd += " --server-url ws://localhost:8080/ws\""
    
    return {
        "status": "instructions",
        "message": "To run the Python client in Docker, use the following command:",
        "docker_command": docker_cmd,
        "note": "The Python client should be run in the Docker container manually, not from the signaling server."
    }

# WebSocket endpoint for signaling
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    
    try:
        while True:
            # Receive and process messages
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                logger.info(f"Received {data['type']} message from client {client_id}")
                
                # Broadcast the message to all other clients
                await manager.broadcast(message, websocket)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error from client {client_id}: {e}")
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {e}")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
        manager.disconnect(websocket)

# Root endpoint to provide basic info
@app.get("/")
async def root():
    return {
        "name": "WebRTC Signaling Server",
        "endpoints": {
            "websocket": "/ws",
            "start_client": "/start_client?source=test_pattern"
        },
        "status": "running"
    }

# Shutdown event handler
@app.on_event("shutdown")
def shutdown_event():
    global python_client_process
    logger.info("Server shutting down")
    
    # Clean up Python client if running
    if python_client_process and python_client_process.poll() is None:
        logger.info("Terminating Python client process")
        python_client_process.terminate()

# Run the server
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="localhost", port=8080, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server shutting down")
        # Clean up Python client if running
        if python_client_process and python_client_process.poll() is None:
            python_client_process.terminate()
    except Exception as e:
        logger.error(f"Server error: {e}")
