# WebRTC Demo

A simple WebRTC demo with a Python client streaming video to a web browser client through a FastAPI-based signaling server.

## Components

1. **Signal Server** (`signal_server.py`): FastAPI server that handles WebSocket signaling and provides HTTP endpoints to start the Python client
2. **Python Client** (`python_client.py`): Generates and streams video content
3. **Frontend Client** (`frontend_client.html`): Displays the video stream in a web browser

## Setup and Usage

### Prerequisites

- Python 3.7+
- Required Python packages:
  ```
  pip install fastapi uvicorn websockets aiortc opencv-python numpy av
  ```

### Running the Demo

1. **Start the Signal Server**:
   ```
   uvicorn signal_server:app --host 0.0.0.0 --port 8080
   ```
   This starts the FastAPI server on port 8080, which handles both HTTP requests and WebSocket connections.

2. **Open the Frontend Client**:
   Open `frontend_client.html` in a web browser. You can use a simple HTTP server:
   ```
   python -m http.server 8000
   ```
   Then navigate to http://localhost:8000/frontend_client.html

3. **Start the Python Client from the UI**:
   - Select a video source from the dropdown (test pattern or camera)
   - Click the "Start Python Client" button
   - Once the Python client is running, click "Start Connection" to establish the WebRTC connection

### Video Source Options

- **Test Pattern**: A moving test pattern generated by the Python client
- **Camera**: Use the first camera device (index 0)
- **Custom Video File**: You can add more options in the dropdown for specific video files

## How It Works

1. The frontend connects to the signal server via WebSocket at `ws://localhost:8080/ws`
2. When "Start Python Client" is clicked, an HTTP request is sent to `http://localhost:8080/start_client`
3. The signal server starts the Python client as a subprocess
4. The Python client connects to the signal server via WebSocket
5. When "Start Connection" is clicked, WebRTC signaling (offers, answers, ICE candidates) is exchanged through the signal server
6. Once the WebRTC connection is established, video streams directly from the Python client to the browser

## API Documentation

The FastAPI server automatically generates API documentation. You can access it at:
- http://localhost:8080/docs (Swagger UI)
- http://localhost:8080/redoc (ReDoc)

## Troubleshooting

- If the video doesn't appear, check the browser console and Python logs for errors
- Make sure all required Python packages are installed
- For camera access issues, verify that your camera is working and accessible to Python

WebRTC Demo Architecture Walkthrough
I'll walk you through your WebRTC demo system, explaining each component's role and how they work together to establish a WebRTC connection.

System Overview
Your WebRTC demo consists of three main components:

Signal Server (signal_server.py): Facilitates the exchange of connection information between peers
Python Client (python_client.py): Generates and streams video content
Frontend Client (frontend_client.html): Displays the video stream in a web browser
Let me explain each component and the overall data flow:

1. Signal Server (signal_server.py)
Role: Acts as an intermediary for WebRTC signaling between the Python client and frontend client.

Key Functions:

Maintains WebSocket connections with all clients
Relays messages between clients without interpreting them
Doesn't participate in the actual media streaming
How It Works:

Runs a FastAPI server on localhost:8080
When a client connects, adds it to a set of active clients
When a message is received from one client, forwards it to all other clients
Handles client disconnections by removing them from the active set
The signal server is essential because WebRTC peers need to exchange information about how to connect directly to each other, but they initially have no way to communicate.

2. Python Client (python_client.py)
Role: Acts as the media source, generating and streaming video content over WebRTC.

Key Components:

VideoFileTrack Class:
Generates video frames (either from a test pattern or a real video source)
Implements the MediaStreamTrack interface required by aiortc
Provides timing and frame generation logic
RTCPeerConnection Management:
Creates and manages the WebRTC peer connection
Adds the video track to the connection
Handles ICE candidates, connection state changes, etc.
Signaling Logic:
Connects to the signaling server via WebSocket
Sends and receives signaling messages (offers, answers, ICE candidates)
Processes incoming signaling messages to establish the WebRTC connection
Data Flow Within Python Client:

Connects to the signaling server
Creates a peer connection and adds a video track
Generates an offer or processes an incoming offer
Exchanges ICE candidates through the signaling server
Once connected, continuously generates video frames and sends them over the established WebRTC connection
3. Frontend Client (frontend_client.html)
Role: Receives and displays the video stream in a web browser.

Key Components:

User Interface:
Video element to display the received stream
Buttons for starting/reconnecting
Status display and logging area
WebRTC Logic:
Creates and manages the browser's RTCPeerConnection
Handles incoming tracks from the Python client
Processes signaling messages (offers, answers, ICE candidates)
Signaling Logic:
Connects to the signaling server via WebSocket
Sends and receives signaling messages
Processes incoming signaling messages to establish the WebRTC connection
Data Flow Within Frontend Client:

Connects to the signaling server
When "Start Connection" is clicked, creates a peer connection
Processes offers from the Python client and generates answers
Exchanges ICE candidates through the signaling server
Once connected, receives video frames and displays them in the video element
Complete Data and Request Flow
1. Initialization Phase:
```
Signal Server                Python Client                Frontend Client
    |                             |                             |
    |<------- WebSocket --------->|                             |
    |                             |                             |
    |<------------------------ WebSocket ---------------------->|
```
Both clients connect to the signal server via WebSockets
The signal server maintains these connections for message relay
2. Signaling Phase:
```
Python Client                Signal Server                Frontend Client
    |                             |                             |
    |------ Offer Message ------->|                             |
    |                             |------ Offer Message ------->|
    |                             |                             |
    |                             |<----- Answer Message -------|
    |<----- Answer Message -------|                             |
    |                             |                             |
    |---- ICE Candidate Event --->|                             |
    |                             |---- ICE Candidate Event --->|
    |                             |                             |
    |                             |<--- ICE Candidate Event ----|
    |<--- ICE Candidate Event ----|                             |
```
The Python client initiates by sending an offer (SDP) through the signal server
The frontend client receives the offer, processes it, and sends back an answer
Both clients exchange ICE candidates to find the best connection path
All these messages pass through the signal server, which simply relays them
3. Direct WebRTC Connection:
```
Python Client                                      Frontend Client
    |                                                    |
    |<---------------- Direct WebRTC Connection -------->|
    |                                                    |
    |-------------------- Video Stream ------------------>|
```
Once the connection is established, the signal server is no longer needed
The Python client sends video frames directly to the frontend client
The connection is peer-to-peer, with no intermediary server for the media
WebRTC Connection Establishment Details
ICE (Interactive Connectivity Establishment):
Both peers gather ICE candidates (potential connection methods)
Candidates include direct IP connections, STUN-derived addresses, or TURN relays
These candidates are exchanged through the signaling server
The ICE protocol tries each candidate pair until it finds one that works
SDP (Session Description Protocol):
The offer and answer contain SDP information
SDP describes media capabilities: codecs, formats, network details
The Python client's offer includes information about its video track
The frontend client's answer includes its supported formats and preferences
Media Flow:
Once connected, the Python client's VideoFileTrack.recv() method is called repeatedly
It generates frames (either test pattern or from a video source)
These frames are encoded and sent over the WebRTC connection
The frontend client receives these frames and displays them in the video element
Summary
Your WebRTC system demonstrates a complete video streaming solution with:

Signaling Server: Facilitates initial connection setup
Python Client: Generates and streams video content
Frontend Client: Receives and displays the video
The beauty of WebRTC is that once the connection is established, the video streams directly between peers without going through a server, making it efficient and low-latency. The signal server is only needed for the initial handshake and can even be disconnected once the WebRTC connection is established.