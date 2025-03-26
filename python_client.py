# python_client.py
import asyncio
import json
import websockets
import logging
import sys
import fractions
import time
import threading
import argparse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Python WebRTC client')
parser.add_argument('--source', default='test_pattern', help='Video source (test_pattern, file path, or camera index)')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('python-client')

# Create a video track that can stream video from a file or camera
class VideoFileTrack(MediaStreamTrack):
    kind = "video"
    
    def __init__(self, source='test_pattern'):
        super().__init__()
        self._frame_counter = 0
        self._start_time = time.time()
        self.source = source
        
        # Initialize video source
        if source == 'test_pattern':
            # Create a test pattern with moving elements
            self._create_test_pattern = True
        else:
            # Use a video file or camera as source
            try:
                self.player = MediaPlayer(source)
                self.video_track = self.player.video
                self._create_test_pattern = False
                logger.info(f"Using video source: {source}")
            except Exception as e:
                logger.error(f"Failed to open video source {source}: {e}")
                self._create_test_pattern = True
                logger.info("Falling back to test pattern")
    
    async def next_timestamp(self):
        self._frame_counter += 1
        # Use a standard 30fps timing
        pts = int(self._frame_counter * 90000 / 30)  # 90kHz clock rate
        time_base = fractions.Fraction(1, 90000)  # standard time base for video
        return pts, time_base
    
    async def recv(self):
        from av import VideoFrame
        import numpy as np
        
        if not self._create_test_pattern:
            try:
                # Get frame from the actual video source
                frame = await self.video_track.recv()
                return frame
            except Exception as e:
                logger.error(f"Error getting video frame: {e}")
                # Fall back to test pattern if video source fails
                self._create_test_pattern = True
        
        # Create a test pattern with moving elements
        pts, time_base = await self.next_timestamp()
        
        # Create a more visually interesting test pattern
        height, width = 480, 640
        frame_data = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a moving element
        t = time.time() - self._start_time
        x = int((width - 100) * (0.5 + 0.5 * np.sin(t)))
        y = int((height - 100) * (0.5 + 0.5 * np.cos(t)))
        
        # Draw a colored rectangle that moves around
        frame_data[y:y+100, x:x+100, 0] = 255  # Red
        
        # Add some text
        font_size = 1
        thickness = 2
        color = (255, 255, 255)  # White
        
        # Draw grid pattern in background
        for i in range(0, height, 40):
            frame_data[i:i+2, :, 1] = 128  # Green horizontal lines
        for i in range(0, width, 40):
            frame_data[:, i:i+2, 1] = 128  # Green vertical lines
            
        # Create the frame
        frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

# Global variables
pc = None
ws = None
loop = None
should_offer = True  # Flag to determine if this client should initiate the offer

# Function to create a new peer connection
def create_peer_connection():
    global pc
    
    # Close existing connection if it exists
    if pc:
        logger.info("Closing existing peer connection")
        loop.run_until_complete(pc.close())
    
    # Create a new RTCPeerConnection
    pc = RTCPeerConnection()
    logger.info("Created new peer connection")
    
    # Add a video track to the connection
    # Use the source specified in command-line arguments
    video_track = VideoFileTrack(source=args.source)
    pc.addTrack(video_track)
    logger.info(f"Added video track to connection with source: {args.source}")
    
    # Set up event handlers
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            logger.info(f"New ICE candidate: {candidate.candidate}")
            await send_signaling_message({
                "type": "candidate",
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex,
                "candidate": candidate.candidate
            })
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state changed to: {pc.iceConnectionState}")
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state changed to: {pc.connectionState}")
    
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track: {track.kind}")
        if track.kind == "video":
            logger.info("Received video track")
        elif track.kind == "audio":
            logger.info("Received audio track")
    
    return pc

async def send_signaling_message(message):
    try:
        if ws:
            await ws.send(json.dumps(message))
            logger.info(f"Sent signaling message: {message['type']}")
        else:
            logger.error("WebSocket not open, cannot send message")
    except Exception as e:
        logger.error(f"Error sending signaling message: {e}")

async def handle_offer(data):
    global pc, should_offer
    
    logger.info("Processing received offer")
    
    # If we're in a state where we can't handle an offer, create a new connection
    if not pc or pc.signalingState != "stable":
        logger.info("Creating new peer connection to handle offer")
        pc = create_peer_connection()
    
    # Set remote description (the offer)
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    await pc.setRemoteDescription(offer)
    logger.info("Set remote description (offer)")
    
    # Create an answer
    logger.info("Creating answer")
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Set local description (answer)")
    
    # Send the answer
    await send_signaling_message({
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp
    })
    
    # We've received an offer, so we shouldn't send our own
    should_offer = False

async def handle_answer(data):
    global pc
    
    if pc and pc.signalingState == "have-local-offer":
        logger.info("Processing received answer")
        answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        await pc.setRemoteDescription(answer)
        logger.info("Set remote description (answer)")
    else:
        logger.warning(f"Ignoring answer in signaling state: {pc.signalingState if pc else 'No connection'}")

async def handle_candidate(data):
    global pc
    
    if pc and pc.remoteDescription:
        logger.info(f"Processing received ICE candidate")
        try:
            # In aiortc, the RTCIceCandidate constructor expects the candidate string
            # without the 'candidate:' prefix that might be present in the browser's format
            candidate_str = data["candidate"]
            if candidate_str and candidate_str.startswith("candidate:"):
                candidate_str = candidate_str[10:]  # Remove 'candidate:' prefix
            
            if candidate_str:
                candidate = RTCIceCandidate(
                    sdpMid=data["sdpMid"],
                    sdpMLineIndex=data["sdpMLineIndex"],
                    candidate=candidate_str
                )
                
                await pc.addIceCandidate(candidate)
                logger.info("ICE candidate added successfully")
            else:
                logger.warning("Received empty ICE candidate, ignoring")
        except Exception as e:
            logger.error(f"Error adding ICE candidate: {e}")
    else:
        logger.warning("Ignoring ICE candidate, no remote description set")

async def handle_signaling():
    global ws, pc, should_offer
    
    try:
        while True:
            try:
                message = await ws.recv()
                data = json.loads(message)
                logger.info(f"Received signaling message: {data['type']}")
                
                if data["type"] == "offer":
                    await handle_offer(data)
                elif data["type"] == "answer":
                    await handle_answer(data)
                elif data["type"] == "candidate":
                    await handle_candidate(data)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON message")
            except KeyError:
                logger.error("Message missing required fields")
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error handling signaling message: {e}")
    except Exception as e:
        logger.error(f"Signaling handler error: {e}")
    finally:
        # Clean up if the signaling connection is lost
        if pc:
            logger.info("Closing peer connection due to signaling connection loss")
            await pc.close()

async def create_offer():
    global pc, should_offer
    
    if should_offer:
        logger.info("Creating offer")
        try:
            # Create a new peer connection if needed
            if not pc or pc.signalingState != "stable":
                pc = create_peer_connection()
            
            # Create and send the offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            logger.info("Set local description (offer)")
            
            await send_signaling_message({
                "type": pc.localDescription.type,
                "sdp": pc.localDescription.sdp
            })
            logger.info("Sent offer")
        except Exception as e:
            logger.error(f"Error creating offer: {e}")
            raise

async def run():
    global ws, pc, loop, should_offer
    
    # Get the event loop
    loop = asyncio.get_running_loop()
    
    try:
        # Connect to the signaling server
        signaling_server = "ws://localhost:8080/ws"  # Updated to use the FastAPI WebSocket endpoint
        logger.info(f"Connecting to signaling server at {signaling_server}")
        
        ws = await websockets.connect(signaling_server)
        logger.info("Connected to signaling server")
        
        # Create the initial peer connection
        pc = create_peer_connection()
        
        # Start the signaling handler in background
        signaling_task = asyncio.create_task(handle_signaling())
        
        # Wait a bit before sending an offer to allow the frontend to connect
        await asyncio.sleep(2)
        
        # Create and send offer if we should initiate
        if should_offer:
            await create_offer()
        
        # Keep the connection open
        logger.info("Waiting for signaling task to complete")
        await signaling_task
    except Exception as e:
        logger.error(f"Error in run function: {e}")

if __name__ == "__main__":
    try:
        logger.info("Starting Python WebRTC client")
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Client shutting down")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
