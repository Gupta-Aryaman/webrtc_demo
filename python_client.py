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
import gi
import numpy as np
from av import VideoFrame
from queue import Queue
from threading import Thread

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer

# Initialize GStreamer
Gst.init(None)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Python WebRTC client')
parser.add_argument('--source', default='test_pattern', help='Video source (test_pattern, file path, camera index, or rtsp_pipeline)')
parser.add_argument('--model', default='', help='Path to the model for DL Streamer')
parser.add_argument('--model-proc', default='', help='Path to the model_proc file for DL Streamer')
parser.add_argument('--rtsp-url', default='rtsp://localhost:8554/input_stream', help='RTSP URL for streaming')
parser.add_argument('--server-url', default='ws://localhost:8080/ws', help='WebSocket URL for the signaling server')
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

class GStreamerTrack(MediaStreamTrack):
    """A video stream track that uses GStreamer to run DL Streamer pipeline and stream the output via WebRTC."""
    kind = "video"

    def __init__(self, rtsp_url, model=None, model_proc=None):
        super().__init__()
        self._frame_counter = 0
        self._start_time = time.time()
        self.rtsp_url = rtsp_url
        self.model = model
        self.model_proc = model_proc
        self.frame_queue = Queue(maxsize=30)  # Buffer for frames
        
        # Create and start the GStreamer pipeline in a separate thread
        self.pipeline_thread = Thread(target=self._run_pipeline)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        
        logger.info(f"GStreamerTrack initialized with RTSP URL: {rtsp_url}")
        if model and model_proc:
            logger.info(f"Using model: {model}, model_proc: {model_proc}")
    
    def _run_pipeline(self):
        try:
            # Create a GStreamer pipeline that will run inference and then output frames to our queue
            pipeline_str = self._create_pipeline_string()
            logger.info(f"Starting GStreamer pipeline: {pipeline_str}")
            
            # Create the pipeline
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Get the appsink element to retrieve frames
            self.appsink = self.pipeline.get_by_name("appsink")
            self.appsink.set_property("emit-signals", True)
            self.appsink.connect("new-sample", self._on_new_sample)
            
            # Create a GLib main loop to run the GStreamer pipeline
            self.loop = GLib.MainLoop()
            
            # Add a bus watch to handle pipeline messages
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            
            # Start the pipeline
            self.pipeline.set_state(Gst.State.PLAYING)
            
            # Run the GLib main loop
            self.loop.run()
        except Exception as e:
            logger.error(f"Error in GStreamer pipeline: {e}")
    
    def _create_pipeline_string(self):
        # Base pipeline with RTSP source
        pipeline = f"rtspsrc location={self.rtsp_url} protocols=tcp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert"
        
        # Add inference elements if model is provided
        if self.model and self.model_proc:
            pipeline += f" ! gvadetect model={self.model} model_proc={self.model_proc} device=CPU"
            pipeline += " ! gvafpscounter interval=1"
            pipeline += " ! gvapython module=assets/scripts/zone_detection.py"
            pipeline += " ! gvametaconvert ! gvawatermark"
        
        # Add appsink to capture frames for WebRTC
        pipeline += " ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink max-buffers=1 drop=true sync=false"
        
        return pipeline
    
    def _on_new_sample(self, appsink):
        sample = appsink.pull_sample()
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            caps_structure = caps.get_structure(0)
            
            # Get width and height from caps
            width = caps_structure.get_value("width")
            height = caps_structure.get_value("height")
            
            # Extract buffer data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Create numpy array from buffer data
                frame_data = np.ndarray(
                    shape=(height, width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Put the frame in the queue for the WebRTC track to consume
                try:
                    # Make a copy of the frame data to avoid issues with buffer being reused
                    self.frame_queue.put(frame_data.copy(), block=False)
                except:
                    # Queue is full, drop the frame
                    pass
                
                # Unmap the buffer
                buffer.unmap(map_info)
            
        return Gst.FlowReturn.OK
    
    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End of stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer error: {err}, {debug}")
            self.loop.quit()
    
    async def next_timestamp(self):
        self._frame_counter += 1
        # Use a standard 30fps timing
        pts = int(self._frame_counter * 90000 / 30)  # 90kHz clock rate
        time_base = fractions.Fraction(1, 90000)  # standard time base for video
        return pts, time_base
    
    async def recv(self):
        try:
            # Try to get a frame from the queue with a timeout
            frame_data = None
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
            except:
                # If no frame is available, create a blank frame
                height, width = 480, 640
                frame_data = np.zeros((height, width, 3), dtype=np.uint8)
                # Add text indicating waiting for video
                # This would require additional code with OpenCV
            
            # Create a VideoFrame from the numpy array
            pts, time_base = await self.next_timestamp()
            frame = VideoFrame.from_ndarray(frame_data, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            return frame
            
        except Exception as e:
            logger.error(f"Error in GStreamerTrack.recv: {e}")
            # Return a blank frame in case of error
            height, width = 480, 640
            frame_data = np.zeros((height, width, 3), dtype=np.uint8)
            pts, time_base = await self.next_timestamp()
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
    
    # Add a video track to the connection based on the source
    if args.source == 'test_pattern':
        # Use the test pattern video track
        video_track = VideoFileTrack(source='test_pattern')
        logger.info("Using test pattern as video source")
    elif args.source == 'rtsp_pipeline':
        # Use the DL Streamer pipeline with RTSP input
        rtsp_url = args.rtsp_url
        video_track = GStreamerTrack(
            rtsp_url=rtsp_url,
            model=args.model,
            model_proc=args.model_proc
        )
        logger.info(f"Using DL Streamer pipeline with RTSP source: {rtsp_url}")
    elif args.source.startswith('rtsp://'):
        # Direct RTSP URL without DL Streamer
        video_track = GStreamerTrack(rtsp_url=args.source)
        logger.info(f"Using direct RTSP source: {args.source}")
    else:
        # Use a file or camera as source
        video_track = VideoFileTrack(source=args.source)
        logger.info(f"Using file/camera as video source: {args.source}")
    
    pc.addTrack(video_track)
    
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
            # The format of ICE candidates can vary between browser and aiortc
            # Check if we have a complete candidate object or just the candidate string
            if "candidate" in data:
                candidate_str = data["candidate"]
                
                # If empty candidate, it means end of candidates
                if not candidate_str:
                    logger.info("Received end-of-candidates signal")
                    return
                
                # Remove 'candidate:' prefix if present
                if candidate_str and candidate_str.startswith("candidate:"):
                    candidate_str = candidate_str[10:]
                
                # Create the ICE candidate with the correct parameters
                if "sdpMid" in data and "sdpMLineIndex" in data:
                    candidate = RTCIceCandidate(
                        component=None,  # Let aiortc determine this
                        foundation=None,  # Let aiortc determine this
                        ip=None,          # Let aiortc determine this
                        port=None,        # Let aiortc determine this
                        priority=None,    # Let aiortc determine this
                        protocol=None,    # Let aiortc determine this
                        type=None,        # Let aiortc determine this
                        sdpMid=data["sdpMid"],
                        sdpMLineIndex=data["sdpMLineIndex"],
                        candidate=candidate_str
                    )
                    
                    await pc.addIceCandidate(candidate)
                    logger.info("ICE candidate added successfully")
                else:
                    logger.warning("Incomplete ICE candidate data, missing sdpMid or sdpMLineIndex")
            else:
                logger.warning("Received ICE candidate message without candidate field")
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
                    # Use asyncio.create_task to avoid blocking the event loop
                    asyncio.create_task(handle_candidate(data))
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
            try:
                await pc.close()
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")

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
        signaling_server = args.server_url  # Updated to use the command-line argument
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
