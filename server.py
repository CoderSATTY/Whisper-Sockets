import asyncio
import websockets
import logging
import numpy as np
from faster_whisper import WhisperModel

PORT = 8765
MIN_VOLUME = 0.005      # Minimum amplitude to consider "speech"
SILENCE_DURATION = 1.0  # Seconds of silence to trigger transcription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Whisper Large V3...")
try:
    model = WhisperModel("large-v3", device=device, compute_type="float16")
    print("Model Loaded!")
except Exception as e:
    logger.error(f"Model loading Error: {e}")

background_tasks = set()

async def transcribe_and_send(websocket, audio_data):
    """Runs Whisper inference in a separate thread and sends text back."""
    try:

        segments, _ = await asyncio.to_thread(
            model.transcribe,
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True
        )
        
        text = " ".join([s.text for s in segments]).strip()
 
        if text and text.lower() not in ["you", "bye.", "thank you.", "mm-hmm."]:
            logger.info(f"Transcribed: {text}")
            await websocket.send(text)
            
    except Exception as e:
        logger.error(f"Transcription Error: {e}")

async def audio_handler(websocket):
    """Handles the audio stream, VAD logic, and connection lifecycle."""
    logger.info(f"Client connected: {websocket.remote_address}")
    
    audio_buffer = []
    is_speaking = False
    last_speech_time = asyncio.get_event_loop().time()

    try:
        async for message in websocket:
            data_float32 = np.frombuffer(message, dtype=np.float32)

            volume = np.sqrt(np.mean(data_float32**2))
            current_time = asyncio.get_event_loop().time()

            if volume > MIN_VOLUME:
                if not is_speaking:
                    is_speaking = True
                    print("Speaking...", end="\r", flush=True)
                last_speech_time = current_time
                audio_buffer.append(data_float32)
            else:
                if is_speaking:
                    audio_buffer.append(data_float32)
                    
                    # Silence detected -> End of sentence
                    if (current_time - last_speech_time) > SILENCE_DURATION:
                        is_speaking = False
                        
                        if len(audio_buffer) > 0:
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer = [] 
                            
                            # Fire-and-forget task (Parallel Processing)
                            task = asyncio.create_task(transcribe_and_send(websocket, full_audio))
                            background_tasks.add(task)
                            task.add_done_callback(background_tasks.discard)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Connection Error: {e}")

async def main():
    logger.info(f"Server started on 0.0.0.0:{PORT}")
    # Serve on 0.0.0.0 to accept traffic from the ngrok tunnel
    async with websockets.serve(audio_handler, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")