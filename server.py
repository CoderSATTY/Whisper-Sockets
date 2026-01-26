import os
import json
import re
import base64
import asyncio
import websockets
import logging
import numpy as np
import torch
import noisereduce as nr
from dotenv import load_dotenv
from faster_whisper import WhisperModel
#from google.cloud import texttospeech

load_dotenv()

torch.set_num_threads(1)

PORT = int(os.getenv("PORT", 8765))
SAMPLE_RATE = 16000
WINDOW_SIZE = 512  
MIN_VOLUME = 0.02
SILENCE_DURATION = 1.0
INTERMEDIATE_INTERVAL = 0.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("server")

try:
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("Google TTS Client initialized")
except Exception as e:
    logger.error(f"Failed to init Google TTS: {e}")

print("Loading Whisper Large V3...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("large-v3", device=device, compute_type="int8")
    print("Whisper Loaded!")
except Exception as e:
    logger.error(f"Whisper loading Error: {e}")

print("Loading Silero VAD...")
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True
    )
    print("Silero VAD Loaded!")
except Exception as e:
    logger.error(f"VAD loading Error: {e}")

background_tasks = set()


def denoise_audio(audio: np.ndarray) -> np.ndarray:
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)


def is_speech(audio: np.ndarray, threshold: float = 0.5) -> bool:
    """Check if audio contains speech using Silero VAD in 512-sample windows"""
    speech_probs = []
    
    for i in range(0, len(audio), WINDOW_SIZE):
        chunk = audio[i:i + WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            break
        tensor = torch.from_numpy(chunk).float()
        prob = vad_model(tensor, SAMPLE_RATE).item()
        speech_probs.append(prob)
    
    vad_model.reset_states()
    
    if not speech_probs:
        return False
    
    # Return True if any chunk has high speech probability
    return max(speech_probs) > threshold


# TTS HANDLER 
async def handle_tts_request(websocket, text):
    logger.info(f"Converting text to speech: {text}")
    
    sentences = re.findall(r'[^.!?]+[.!?]+|[^.!?]+$', text)
    if not sentences: sentences = [text]

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence: continue

        try:
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Chirp3-HD-Alnilam",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0
            )

            response = await asyncio.to_thread(
                tts_client.synthesize_speech,
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            audio_b64 = base64.b64encode(response.audio_content).decode('utf-8')

            await websocket.send(json.dumps({
                "type": "tts_audio_chunk",
                "audio": audio_b64,
                "message": text,
                "index": i,
                "is_last": i == len(sentences) - 1
            }))

        except Exception as e:
            logger.error(f"TTS Error: {e}")

    await websocket.send(json.dumps({"type": "tts_audio_done"}))


# STT LOGIC
async def transcribe_and_send(websocket, audio_data, is_final=False):
    try:
        segments, _ = await asyncio.to_thread(
            model.transcribe,
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=False
        )
        
        text = " ".join([s.text for s in segments]).strip()
 
        if text and text.lower() not in ["you", "bye.", "thank you.", "mm-hmm."]:
            logger.info(f"{'Final' if is_final else 'Partial'} Transcribed: {text}")
            await websocket.send(text)
            
    except Exception as e:
        logger.error(f"Transcription Error: {e}")


# MAIN HANDLER
async def connection_handler(websocket):
    logger.info(f"Client connected: {websocket.remote_address}")
    
    audio_buffer = []
    is_speaking = False
    last_speech_time = asyncio.get_event_loop().time()
    last_transcribe_time = asyncio.get_event_loop().time()
    is_processing = False 

    def done_callback(task):
        nonlocal is_processing
        is_processing = False
        background_tasks.discard(task)

    try:
        async for message in websocket:
            
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get("type") == "text":
                        await handle_tts_request(websocket, data.get("text"))
                    continue 
                except json.JSONDecodeError:
                    pass 

            if isinstance(message, bytes):
                data_float32 = np.frombuffer(message, dtype=np.float32)
                cleaned_audio = denoise_audio(data_float32)
                volume = np.sqrt(np.mean(cleaned_audio**2))
                current_time = asyncio.get_event_loop().time()

                print(f"Volume: {volume:.4f} ", end="\r", flush=True)
                speech_detected = volume > MIN_VOLUME and is_speech(cleaned_audio)

                if speech_detected:
                    if not is_speaking:
                        is_speaking = True
                        print("Speaking...", end="\r", flush=True)
                    
                    last_speech_time = current_time
                    audio_buffer.append(cleaned_audio)

                    if (current_time - last_transcribe_time) > INTERMEDIATE_INTERVAL and not is_processing:
                        last_transcribe_time = current_time
                        if len(audio_buffer) > 0:
                            is_processing = True
                            full_audio = np.concatenate(audio_buffer)
                            task = asyncio.create_task(transcribe_and_send(websocket, full_audio, is_final=False))
                            background_tasks.add(task)
                            task.add_done_callback(done_callback)

                else:
                    if is_speaking:
                        audio_buffer.append(cleaned_audio)
                        
                        if (current_time - last_speech_time) > SILENCE_DURATION:
                            is_speaking = False
                            
                            if len(audio_buffer) > 0:
                                full_audio = np.concatenate(audio_buffer)
                                audio_buffer = [] 
                                
                                task = asyncio.create_task(transcribe_and_send(websocket, full_audio, is_final=True))
                                background_tasks.add(task)
                                task.add_done_callback(background_tasks.discard)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Connection Error: {e}")


async def main():
    logger.info(f"Server started on 0.0.0.0:{PORT}")
    async with websockets.serve(connection_handler, "0.0.0.0", PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")