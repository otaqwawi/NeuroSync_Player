# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pygame
import warnings
warnings.filterwarnings(
    "ignore", 
    message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)
from threading import Thread

from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.audio_face_workers import process_wav_file
from utils.files.file_utils import initialize_directories, save_generated_data, load_facial_data_from_csv, GENERATED_DIR
from utils.neurosync.neurosync_api_connect import send_audio_to_neurosync
from utils.generated_runners import run_audio_animation_from_bytes

# Create FastAPI app
app = FastAPI()

# Define request model
class AudioRequest(BaseModel):
    audio_base64: str

# Global variables for animation system
py_face = None
socket_connection = None
default_animation_thread = None

def initialize_animation_system():
    global py_face, socket_connection, default_animation_thread
    if py_face is None:
        py_face = initialize_py_face()
        socket_connection = create_socket_connection()
        default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
        default_animation_thread.start()

@app.on_event("startup")
async def startup_event():
    initialize_directories()
    initialize_animation_system()

@app.on_event("shutdown")
async def shutdown_event():
    global default_animation_thread
    if default_animation_thread:
        stop_default_animation.set()
        default_animation_thread.join()
    pygame.quit()
    if socket_connection:
        socket_connection.close()

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {"status": "healthy", "service": "NeuroSync Player API"}

@app.post("/process-audio")
async def process_audio(request: AudioRequest):
    try:
        print('Received audio request')
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(request.audio_base64)
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data received")

        # Send the audio bytes to the API and get the blendshapes
        generated_facial_data = send_audio_to_neurosync(audio_bytes)

        if generated_facial_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate facial data")

        # Save the generated data and get unique ID
        unique_id = save_generated_data(audio_bytes, generated_facial_data)

        return {"status": "success", "message": "Generate blendshape completed", "id": unique_id[0]}

    except Exception as e:
        print(f"Error processing audio: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/play-animation/{animation_id}")
async def play_animation(animation_id: str):
    try:
        print(f"Received request to play animation with ID: {animation_id}")
        # Get the file paths for the animation
        audio_path = os.path.join(GENERATED_DIR, animation_id, 'audio.wav')
        shapes_path = os.path.join(GENERATED_DIR, animation_id, 'shapes.csv')

        if not os.path.exists(audio_path) or not os.path.exists(shapes_path):
            raise HTTPException(status_code=404, detail="Animation data not found")

        # Load the facial data
        generated_facial_data = load_facial_data_from_csv(shapes_path)

        # Read the audio file
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        # Run the animation directly without creating a background thread
        run_audio_animation_from_bytes(
            audio_bytes,
            generated_facial_data,
            py_face,
            socket_connection,
            default_animation_thread
        )

        # Delete generated files after playback
        try:
            import shutil
            animation_dir = os.path.join(GENERATED_DIR, animation_id)
            if os.path.exists(animation_dir):
                shutil.rmtree(animation_dir)
        except Exception as e:
            print(f"Warning: Failed to delete animation files: {e}")

        return {"status": "success", "message": "Animation playback completed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6502)
