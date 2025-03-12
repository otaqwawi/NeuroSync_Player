# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import pygame
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from threading import Thread

warnings.filterwarnings(
    "ignore", 
    message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)

from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.files.file_utils import initialize_directories, ensure_wav_input_folder_exists
from utils.neurosync.neurosync_api_connect import send_audio_to_neurosync
from utils.generated_runners import prepare_facial_data_for_animation, run_prepared_animation

app = FastAPI()

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
    ensure_wav_input_folder_exists(os.path.join(os.getcwd(), 'wav_input'))
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

@app.post("/process-audio")
async def process_audio(audio_file: UploadFile = File(...)):
    try:
        if not audio_file:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio file is required"}
            )

        # Read the uploaded file
        audio_bytes = await audio_file.read()

        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file received"}
            )

        # Process the audio with Neurosync API
        facial_data = send_audio_to_neurosync(audio_bytes)
        
        if facial_data is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to generate facial data"}
            )
        
        # Prepare facial data for animation
        encoded_facial_data = prepare_facial_data_for_animation(facial_data)
        if encoded_facial_data is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to prepare facial data for animation"}
            )
            
        # Run the animation in a background thread
        animation_thread = Thread(
            target=run_prepared_animation,
            args=(
                audio_bytes,
                encoded_facial_data,
                py_face,
                socket_connection,
                default_animation_thread
            )
        )
        animation_thread.start()
        
        return JSONResponse(
            content={"message": "Animation started successfully"},
            status_code=200
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6502)
