import logging
import os
import tempfile
from typing import Optional

import uvicorn
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local Whisper API",
    description="Local speech-to-text API using OpenAI Whisper",
)

# Add CORS middleware to allow requests from Obsidian
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Obsidian app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the loaded model and its name
whisper_model = None
current_model_name = None


def load_whisper_model(model_name: str = "turbo"):
    """Load the Whisper model into memory"""
    global whisper_model, current_model_name
    try:
        logger.info(f"Loading Whisper model: {model_name}")
        whisper_model = whisper.load_model(model_name)
        current_model_name = model_name
        logger.info(f"Whisper model {model_name} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load the Whisper model when the server starts"""
    load_whisper_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Local Whisper API is running",
        "model_loaded": whisper_model is not None,
        "current_model": current_model_name or "none",
    }


@app.get("/models")
async def get_available_models():
    """Get list of available Whisper models"""
    models = ["tiny", "base", "small", "medium", "large"]
    return {"available_models": models, "current_model": current_model_name or "none"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form("turbo"),
    language: Optional[str] = Form("en"),
    prompt: Optional[str] = Form(""),
):
    """
    Transcribe audio file using local Whisper model

    Args:
        file: Audio file to transcribe
        model: Whisper model to use (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'es', 'fr')
        prompt: Optional prompt to guide transcription

    Returns:
        JSON with transcribed text
    """
    logger.info(f"Transcribing audio file: {file.filename}")
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Load model if different from current one or if no model is loaded
        global whisper_model, current_model_name
        if whisper_model is None or current_model_name != model:
            logger.info(f"Model change detected: {current_model_name} -> {model}")
            if not load_whisper_model(model):
                raise HTTPException(
                    status_code=500, detail="Failed to load Whisper model"
                )
        else:
            logger.info(f"Using already loaded model: {current_model_name}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file.filename.split('.')[-1]}"
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Transcribe the audio
            logger.info(f"Transcribing audio file: {file.filename}")

            # Prepare transcription options
            options = {}
            if language and language != "auto":
                options["language"] = language
            if prompt:
                options["initial_prompt"] = prompt

            # Perform transcription
            result = whisper_model.transcribe(temp_file_path, **options)

            # Clean up temporary file
            os.unlink(temp_file_path)

            logger.info("Transcription completed successfully")

            # Ensure first word starts with capital letter
            transcribed_text = result["text"].strip()
            if transcribed_text:
                transcribed_text = transcribed_text[0].upper() + transcribed_text[1:]
            
            return {
                "text": transcribed_text,
                "language": result.get("language", language),
                "segments": result.get("segments", []),
            }

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/v1/audio/transcriptions")
async def openai_compatible_endpoint(
    file: UploadFile = File(...),
    model: Optional[str] = Form("turbo"),
    language: Optional[str] = Form("en"),
    prompt: Optional[str] = Form(""),
):
    """
    OpenAI-compatible endpoint for the Obsidian plugin

    This endpoint mimics the OpenAI API response format to maintain compatibility
    """
    try:
        # Call the main transcribe function
        result = await transcribe_audio(file, model, language, prompt)

        # Return in OpenAI-compatible format
        return {"text": result["text"], "language": result["language"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI-compatible endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    # Load model on startup
    load_whisper_model()

    # Run the server
    uvicorn.run(
        "whisper_api_server:app",
        host="0.0.0.0",
        port=6431,
        reload=True,
        log_level="info",
    )
