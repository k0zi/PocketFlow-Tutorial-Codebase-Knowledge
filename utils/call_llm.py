import os
import logging
import json
from datetime import datetime
import requests

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")

        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    # Call Ollama API
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11435")
    model = os.getenv("OLLAMA_MODEL", "Qwen3")

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            error_msg = f"Ollama API call failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        response_text = response.json().get("response", "")
    except Exception as e:
        error_msg = f"Failed to call Ollama API: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass

        # Add to cache and save
        cache[prompt] = response_text
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")