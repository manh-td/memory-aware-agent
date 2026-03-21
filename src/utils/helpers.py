"""Common utility functions."""

import os
import logging
from dotenv import load_dotenv, find_dotenv


def load_env():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())


def suppress_warnings():
    """Suppress warnings from HuggingFace and transformers libraries."""
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def get_openai_api_key():
    """Get OpenAI API key from environment."""
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key
