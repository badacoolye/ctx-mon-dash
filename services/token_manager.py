#!/usr/bin/env python
import requests
import os
from datetime import datetime, timezone
import logging
from urllib.parse import urlencode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_token_directly(client_id: str, client_secret: str, base_url: str = "https://api.cloud.com", timeout: int = 10) -> str:
    """
    Fetch the token directly from the API without saving it to a file.
    
    :param client_id: The client ID for authentication.
    :param client_secret: The client secret for authentication.
    :param base_url: The base URL of the API.
    :param timeout: Request timeout in seconds.
    :return: The access token string.
    """
    token_url = f"{base_url.rstrip('/')}/cctrustoauth2/root/tokens/clients"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    })

    try:
        logger.info(f"Attempting to generate token from: {token_url}")
        response = requests.post(token_url, headers=headers, data=data, timeout=timeout)
        response.raise_for_status()
        token_data = response.json()
        if "access_token" not in token_data:
            logger.error(f"Unexpected response format: {token_data}")
            raise ValueError("Response did not contain access_token")
        token = token_data["access_token"]
        logger.info("Successfully generated token directly without saving to file")
        return token

    except Exception as e:
        logger.error("Error generating token", exc_info=True)
        raise

if __name__ == "__main__":
    # When running this script directly, read client credentials from environment variables
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.error("Client credentials not set in .env file. Please set CLIENT_ID and CLIENT_SECRET.")
    else:
        token = get_token_directly(client_id, client_secret)
        print("Access Token:", f"{token[:6]}...{token[-6:]}")
