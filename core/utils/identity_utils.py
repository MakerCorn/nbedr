try:
    from azure.identity import DefaultAzureCredential, CredentialUnavailableError
    credential = DefaultAzureCredential()
    AZURE_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    CredentialUnavailableError = Exception
    credential = None
    AZURE_AVAILABLE = False

import logging
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

tokens = {}

def get_db_token() -> Optional[str]:
    """Retrieves a token for database access using Azure Entra ID.

    Returns:
        Optional[str]: The database access token, or None if unavailable.
    """
    return _get_token("db_token", "https://ossrdbms-aad.database.windows.net/.default")

def get_azure_openai_token() -> Optional[str]:
    """Retrieves a token for Azure OpenAI service using Azure Entra ID.

    Returns:
        Optional[str]: The Azure OpenAI service token, or None if unavailable.
    """
    return get_cognitive_service_token()

def get_cognitive_service_token() -> Optional[str]:
    """Retrieves a token for Azure Cognitive Services using Azure Entra ID.

    Returns:
        Optional[str]: The Azure Cognitive Services token, or None if unavailable.
    """
    return _get_token("cognitive_token", "https://cognitiveservices.azure.com/.default")

def _format_datetime(dt):
    """Formats a datetime object as a string in the local timezone."""
    return datetime.utcfromtimestamp(dt).replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')

def _get_token(token_key, resource) -> Optional[str]:
    """Retrieves and caches an Azure AD token for the specified resource.

    Args:
        token_key (str): The key identifying the token.
        resource (str): The resource URI for which the token is requested.

    Returns:
        Optional[str]: The Azure AD token, or None if unavailable.
    """
    if not AZURE_AVAILABLE or credential is None:
        log.warning("Azure identity libraries not available, returning None for token")
        return None
        
    now = int(time.time())
    global tokens
    token = tokens.get(token_key)
    try:
        if token is None or now > token.expires_on - 60:
            log.debug(f"Requesting new Azure AD token for {resource}...")
            token = credential.get_token(resource)
            tokens[token_key] = token
            log.debug(f"Got new Azure AD token for {resource} (expires: {_format_datetime(token.expires_on)}, now: {_format_datetime(now)})")
        else:
            log.debug(f"Using cached Azure AD token for {resource} (expires: {_format_datetime(token.expires_on)}, now: {_format_datetime(now)})")
        return token.token
    except CredentialUnavailableError as e:
        log.error(f"Azure credential unavailable: {e}")
        return None
    except Exception as e:
        log.error(f"Failed to get Azure AD token for {resource}: {e}")
        return None
