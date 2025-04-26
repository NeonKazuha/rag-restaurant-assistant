import requests
import logging

# Configure logging (basic example, might be configured elsewhere in a larger app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"

def fetch_data(url: str, user_agent: str = DEFAULT_USER_AGENT) -> str | None:
    """
    Fetches HTML content from a given URL with appropriate headers and error handling.

    Args:
        url: The URL to fetch data from.
        user_agent: The User-Agent string to use for the request.

    Returns:
        The HTML content as a string if successful, None otherwise.
    """
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Successfully fetched data from {url}")
        return response.text
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while fetching {url}")
        return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching {url}: {http_err} - Status Code: {http_err.response.status_code}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error fetching {url}: {req_err}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while fetching {url}: {e}") # Catch other potential errors
        return None

def handle_general_error(error: Exception, context: str = "General operation"):
    """
    Logs errors with context.

    Args:
        error: The exception object that occurred.
        context: A string describing the context where the error happened.
    """
    logger.error(f"An error occurred during {context}: {error}", exc_info=True)
    # In a real application, you might add more sophisticated handling:
    # - Send notifications (e.g., email, Slack)
    # - Implement specific retry logic based on error type
    # - Graceful shutdown or fallback mechanisms