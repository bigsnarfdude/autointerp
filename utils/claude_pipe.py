"""Wrapper for claude -p (pipe mode) calls."""
import subprocess
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def claude_pipe(
    prompt: str,
    model: str = "claude-opus-4-5-20251101",
    timeout: int = 120,
    max_retries: int = 3,
) -> str:
    """Send prompt to claude -p and return response.

    Args:
        prompt: The prompt text to send.
        model: Model name for --model flag.
        timeout: Seconds before timeout.
        max_retries: Number of retries on failure.

    Returns:
        Response text from claude.
    """
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(
                    f"claude -p returned {result.returncode}: {result.stderr[:200]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"claude -p timed out (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"claude -p error: {e} (attempt {attempt + 1}/{max_retries})")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # exponential backoff

    raise RuntimeError(f"claude -p failed after {max_retries} attempts")


def claude_pipe_json(
    prompt: str,
    model: str = "claude-opus-4-5-20251101",
    timeout: int = 120,
    max_retries: int = 3,
) -> dict:
    """Send prompt and parse JSON response."""
    response = claude_pipe(prompt, model=model, timeout=timeout, max_retries=max_retries)
    # Try to extract JSON from response (may have markdown fences)
    text = response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())
