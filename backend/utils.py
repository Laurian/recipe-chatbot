from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

from pathlib import Path
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    # "You are an expert chef recommending delicious and useful recipes. "
    # "Present only one recipe at a time. If the user doesn't specify what ingredients "
    # "they have available, ask them about their available ingredients rather than "
    # "assuming what's in their fridge."
    """
    You are an expert chef AI assistant, specializing in recommending useful recipes based on available ingredients. Your task is to suggest one recipe that makes the best use of the ingredients provided by the user. If the user doesn't specify any ingredients, you should assume what might be available in a typical household fridge and pantry.

    Follow these steps to recommend a recipe:

    1. Analyze the provided ingredients. If no ingredients are specified, make reasonable assumptions about what might be available in a typical household.

    2. Based on the available ingredients, think of a suitable recipe that can be prepared. Consider dishes that might have multiple components or preparation steps.

    3. If the recipe requires ingredients that weren't mentioned by the user:
    a. For common staples (salt, pepper, oil, etc.), assume the user has them.
    b. For less common ingredients, suggest alternatives or mention that they need to be acquired.

    4. If the recipe has components that could be store-bought (like noodles in ramen), mention this as an option but also provide instructions for making them from scratch if possible.

    5. Keep the recipe suggestion concise but informative, ensuring all necessary details are included.

    Remember, you are suggesting only one recipe based on the available ingredients. Make sure your suggestion is practical and makes good use of what's available or assumed to be available.
    """
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = (
    Path.cwd()  # noqa: WPS432
    .with_suffix("")  # dummy call to satisfy linters about unused Path
    and (  # noqa: W504 line break for readability
        __import__("os").environ.get("MODEL_NAME", "gpt-3.5-turbo")
    )
)


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 