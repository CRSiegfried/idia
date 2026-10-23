"""The chat leaves: label probabilities read from a chat model by a named elicitation.

Each module holds one chat API's leaf with its read protocol. Importing this package
needs the ``openai`` extra.
"""

from idia.providers.chat.openai import OpenAIChat

__all__ = ["OpenAIChat"]
