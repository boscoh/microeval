"""
Simple chat client abstraction for LLM providers.

- Async only, with proper async context management
- No streaming, only conversation with tools and embeddings
- Mostly OpenAI JSON structure, but without choices, and easier token usage metadata
- No langchain, litellm etc., just vendor-provided Python packages
"""

import copy
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import aioboto3
import boto3
import groq
import ollama
import openai
from botocore.exceptions import ClientError, ProfileNotFound

logger = logging.getLogger(__name__)

LLMService = Literal["openai", "ollama", "bedrock", "groq"]


def load_config() -> Dict[str, Any]:
    """
    Load and return the models configuration from models.json.

    Returns:
        Dict[str, Any]: Configuration dictionary with chat_models and embed_models
    """
    config_path = Path(__file__).parent / "models.json"
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using fallback config")
        # Fallback config if file doesn't exist
        return {
            "chat_models": {
                "bedrock": ["amazon.nova-pro-v1:0"],
                "openai": ["gpt-4o"],
                "ollama": ["llama3.2"],
                "groq": ["llama-3.3-70b-versatile"],
            },
            "embed_models": {
                "openai": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                "ollama": ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
                "bedrock": ["amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"],
            },
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing models.json: {e}")
        raise


def get_llm_client(client_type: LLMService, **kwargs) -> "SimpleLLMClient":
    """
    Gets a chat client that satisfies SimpleLLMClient interface.

    Args:
        client_type: "openai", "ollama", "bedrock", or "groq"
        **kwargs: Additional keyword arguments specific to the chat client type:
            - model: str (optional, defaults from models.json if not provided)
    """
    client_type = client_type.lower()

    # Use config default model if not provided
    if "model" not in kwargs:
        config = load_config()
        default_models = config.get("chat_models", {}).get(client_type, [])
        if default_models:
            # Take the first model from the list as the default
            kwargs["model"] = default_models[0] if isinstance(default_models, list) else default_models

    if client_type == "openai":
        return OpenAIClient(**kwargs)
    if client_type == "ollama":
        return OllamaClient(**kwargs)
    if client_type == "bedrock":
        return BedrockClient(**kwargs)
    if client_type == "groq":
        return GroqClient(**kwargs)
    raise ValueError(f"Unknown chat client type: {client_type}")


class SimpleLLMClient(ABC):
    """Abstract base class for SimpleLLMClient, an API for LLM with async interface"""

    @abstractmethod
    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the model.

        Note:
            - All implementations should handle errors gracefully by returning
              the standardized error format rather than raising exceptions
            - Tool/function calling support varies by provider
            - Token counting methods may vary between providers

        Args:
            messages: List of message dictionaries representing the conversation history.
                Each message dict must contain:
                - 'role': str - One of 'system', 'user', 'assistant', or 'tool'
                - 'content': str | None - The message content/text (None allowed for assistant with tool_calls)
                - 'name': str (optional) - Name of the message sender

                Message types:

                1. System messages (role='system'):
                   - 'role': 'system' (required)
                   - 'content': str (required) - System instructions or context
                   - 'name': str (optional) - Name identifier

                2. User messages (role='user'):
                   - 'role': 'user' (required)
                   - 'content': str (required) - User's message text
                   - 'name': str (optional) - Name identifier

                3. Assistant messages (role='assistant'):
                   - 'role': 'assistant' (required)
                   - 'content': str | None (optional) - Assistant's response text
                     Can be None if only tool_calls are present
                   - 'tool_calls': list[dict] (optional) - List of tool calls when assistant wants to use tools
                     Each tool call dict contains:
                     - 'id': str - Unique identifier for this tool call
                     - 'type': str - Usually 'function'
                     - 'function': dict - Function call details:
                       - 'name': str - Name of the function to call
                       - 'arguments': str - JSON string of function arguments
                   - 'name': str (optional) - Name identifier

                4. Tool messages (role='tool'):
                   - 'role': 'tool' (required)
                   - 'content': str (required) - Result of the tool execution
                   - 'tool_call_id': str (required) - ID of the tool call this result corresponds to
                   - 'status': str (optional) - Status of tool execution: 'success' or 'error'
                   - 'name': str (optional) - Name identifier

                Example:
                [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'What is the weather in Paris?'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'call_123',
                                'type': 'function',
                                'function': {
                                    'name': 'get_weather',
                                    'arguments': '{"location": "Paris"}'
                                }
                            }
                        ]
                    },
                    {
                        'role': 'tool',
                        'content': 'Sunny, 22°C',
                        'tool_call_id': 'call_123',
                        'status': 'success'
                    },
                    {'role': 'assistant', 'content': 'The weather in Paris is sunny and 22°C.'}
                ]

            tools: Optional list of tool/function definitions for function calling.
                Each tool dict must contain:
                - 'type': str - Tool type (typically 'function')
                - 'function': dict - Function specification sub-dictionary containing:
                  - 'name': str (required) - Function name, must be unique
                  - 'description': str (required) - Function description explaining what it does
                  - 'parameters': dict (required) - JSON Schema object defining the function parameters
                    The parameters dict must follow JSON Schema format with:
                    - 'type': str - Usually 'object'
                    - 'properties': dict - Dictionary of parameter definitions
                      Each property should have 'type' (e.g., 'string', 'number', 'boolean')
                    - 'required': list[str] (optional) - List of required parameter names

                Example:
                [{
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'description': 'Get current weather for a location',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'location': {
                                    'type': 'string',
                                    'description': 'City name or location'
                                },
                                'unit': {
                                    'type': 'string',
                                    'enum': ['celsius', 'fahrenheit'],
                                    'description': 'Temperature unit'
                                }
                            },
                            'required': ['location']
                        }
                    }
                }]

            max_tokens: Maximum number of tokens to generate in the completion.
                If None, uses the model's default limit. Different models have
                different default and maximum token limits.

            temperature: Controls randomness in generation (0.0 to 1.0).
                - 0.0: Deterministic, always picks most likely token
                - 1.0: Maximum randomness
                - Values between 0.1-0.7 are typically good for most use cases

        Returns:
            Dict[str, Any]: Standardized response dictionary containing:
            {
                'text': str,
                'metadata': {
                    'usage': {
                        'prompt_tokens': int,
                        'completion_tokens': int,
                        'total_tokens': int,
                        'elapsed_seconds': float
                    },
                    'model': str,
                    'finish_reason': str
                },
                'tool_calls': [
                    {
                        'function': {
                            'name': str,
                            'arguments': str,
                            'tool_call_id': str
                        }
                    }
                ] | None
            }

            On error, returns:
            {
                'text': 'Error: <error_message>',
                'metadata': {
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'elapsed_seconds': float
                    },
                    'model': str,
                    'error': str
                }
            }

        """
        pass

    @abstractmethod
    async def embed(self, input: str) -> List[float]:
        """Generate a text embedding vector for the given input string."""
        pass

    @abstractmethod
    def get_token_cost(self) -> float:
        """Get the cost per 1K tokens for the model in AUD."""
        pass

    async def connect(self):
        """Initialize async resources. Override in subclasses as needed."""
        pass

    async def close(self):
        """Clean up async resources. Override in subclasses as needed."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def parse_response_as_json_list(response):
    """Parse JSON from text response, extracting from markdown or .transactions if needed."""
    import re

    if isinstance(response, dict):
        response_text = response.get("text", "")
    elif isinstance(response, str):
        response_text = response
    else:
        return None

    if not response_text:
        return None

    def try_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    parsed = try_parse(response_text)
    if parsed:
        return parsed

    patterns = [
        r"```(?:json|python)?\s*([\s\S]*?)\s*```",
        r"```(?:json)?\s*({[\s\S]*})\s*```",
        r"\{[\s\S]*\}",
        r"({[\s\S]*})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            parsed = try_parse(match)
            if parsed:
                return parsed

    return None


class OllamaClient(SimpleLLMClient):
    def __init__(self, model: str = None):
        """Initialize Ollama chat client.

        Args:
            model: Name of the Ollama model to use (default from config)

        Raises:
            RuntimeError: If Ollama is not running or the model is not available
        """
        self.model = model
        self.client = None

    async def connect(self):
        if self.client:
            return

        logger.info(f"Initializing 'ollama:{self.model}'")
        self.client = ollama.AsyncClient()
        try:
            await self.client.list()
        except Exception as e:
            raise RuntimeError(
                "Ollama is not running or not installed. "
                "Please start the Ollama service and try again."
            ) from e

        try:
            ollama.show(self.model)
        except Exception as e:
            raise RuntimeError(
                f"Model '{self.model}' is not available. "
                f"Please ensure the model is pulled and available. Error: {str(e)}"
            )

    def _transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform standardized message format to Ollama's expected format.

        Ollama expects tool_calls arguments as dicts, not JSON strings.
        """
        transformed = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                new_msg = dict(msg)
                new_msg["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    new_tc = dict(tc)
                    if "function" in new_tc:
                        new_func = dict(new_tc["function"])
                        args = new_func.get("arguments", {})

                        # Ensure arguments is always a dict for Ollama
                        if isinstance(args, str):
                            if args:  # Only parse non-empty strings
                                try:
                                    parsed_args = json.loads(args)
                                    new_func["arguments"] = parsed_args
                                    logger.debug(f"Converted string args to dict: {args[:50]}...")
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse tool arguments: {args[:50]}... Error: {e}")
                                    new_func["arguments"] = {}
                            else:
                                new_func["arguments"] = {}
                        elif not isinstance(args, dict):
                            # Handle any other type by converting to dict
                            logger.warning(f"Unexpected arguments type {type(args)}, converting to empty dict")
                            new_func["arguments"] = {}
                        # If already a dict, keep it as is

                        new_tc["function"] = new_func
                    new_msg["tool_calls"].append(new_tc)
                transformed.append(new_msg)
            else:
                transformed.append(msg)
        return transformed

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Ollama implementation of get_completion with tool support."""
        await self.connect()

        start_time = time.time()

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            transformed_messages = self._transform_messages(messages)

            # Validate that all tool_calls have dict arguments before sending to Ollama
            for msg in transformed_messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        if "function" in tc:
                            args = tc["function"].get("arguments")
                            if isinstance(args, str):
                                logger.error(
                                    f"Tool call still has string arguments after transformation: "
                                    f"tool={tc['function'].get('name')}, args={args[:100]}"
                                )
                                # Force convert to dict as fallback
                                try:
                                    tc["function"]["arguments"] = json.loads(args) if args else {}
                                except json.JSONDecodeError:
                                    tc["function"]["arguments"] = {}

            chat_kwargs = {
                "model": self.model,
                "messages": transformed_messages,
                "options": options,
            }
            if tools is not None:
                chat_kwargs["tools"] = tools

            response = await self.client.chat(**chat_kwargs)
            elapsed_seconds = time.time() - start_time

            if isinstance(response, dict):
                message_dict = response.get("message", {})
                done_reason = response.get("done_reason", "stop")
            else:
                message_obj = getattr(response, "message", None)
                done_reason = getattr(response, "done_reason", "stop")
                
                if message_obj is not None and hasattr(message_obj, "model_dump"):
                    message_dict = message_obj.model_dump()
                elif message_obj is not None:
                    message_dict = {
                        "content": getattr(message_obj, "content", None) or "",
                        "tool_calls": getattr(message_obj, "tool_calls", None),
                    }
                else:
                    message_dict = {}
            
            response_text = (message_dict.get("content") or "") if isinstance(message_dict, dict) else ""
            raw_tool_calls = message_dict.get("tool_calls") if isinstance(message_dict, dict) else None
            
            completion_tokens = len(response_text.split()) if response_text else 0
            prompt_tokens = sum(len((m.get("content") or "").split()) for m in messages)
            total_tokens = prompt_tokens + completion_tokens

            tool_calls = None
            if raw_tool_calls:
                tool_calls = []
                import uuid
                for idx, tool_call in enumerate(raw_tool_calls):
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        func = tool_call["function"]
                        function_name = func.get("name", "")
                        function_args = func.get("arguments", {})
                        tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        
                        if isinstance(function_args, dict):
                            function_args = json.dumps(function_args)
                        elif not isinstance(function_args, str):
                            function_args = str(function_args)
                        
                        tool_calls.append(
                            {
                                "function": {
                                    "name": function_name,
                                    "arguments": function_args,
                                    "tool_call_id": tool_call_id,
                                }
                            }
                        )
                    elif hasattr(tool_call, "function"):
                        function = tool_call.function
                        function_name = getattr(function, "name", "")
                        function_args = getattr(function, "arguments", {})
                        tool_call_id = getattr(tool_call, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                        
                        if isinstance(function_args, dict):
                            function_args = json.dumps(function_args)
                        elif not isinstance(function_args, str):
                            function_args = str(function_args)
                        
                        tool_calls.append(
                            {
                                "function": {
                                    "name": function_name,
                                    "arguments": function_args,
                                    "tool_call_id": tool_call_id,
                                }
                            }
                        )

            result = {
                "text": response_text,
                "metadata": {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "elapsed_seconds": elapsed_seconds,
                    },
                    "model": self.model,
                    "finish_reason": done_reason,
                },
            }

            if tool_calls:
                result["tool_calls"] = tool_calls

            return result
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "elapsed_seconds": time.time() - start_time,
                    }
                },
            }

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using Ollama's embedding capabilities."""
        await self.connect()

        try:
            response = await self.client.embeddings(model=self.model, prompt=input)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error calling Ollama embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """Returns 0.0 AUD since Ollama runs locally with no API costs."""
        return 0.0


class OpenAIClient(SimpleLLMClient):
    def __init__(
        self,
        model: str = None,
    ):
        """Initialize OpenAI chat client.

        Args:
            model: Name of the OpenAI model to use (default from config)

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
            RuntimeError: If the API key is invalid or the model is not available
        """
        self.model = model
        self.client = None
        self._closed = True

    async def connect(self):
        if self.client and not self._closed:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )

        logger.info(f"Initializing 'openai:{self.model}'")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._closed = False

        try:
            await self.client.models.retrieve(self.model)
        except openai.AuthenticationError as e:
            raise RuntimeError(
                "Invalid OpenAI API key. Please check your API key and try again."
            ) from e
        except openai.NotFoundError as e:
            raise RuntimeError(
                f"Model '{self.model}' not found or you don't have access to it. "
                f"Please check the model name and your API permissions."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {str(e)}") from e

    async def close(self):
        """Close the OpenAI client and release resources."""
        if self.client is not None and not self._closed:
            await self.client.close()
            self.client = None
            self._closed = True

    def _transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform intermediate message format to OpenAI API format. Ensures tool messages have correct structure."""
        formatted_messages = []

        for msg in messages:
            role = msg["role"]

            if role == "tool":
                formatted_messages.append(
                    {
                        "role": "tool",
                        "content": msg.get("content", ""),
                        "tool_call_id": msg.get("tool_call_id", ""),
                    }
                )
            else:
                formatted_messages.append(msg)

        return formatted_messages

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """OpenAI implementation of get_completion with full tool support."""
        await self.connect()

        start_time = time.time()

        try:
            formatted_messages = self._transform_messages(messages)
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed_seconds = time.time() - start_time

            text = completion.choices[0].message.content if completion.choices else ""

            if hasattr(completion, "usage") and completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                    "elapsed_seconds": elapsed_seconds,
                }
            else:
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "elapsed_seconds": elapsed_seconds,
                }

            # Extract tool calls if present
            tool_calls = None
            if completion.choices and completion.choices[0].message.tool_calls:
                tool_calls = []
                for tool_call in completion.choices[0].message.tool_calls:
                    tool_calls.append(
                        {
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                                "tool_call_id": tool_call.id,
                            }
                        }
                    )

            return {
                "text": text,
                "metadata": {
                    "usage": usage,
                    "model": self.model,
                    "finish_reason": completion.choices[0].finish_reason
                    if completion.choices and completion.choices[0].finish_reason
                    else "stop",
                },
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "elapsed_seconds": time.time() - start_time,
                    },
                    "model": self.model,
                    "error": str(e),
                },
            }

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using OpenAI's embedding model."""
        await self.connect()

        try:
            response = await self.client.embeddings.create(
                model=self.model, input=input
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error calling OpenAI embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """Returns OpenAI model pricing per 1K tokens in AUD.

        USD prices converted to AUD using 1.52 exchange rate (Dec 2024).
        Blended rates assume 50% input / 50% output token mix:
        - gpt-4: $0.045 USD → $0.0684 AUD per 1K tokens (blended: $0.03 in + $0.06 out)
        - gpt-4o: $0.00625 USD → $0.0095 AUD per 1K tokens (blended: $0.0025 in + $0.01 out)
        - gpt-4o-mini: $0.000375 USD → $0.00057 AUD per 1K tokens (blended: $0.00015 in + $0.0006 out)
        - gpt-4-turbo: $0.02 USD → $0.0304 AUD per 1K tokens (blended: $0.01 in + $0.03 out)
        - gpt-3.5-turbo: $0.001 USD → $0.00152 AUD per 1K tokens (blended: $0.0005 in + $0.0015 out)
        """
        pricing = {
            "gpt-4": 0.0684,
            "gpt-4o": 0.0095,
            "gpt-4o-mini": 0.00057,
            "gpt-4-turbo": 0.0304,
            "gpt-3.5-turbo": 0.00152,
        }
        model_key = self.model.lower()
        if model_key not in pricing:
            logger.warning(
                f"Unknown OpenAI model '{self.model}', using default cost of 0.0 AUD"
            )
        return pricing.get(model_key, 0.0)


class GroqClient(OpenAIClient):
    """Groq chat client that inherits from OpenAI client (Groq uses OpenAI-compatible API)."""

    async def connect(self):
        if self.client and not self._closed:
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set GROQ_API_KEY in your .env file or environment variables."
            )

        logger.info(f"Initializing 'groq:{self.model}'")
        self.client = groq.AsyncGroq(api_key=api_key)
        self._closed = False

    async def embed(self, input: str) -> List[float]:
        """Groq does not currently support embeddings."""
        raise NotImplementedError(
            "Groq does not currently support text embeddings. "
            "Please use OpenAI or another provider for embedding generation."
        )

    def get_token_cost(self) -> float:
        """Returns Groq model pricing per 1K tokens in AUD.

        USD prices converted to AUD using 1.52 exchange rate (Dec 2024).
        Blended rates assume 50% input / 50% output token mix:
        - llama-3.3-70b: $0.00069 USD → $0.001049 AUD per 1K tokens (blended: $0.00059 in + $0.00079 out)
        - llama-3.1-70b: $0.00069 USD → $0.001049 AUD per 1K tokens (blended: $0.00059 in + $0.00079 out)
        - llama-3.1-8b: $0.000065 USD → $0.0001 AUD per 1K tokens (blended: $0.00005 in + $0.00008 out)
        - mixtral-8x7b: $0.00024 USD → $0.00036 AUD per 1K tokens (estimated blended)
        - gemma2-9b: $0.0002 USD → $0.0003 AUD per 1K tokens (estimated blended)
        """
        pricing = {
            "llama-3.3-70b-versatile": 0.001049,
            "llama-3.1-70b-versatile": 0.001049,
            "llama-3.1-8b-instant": 0.0001,
            "llama3-70b-8192": 0.001049,
            "llama3-8b-8192": 0.0001,
            "mixtral-8x7b-32768": 0.00036,
            "gemma2-9b-it": 0.0003,
        }
        model_key = self.model.lower()
        if model_key not in pricing:
            logger.warning(
                f"Unknown Groq model '{self.model}', using default cost of 0.0 AUD"
            )
        return pricing.get(model_key, 0.0)


@lru_cache(maxsize=None)
def get_aws_config(is_raise_exception: bool = True):
    """
    Returns AWS configuration for boto3 client initialization.

    This function searches for AWS profiles and saved credentials to build a
    configuration dictionary that can be used to initialize boto3 clients and
    sessions. It validates the discovered credentials to ensure they are properly
    configured and not expired.

    Credential Discovery Process:
    1. Looks for AWS_PROFILE environment variable to determine profile name
    2. Searches for saved credentials in ~/.aws/credentials and ~/.aws/config
    3. Validates the profile exists if AWS_PROFILE is set
    4. Falls back to default credential chain if profile not found
    5. Creates a boto3 session using the discovered profile (or default)
    6. Validates credentials contain required access_key and secret_key
    7. Tests credential validity with an STS GetCallerIdentity call
    8. Checks for SSO token expiration on SSO profiles

    Environment Variables:
        AWS_PROFILE (str, optional): Name of the AWS profile to use from
                                   ~/.aws/credentials. If not set, uses the
                                   default profile.
        AWS_REGION (str, optional): AWS region to use for AWS services

    Returns:
        dict: AWS configuration dictionary for boto3 client initialization:
            - profile_name (str, optional): The AWS profile name to pass to
                                          boto3.client() or boto3.Session()
            - region_name (str): AWS region name for service clients

    Note:
        This function is cached to avoid repeated credential discovery and
        validation. The returned configuration can be unpacked directly into
        boto3 client constructors. All validation errors are logged but do not
        raise exceptions - returns gracefully with empty config on failure.

    Examples:
        >>> aws_config = get_aws_config()
        >>> s3_client = boto3.client('s3', **aws_config)
        >>>
        >>> # Or with session
        >>> session = boto3.Session(**aws_config)
        >>> dynamodb = session.client('dynamodb')
    """
    aws_config = {}
    available_profiles = set()
    credentials_path = os.path.expanduser("~/.aws/credentials")
    config_path = os.path.expanduser("~/.aws/config")

    # Discover available profiles from credentials file
    if os.path.exists(credentials_path):
        import configparser
        config = configparser.ConfigParser()
        config.read(credentials_path)
        available_profiles.update(config.sections())

    # Discover available profiles from config file
    if os.path.exists(config_path):
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        for section in config.sections():
            if section.startswith("profile "):
                available_profiles.add(section[8:])
            elif section != "default":
                available_profiles.add(section)

    # Validate AWS_PROFILE exists if specified
    profile_name = os.getenv("AWS_PROFILE")
    profile_not_found = False
    if profile_name:
        if profile_name in available_profiles:
            aws_config["profile_name"] = profile_name
        else:
            logger.info(f"AWS profile '{profile_name}' not found, using default credential chain")
            profile_not_found = True

    region = os.getenv("AWS_REGION")
    if region:
        aws_config["region_name"] = region

    # Remove AWS_PROFILE from env if profile not found to allow fallback
    if profile_not_found:
        os.environ.pop("AWS_PROFILE", None)

    try:
        session = boto3.Session(**aws_config)
        credentials = session.get_credentials()

        if not credentials:
            if is_raise_exception:
                if available_profiles:
                    raise ValueError(
                        f"No AWS credentials found.\n"
                        f"Available profiles: {', '.join(available_profiles)}\n"
                        f"To configure: aws configure\n"
                        f"Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
                else:
                    raise ValueError(
                        f"No AWS credentials found.\n"
                        f"To configure: aws configure\n"
                        f"Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
            return aws_config

        if not credentials.access_key or not credentials.secret_key:
            if is_raise_exception:
                raise ValueError("Incomplete AWS credentials (missing access key or secret key)")
            logger.warning("Incomplete AWS credentials")
            return aws_config

        # Validate credentials work
        sts = session.client("sts")
        sts.get_caller_identity()

        # Check for SSO expiry with helpful error message
        if profile_name and profile_name in aws_config.get("profile_name", ""):
            if os.path.exists(config_path):
                import configparser
                config = configparser.ConfigParser()
                config.read(config_path)
                section = f"profile {profile_name}"
                if config.has_section(section) and config.has_option(section, "sso_start_url"):
                    if hasattr(credentials, "token"):
                        creds = credentials.get_frozen_credentials()
                        if hasattr(creds, "expiry_time") and creds.expiry_time < datetime.now(timezone.utc):
                            login_cmd = f"aws sso login --profile {profile_name}"
                            if is_raise_exception:
                                raise ValueError(f"AWS SSO session expired. Please run:\n  {login_cmd}")
                            logger.warning(f"AWS SSO session expired. Please run: {login_cmd}")
                            return aws_config

        return aws_config

    except ClientError as e:
        if is_raise_exception:
            raise
        error_code = e.response["Error"]["Code"]

        if error_code == "ExpiredToken":
            # Check if SSO to provide better error message
            profile_to_check = aws_config.get("profile_name", profile_name)
            if profile_to_check and os.path.exists(config_path):
                import configparser
                config = configparser.ConfigParser()
                config.read(config_path)
                section = f"profile {profile_to_check}"
                if config.has_section(section) and config.has_option(section, "sso_start_url"):
                    login_cmd = f"aws sso login --profile {profile_to_check}"
                    logger.warning(f"AWS SSO session expired. Please run: {login_cmd}")
                    return aws_config
            logger.warning("AWS credentials have expired")
        elif error_code == "InvalidClientTokenId":
            logger.warning("AWS credentials are invalid. Please reconfigure: aws configure")
        else:
            logger.warning(f"AWS API error: {error_code}")
    except Exception as e:
        if is_raise_exception:
            raise
        logger.error(f"AWS credential check failed: {str(e)}")

    return aws_config


class BedrockClient(SimpleLLMClient):
    def __init__(
        self,
        model: str = None,
    ):
        """
        Initialize Bedrock chat client.

        This implementation exclusively uses the Bedrock Converse API to enable
        tool/function calling support. As a result, only Claude models are supported
        since they are the primary models that work well with the Converse API for
        tool usage. Other Bedrock models may not support tools through this API.

        Args:
            model: Claude model ID for Bedrock (default from config).
        """
        self.model = model
        self.client = None
        self._session = None
        self._closed = True

    async def connect(self):
        """Initialize the async client session and client."""
        if self.client is not None and not self._closed:
            return

        logger.info(f"Initializing 'bedrock:{self.model}'")
        aws_config = get_aws_config()
        self._session = aioboto3.Session(**aws_config)
        self.client = await self._session.client("bedrock-runtime").__aenter__()
        self._closed = False

    async def close(self):
        """Close the client and release resources."""
        if self.client is not None and not self._closed:
            await self.client.__aexit__(None, None, None)
            self.client = None
            self._closed = True

    def _build_result_from_response(
        self, response: Any, start_time: float
    ) -> Dict[str, Any]:
        """Build standardized result structure from Bedrock response."""
        text_parts = []
        tool_calls = []

        if isinstance(response, str):
            text_parts.append(response)
            usage = {}
            stop_reason = "stop"
        else:
            output = response.get("output", {})
            if isinstance(output, dict) and "message" in output:
                message = output["message"]
                for content in message.get("content", []):
                    if "text" in content:
                        text_parts.append(content["text"])
                    elif "toolUse" in content:
                        tool_use = content["toolUse"]
                        tool_calls.append(
                            {
                                "function": {
                                    "name": tool_use["name"],
                                    "arguments": json.dumps(tool_use.get("input", {})),
                                    "tool_call_id": tool_use.get("toolUseId", ""),
                                }
                            }
                        )
            usage = response.get("usage", {})
            stop_reason = response.get("stopReason", "unknown")

        result = {
            "text": "\n".join(text_parts).strip(),
            "metadata": {
                "usage": {
                    "prompt_tokens": usage.get("inputTokens", 0),
                    "completion_tokens": usage.get("outputTokens", 0),
                    "total_tokens": usage.get("inputTokens", 0)
                    + usage.get("outputTokens", 0),
                    "elapsed_seconds": time.time() - start_time,
                },
                "model": self.model,
                "finish_reason": stop_reason,
            },
        }

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    def _transform_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Transform intermediate message format to Bedrock Converse API format.
        Returns partially filled request_kwargs with messages, system, and toolConfig.
        """
        system_parts = []
        formatted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
            elif role == "assistant" and "tool_calls" in msg:
                assistant_content = []
                if content:
                    assistant_content.append({"text": content})
                for tool_call in msg.get("tool_calls", []):
                    tool_call_id = tool_call.get("id", "")
                    if tool_call_id:
                        assistant_content.append(
                            {
                                "toolUse": {
                                    "toolUseId": tool_call_id,
                                    "name": tool_call["function"]["name"],
                                    "input": json.loads(
                                        tool_call["function"]["arguments"]
                                    )
                                    if isinstance(
                                        tool_call["function"]["arguments"], str
                                    )
                                    else tool_call["function"]["arguments"],
                                }
                            }
                        )
                if assistant_content:
                    formatted_messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_content,
                        }
                    )
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                tool_content = (
                    content.rstrip() if isinstance(content, str) else str(content)
                )
                tool_status = msg.get("status", "success")
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": tool_call_id,
                                    "content": [{"text": tool_content}],
                                    "status": tool_status,
                                }
                            }
                        ],
                    }
                )
            elif (
                role == "user"
                and isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and "toolResult" in content[0]
            ):
                formatted_messages.append(msg)
            elif role == "assistant" and isinstance(content, list):
                formatted_messages.append(msg)
            else:
                role = "user" if role == "user" else "assistant"
                content = content.rstrip() if isinstance(content, str) else content
                formatted_messages.append(
                    {"role": role, "content": [{"text": content}]}
                )

        formatted_tools = None
        if tools:
            formatted_tools = [
                {
                    "toolSpec": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "inputSchema": {"json": tool["function"].get("parameters", {})},
                    }
                }
                for tool in tools
            ]

        system_blocks = [{"text": "\n\n".join(system_parts)}] if system_parts else []

        request_kwargs = {
            "messages": formatted_messages,
            "system": system_blocks,
        }

        if tools:
            request_kwargs["toolConfig"] = {"tools": formatted_tools}

        return request_kwargs

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Bedrock implementation using Converse API with tool support."""
        await self.connect()
        start_time = time.time()

        try:
            request_kwargs = self._transform_messages(messages, tools)

            request_kwargs.update(
                {
                    "modelId": self.model,
                    "inferenceConfig": {
                        "temperature": temperature,
                        "maxTokens": max_tokens or 1024,
                    },
                }
            )

            try:
                response = await self.client.converse(**request_kwargs)
                return self._build_result_from_response(response, start_time)
            except Exception as e:
                logger.error(f"Error in Converse API call: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            return {
                "text": f"Error: {str(e)}",
                "metadata": {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "elapsed_seconds": time.time() - start_time,
                    },
                    "model": self.model,
                    "error": str(e),
                },
            }

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using Bedrock's embedding model."""
        try:
            await self.connect()

            response = await self.client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": input}),
            )

            raw_body = await response["body"].read()
            body = json.loads(raw_body.decode("utf-8"))
            return body["embedding"]

        except Exception as e:
            logger.error(f"Error calling Bedrock embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """Returns Bedrock model pricing per 1K tokens in AUD based on model type.

        USD prices converted to AUD using 1.52 exchange rate (Dec 2024).
        Blended rates assume 50% input / 50% output token mix:
        - Claude Opus: $0.015 USD → $0.0228 AUD per 1K tokens (estimated blended)
        - Claude Sonnet: $0.009 USD → $0.01368 AUD per 1K tokens (blended: $0.003 in + $0.015 out)
        - Claude Haiku: $0.00025 USD → $0.00038 AUD per 1K tokens (estimated blended)
        - Amazon Nova Pro: $0.002 USD → $0.00304 AUD per 1K tokens (blended: $0.0008 in + $0.0032 out)
        - Amazon Nova Lite: $0.00006 USD → $0.000091 AUD per 1K tokens (estimated blended)
        - Amazon Nova Micro: $0.000035 USD → $0.000053 AUD per 1K tokens (estimated blended)
        """
        model_lower = self.model.lower()

        if "opus" in model_lower:
            return 0.0228
        elif "sonnet" in model_lower:
            return 0.01368
        elif "haiku" in model_lower:
            return 0.00038
        elif "nova-pro" in model_lower or "novapro" in model_lower:
            return 0.00304
        elif "nova-lite" in model_lower or "novalite" in model_lower:
            return 0.000091
        elif "nova-micro" in model_lower or "novamicro" in model_lower:
            return 0.000053
        elif "nova" in model_lower:
            logger.info(f"Using Nova Pro pricing for model '{self.model}'")
            return 0.00304
        else:
            logger.warning(
                f"Unknown Bedrock model '{self.model}', using default cost of 0.0 AUD"
            )
            return 0.0
