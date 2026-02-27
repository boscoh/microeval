"""
Simple chat client abstraction for LLM providers.

- Async only, with proper async context management
- No streaming, only conversation with tools and embeddings
- Mostly OpenAI JSON structure, but without choices, and easier token usage metadata
- No langchain, litellm etc., just vendor-provided Python packages
"""

import asyncio
import configparser
import json
import logging
import os
import time
import uuid
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
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

AIOBOTO3_CLEANUP_DELAY_SECONDS = 0.1


@lru_cache
def load_config() -> Dict[str, Any]:
    """Load and return the models configuration from models.json.

    :return: Configuration dictionary with chat_models, embed_models, and pricing.
    """
    config_path = Path(__file__).parent / "models.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        logger.info(f"Loaded selectable models from '{config_path}'")
        return config


class SimpleLLMClient(ABC):
    """Async LLM client interface with a simplified, consistent message format.

    The interface follows OpenAI's message structure closely, since it is the
    de-facto standard and all major providers either adopt it or offer compatible
    mappings. However, several OpenAI quirks are deliberately removed:

    No choices wrapper
        OpenAI wraps responses in a choices list to support multiple completions
        (n > 1), which we never use. The single completion is returned directly
        as {text, metadata, tool_calls}.

    Flat usage metadata
        Token counts and timing live in a single metadata.usage dict rather than
        split across the response and a separate usage object.

    Consistent tool-call structure
        Assistant tool_calls use {id, function} on both input and output, so the
        structure is symmetric. The raw Bedrock and Ollama APIs each use their
        own conventions; we normalise to OpenAI's shape.

    No type: "function" noise
        The type field on tool call objects is omitted — it has always been
        "function" and carries no information.
    """

    @abstractmethod
    async def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Get a chat completion from the model.

        Each message has a role: system, user, assistant, or tool.
        Assistant messages may carry tool_calls; tool messages must include
        tool_call_id referencing the call. Tool calls use the structure::

            {
              "id": "call_123",
              "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"Paris\"}"
              }
            }

        :param messages: Conversation history as a list of message dicts.
        :param tools: Optional list of tool/function definitions (JSON Schema format).
        :param max_tokens: Maximum tokens to generate; uses model default if None.
        :param temperature: Sampling temperature 0.0–1.0 (0.0 = deterministic).
        :return: On success::

                {
                  "text": str,
                  "metadata": {
                    "usage": {
                      "prompt_tokens": int,
                      "completion_tokens": int,
                      "elapsed_seconds": float
                    },
                    "model": str,
                    "finish_reason": str
                  },
                  "tool_calls": [
                    {"id": str, "function": {"name": str, "arguments": str}}
                  ]
                }

        :raises Exception: Propagates any exception from the underlying API.
        """
        pass

    @abstractmethod
    async def embedding(self, input: str) -> List[float]:
        """Generate a text embedding vector for the given input string."""
        pass

    def get_token_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate token cost in USD using pricing from models.json.

        Returns None if pricing data is not available for this model.
        """
        if not hasattr(self, "service"):
            return None

        pricing = load_config().get("pricing", {}).get(self.service, {})
        model_pricing = pricing.get(self.model)

        if not model_pricing:
            for model_key in pricing:
                if model_key in self.model:
                    model_pricing = pricing[model_key]
                    break

        if model_pricing:
            return (prompt_tokens / 1_000_000) * model_pricing["prompt"] + \
                   (completion_tokens / 1_000_000) * model_pricing["completion"]

        return None

    def _build_usage_metadata(
        self, prompt_tokens: int, completion_tokens: int, elapsed_seconds: float
    ) -> Dict[str, Any]:
        """Build standardized usage metadata structure.

        :param prompt_tokens: Number of prompt tokens used.
        :param completion_tokens: Number of completion tokens used.
        :param elapsed_seconds: Elapsed time for the request.
        :return: Dict with token counts and elapsed time.
        """
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_seconds": elapsed_seconds,
        }

    def _build_success_response(
        self,
        text: str,
        usage: Dict[str, Any],
        finish_reason: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build standardized success response structure.

        :param text: Response text content.
        :param usage: Usage metadata from _build_usage_metadata().
        :param finish_reason: Why generation stopped.
        :param tool_calls: Optional list of tool calls.
        :return: Dict with response data and metadata.
        """
        result = {
            "text": text,
            "metadata": {
                "usage": usage,
                "model": self.model,
                "finish_reason": finish_reason,
            },
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def _format_tool_call_output(
        self, name: str, arguments: str, tool_call_id: str
    ) -> Dict[str, Any]:
        """Format a tool call into standardized output structure.

        :param name: Function/tool name.
        :param arguments: JSON string of arguments.
        :param tool_call_id: Unique identifier for this tool call.
        :return: Dict with function call details.
        """
        return {
            "id": tool_call_id,
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

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


class OllamaClient(SimpleLLMClient):
    def __init__(self, model: str = None):
        """Initialize Ollama chat client.

        :param model: Name of the Ollama model to use (default from config).
        :raises RuntimeError: If Ollama is not running or the model is not available.
        """
        self.service = "ollama"
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
                new_tool_calls = []
                for tc in msg["tool_calls"]:
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse tool arguments: {args[:50]}... Error: {e}")
                            args = {}
                    elif not isinstance(args, dict):
                        args = {}
                    new_tool_calls.append({
                        "id": tc.get("id", ""),
                        "function": {"name": tc["function"]["name"], "arguments": args},
                    })
                transformed.append({**msg, "tool_calls": new_tool_calls})
            else:
                transformed.append(msg)
        return transformed

    async def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Ollama implementation of completion with tool support."""
        await self.connect()

        start_time = time.time()

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            transformed_messages = self._transform_messages(messages)

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

            response_text = (
                (message_dict.get("content") or "")
                if isinstance(message_dict, dict)
                else ""
            )
            raw_tool_calls = (
                message_dict.get("tool_calls")
                if isinstance(message_dict, dict)
                else None
            )

            completion_tokens = len(response_text.split()) if response_text else 0
            prompt_tokens = sum(len((m.get("content") or "").split()) for m in messages)

            tool_calls = None
            if raw_tool_calls:
                tool_calls = []
                for tool_call in raw_tool_calls:
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        func = tool_call["function"]
                        name = func.get("name", "")
                        args = func.get("arguments", {})
                        tc_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    elif hasattr(tool_call, "function"):
                        function = tool_call.function
                        name = getattr(function, "name", "")
                        args = getattr(function, "arguments", {})
                        tc_id = getattr(tool_call, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                    else:
                        continue

                    if isinstance(args, dict):
                        args = json.dumps(args)
                    elif not isinstance(args, str):
                        args = str(args)

                    tool_calls.append(self._format_tool_call_output(name, args, tc_id))

            usage = self._build_usage_metadata(
                prompt_tokens, completion_tokens, elapsed_seconds
            )
            return self._build_success_response(response_text, usage, done_reason, tool_calls)
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    async def embedding(self, input: str) -> List[float]:
        """Generate text embeddings using Ollama's embedding capabilities."""
        await self.connect()

        try:
            response = await self.client.embeddings(model=self.model, prompt=input)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error calling Ollama embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Returns 0.0 since Ollama runs locally with no API costs.

        :param prompt_tokens: Unused.
        :param completion_tokens: Unused.
        :return: Always 0.0.
        """
        return 0.0


class OpenAIClient(SimpleLLMClient):
    def __init__(
        self,
        model: str = None,
    ):
        """Initialize OpenAI chat client.

        :param model: Name of the OpenAI model to use (default from config).
        :raises ValueError: If ``OPENAI_API_KEY`` is not set.
        :raises RuntimeError: If the API key is invalid or the model is not available.
        """
        self.service = "openai"
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

    def _handle_incomplete_tool_sequence(
        self, msg: Dict, i: int, messages: List[Dict]
    ) -> Optional[Dict]:
        """Handle assistant message with tool_calls that has no following tool message.

        This occurs when loading conversation history with incomplete tool sequences.
        OpenAI requires: assistant (with tool_calls) → tool → assistant.

        :param msg: The assistant message with tool_calls.
        :param i: Index of the message in the list.
        :param messages: Full message list.
        :return: Cleaned message without tool_calls if incomplete, original if complete.
        """
        has_following_tool = (
            i + 1 < len(messages) and messages[i + 1].get("role") == "tool"
        )
        if not has_following_tool:
            clean_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            if not clean_msg.get("content"):
                clean_msg["content"] = ""
            logger.debug("Stripped tool_calls from assistant message (incomplete sequence)")
            return clean_msg
        return msg

    def _should_skip_orphaned_tool_message(self, in_active_sequence: bool) -> bool:
        """Check if a tool message should be skipped because it's orphaned.

        :param in_active_sequence: Whether we're currently in an active tool sequence.
        :return: True if the message should be skipped.
        """
        if not in_active_sequence:
            logger.debug("Skipping orphaned tool message (no active tool sequence)")
            return True
        return False

    def _transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform intermediate message format to OpenAI API format.

        OpenAI requires strict message sequencing for tool calls:
        assistant (with tool_calls) → tool (with tool_call_id) → assistant.
        Filters out incomplete tool sequences to avoid 400 validation errors.

        :param messages: List of messages in intermediate format.
        :return: List of messages formatted for the OpenAI API.
        """
        formatted_messages = []
        in_active_tool_sequence = False

        for i, msg in enumerate(messages):
            role = msg["role"]

            if role == "system":
                in_active_tool_sequence = False
                formatted_messages.append(msg)
            elif role == "tool":
                if not self._should_skip_orphaned_tool_message(in_active_tool_sequence):
                    formatted_messages.append(
                        {
                            "role": "tool",
                            "content": msg.get("content", ""),
                            "tool_call_id": msg.get("tool_call_id", ""),
                        }
                    )
            elif role == "assistant":
                if "tool_calls" in msg:
                    cleaned_msg = self._handle_incomplete_tool_sequence(msg, i, messages)
                    if cleaned_msg:
                        in_active_tool_sequence = "tool_calls" in cleaned_msg
                        formatted_messages.append(cleaned_msg)
                else:
                    in_active_tool_sequence = False
                    formatted_messages.append(msg)
            elif role == "user":
                in_active_tool_sequence = False
                formatted_messages.append(msg)
            else:
                formatted_messages.append(msg)

        return formatted_messages

    async def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """OpenAI implementation of completion with full tool support."""
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
                usage = self._build_usage_metadata(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    elapsed_seconds,
                )
            else:
                usage = self._build_usage_metadata(0, 0, elapsed_seconds)

            tool_calls = None
            if completion.choices and completion.choices[0].message.tool_calls:
                tool_calls = [
                    self._format_tool_call_output(
                        tc.function.name, tc.function.arguments, tc.id
                    )
                    for tc in completion.choices[0].message.tool_calls
                ]

            finish_reason = (
                completion.choices[0].finish_reason
                if completion.choices and completion.choices[0].finish_reason
                else "stop"
            )
            return self._build_success_response(text, usage, finish_reason, tool_calls)
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            raise

    async def embedding(self, input: str) -> List[float]:
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


class GroqClient(OpenAIClient):
    """Groq chat client that inherits from OpenAI client (Groq uses OpenAI-compatible API)."""

    def __init__(self, model: str = None):
        super().__init__(model=model)
        self.service = "groq"

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

    async def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Groq implementation with intelligent parallel_tool_calls handling.

        Only certain Groq models support parallel tool calling.
        GPT-OSS models (openai/gpt-oss-*) do NOT support parallel_tool_calls.
        """
        await self.connect()
        start_time = time.time()

        try:
            formatted_messages = self._transform_messages(messages)
            api_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if tools:
                api_params["tools"] = tools
                supports_parallel = any([
                    self.model.startswith("llama-3."),
                    self.model.startswith("llama-4"),
                    self.model.startswith("meta-llama/llama-4"),
                    self.model.startswith("qwen/"),
                    self.model.startswith("moonshotai/"),
                ])
                if supports_parallel:
                    api_params["parallel_tool_calls"] = True

            completion = await self.client.chat.completions.create(**api_params)
            elapsed_seconds = time.time() - start_time

            text = completion.choices[0].message.content if completion.choices else ""

            if hasattr(completion, "usage") and completion.usage:
                usage = self._build_usage_metadata(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    elapsed_seconds,
                )
            else:
                usage = self._build_usage_metadata(0, 0, elapsed_seconds)

            tool_calls = None
            if completion.choices and completion.choices[0].message.tool_calls:
                tool_calls = [
                    self._format_tool_call_output(
                        tc.function.name, tc.function.arguments, tc.id
                    )
                    for tc in completion.choices[0].message.tool_calls
                ]

            finish_reason = (
                completion.choices[0].finish_reason
                if completion.choices and completion.choices[0].finish_reason
                else "stop"
            )
            return self._build_success_response(text, usage, finish_reason, tool_calls)
        except Exception as e:
            logger.error(f"Error calling Groq: {e}")
            raise

    async def embedding(self, input: str) -> List[float]:
        """Groq does not currently support embeddings."""
        raise NotImplementedError(
            "Groq does not currently support text embeddings. "
            "Please use OpenAI or another provider for embedding generation."
        )


@lru_cache(maxsize=None)
def get_aws_config(is_raise_exception: bool = True):
    """Return AWS configuration dict for boto3 client initialization.

    Searches for AWS profiles and credentials, validates them, and checks for
    SSO token expiration. Uses ``AWS_PROFILE`` and ``AWS_REGION`` env vars if set.

    :param is_raise_exception: If True, raise on credential errors; otherwise log and return partial config.
    :return: Dict with optional ``profile_name`` and ``region_name`` keys, suitable for unpacking into boto3 constructors.
    """
    aws_config = {}
    available_profiles = set()
    credentials_path = os.path.expanduser("~/.aws/credentials")
    config_path = os.path.expanduser("~/.aws/config")

    # Discover available profiles from credentials file
    if os.path.exists(credentials_path):
        config = configparser.ConfigParser()
        config.read(credentials_path)
        available_profiles.update(config.sections())

    # Discover available profiles from config file
    if os.path.exists(config_path):
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
            logger.info(
                f"AWS profile '{profile_name}' not found, using default credential chain"
            )
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
                        "No AWS credentials found.\n"
                        "To configure: aws configure\n"
                        "Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
            return aws_config

        if not credentials.access_key or not credentials.secret_key:
            if is_raise_exception:
                raise ValueError(
                    "Incomplete AWS credentials (missing access key or secret key)"
                )
            logger.warning("Incomplete AWS credentials")
            return aws_config

        # Validate credentials work
        sts = session.client("sts")
        sts.get_caller_identity()

        # Check for SSO expiry by reading cache files (more reliable than frozen credentials)
        if profile_name:
            sso_cache_dir = Path.home() / ".aws" / "sso" / "cache"
            if sso_cache_dir.exists():
                for cache_file in sso_cache_dir.glob("*.json"):
                    try:
                        with open(cache_file) as f:
                            cache_data = json.load(f)
                        if "expiresAt" in cache_data:
                            expires_at = datetime.fromisoformat(
                                cache_data["expiresAt"].replace("Z", "+00:00")
                            )
                            if expires_at < datetime.now(timezone.utc):
                                msg = (
                                    f"AWS SSO session expired for profile '{profile_name}'. "
                                    f"Run: aws sso login --profile {profile_name}"
                                )
                                if is_raise_exception:
                                    raise ValueError(msg)
                                logger.warning(msg)
                                return aws_config
                    except (json.JSONDecodeError, KeyError):
                        continue

        return aws_config

    except ClientError as e:
        if is_raise_exception:
            raise
        error_code = e.response["Error"]["Code"]

        if error_code == "ExpiredToken":
            # Check if SSO to provide better error message
            profile_to_check = aws_config.get("profile_name", profile_name)
            if profile_to_check and os.path.exists(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                section = f"profile {profile_to_check}"
                if config.has_section(section) and config.has_option(
                    section, "sso_start_url"
                ):
                    login_cmd = f"aws sso login --profile {profile_to_check}"
                    logger.warning(f"AWS SSO session expired. Please run: {login_cmd}")
                    return aws_config
            logger.warning("AWS credentials have expired")
        elif error_code == "InvalidClientTokenId":
            logger.warning(
                "AWS credentials are invalid. Please reconfigure: aws configure"
            )
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
        """Initialize Bedrock chat client using the Converse API.

        :param model: Claude model ID for Bedrock (default from config).
        """
        self.service = "bedrock"
        self.model = model
        self.client = None
        self._session = None
        self._client_ctx = None
        self._closed = True

    async def connect(self):
        """Initialize the async client session and client."""
        if self.client is not None and not self._closed:
            return

        logger.info(f"Initializing 'bedrock:{self.model}'")
        aws_config = get_aws_config()
        self._session = aioboto3.Session(**aws_config)
        self._client_ctx = self._session.client("bedrock-runtime")
        self.client = await self._client_ctx.__aenter__()
        self._closed = False

    async def close(self):
        """Close the client and properly clean up aiohttp sessions."""
        if self.client is not None and not self._closed:
            await self._client_ctx.__aexit__(None, None, None)
            self.client = None
            self._closed = True
            await asyncio.sleep(AIOBOTO3_CLEANUP_DELAY_SECONDS)

    def _build_result_from_response(
        self, response: Any, start_time: float
    ) -> Dict[str, Any]:
        """Build standardized result structure from Bedrock Converse API response.

        :param response: Bedrock Converse API response dict, or error string.
        :param start_time: Request start time for elapsed calculation.
        :return: Standardized response dict with text, metadata, and optional tool_calls.
        """
        text_parts = []
        tool_calls = []
        usage_dict = {}

        if isinstance(response, str):
            text_parts.append(response)
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
                            self._format_tool_call_output(
                                tool_use["name"],
                                json.dumps(tool_use.get("input", {})),
                                tool_use.get("toolUseId", ""),
                            )
                        )
            usage_dict = response.get("usage", {})
            stop_reason = response.get("stopReason", "unknown")

        usage = self._build_usage_metadata(
            usage_dict.get("inputTokens", 0),
            usage_dict.get("outputTokens", 0),
            time.time() - start_time,
        )
        return self._build_success_response(
            "\n".join(text_parts).strip(),
            usage,
            stop_reason,
            tool_calls if tool_calls else None,
        )

    def _batch_consecutive_tool_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch consecutive tool messages into single user messages.

        Bedrock Converse API requires all tool results following an assistant
        message with tool calls to be in a single user message with multiple
        toolResult blocks in the content array.

        :param messages: List of messages in intermediate format.
        :return: Messages with consecutive tool results batched together.
        """
        batched = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "tool":
                tool_batch = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_batch.append(messages[i])
                    i += 1
                if len(tool_batch) > 1:
                    logger.debug(f"Batching {len(tool_batch)} tool results into single user message")
                    combined_content = [
                        {
                            "toolResult": {
                                "toolUseId": tm.get("tool_call_id", ""),
                                "content": [{"text": str(tm.get("content", ""))}],
                                "status": tm.get("status", "success"),
                            }
                        }
                        for tm in tool_batch
                    ]
                    batched.append({"role": "user", "content": combined_content})
                else:
                    batched.append(tool_batch[0])
            else:
                batched.append(msg)
                i += 1
        return batched

    def _transform_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Transform intermediate message format to Bedrock Converse API format.

        Batches consecutive tool messages before formatting to meet Bedrock's
        requirement that all tool results be in a single user message.

        :param messages: List of messages in intermediate format.
        :param tools: Optional list of tool definitions.
        :return: Partially filled request_kwargs with messages, system, and toolConfig.
        """
        messages = self._batch_consecutive_tool_messages(messages)

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
                    else:
                        logger.warning(
                            f"Tool call missing id, skipping: "
                            f"{tool_call.get('function', {}).get('name', 'unknown')}"
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

    async def completion(
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
            response = await self.client.converse(**request_kwargs)
            return self._build_result_from_response(response, start_time)
        except Exception as e:
            logger.error(f"Error in Bedrock completion: {e}")
            raise

    async def embedding(self, input: str) -> List[float]:
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


LLM_CLIENTS = {
    "openai": OpenAIClient,
    "ollama": OllamaClient,
    "bedrock": BedrockClient,
    "groq": GroqClient,
}

LLMService = Literal[*LLM_CLIENTS]


def get_llm_client(client_type: LLMService, **kwargs) -> SimpleLLMClient:
    """Return a chat client satisfying the SimpleLLMClient interface.

    :param client_type: One of ``'openai'``, ``'ollama'``, ``'bedrock'``, ``'groq'``.
    :param kwargs: Passed to the client constructor; ``model`` defaults to the first entry in models.json.
    :return: Configured SimpleLLMClient instance.
    """
    client_type = client_type.lower()

    if "model" not in kwargs:
        default_models = load_config().get("chat_models", {}).get(client_type, [])
        if default_models:
            kwargs["model"] = (
                default_models[0]
                if isinstance(default_models, list)
                else default_models
            )

    if client_type not in LLM_CLIENTS:
        raise ValueError(f"Unknown chat client type: {client_type}")
    return LLM_CLIENTS[client_type](**kwargs)
