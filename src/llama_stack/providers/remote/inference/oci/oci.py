# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import AsyncIterator, Iterable
from typing import Any
import json

import httpx
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    OnDemandServingMode,
    GenericChatRequest,
    CohereMessage,
    CohereChatRequest,
)
from oci.generative_ai.generative_ai_client import GenerativeAiClient
from oci.generative_ai.models import ModelCollection
from openai._base_client import DefaultAsyncHttpxClient

from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.oci.auth import OciInstancePrincipalAuth, OciUserPrincipalAuth
from llama_stack.providers.remote.inference.oci.config import OCIConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack_api import (
    ModelType,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

logger = get_logger(name=__name__, category="inference::oci")

OCI_AUTH_TYPE_INSTANCE_PRINCIPAL = "instance_principal"
OCI_AUTH_TYPE_CONFIG_FILE = "config_file"
VALID_OCI_AUTH_TYPES = [OCI_AUTH_TYPE_INSTANCE_PRINCIPAL, OCI_AUTH_TYPE_CONFIG_FILE]
DEFAULT_OCI_REGION = "us-ashburn-1"

MODEL_CAPABILITIES = ["TEXT_GENERATION", "TEXT_SUMMARIZATION", "TEXT_EMBEDDINGS", "CHAT"]


class OCIInferenceAdapter(OpenAIMixin):
    config: OCIConfig
    # OCID mapping removed - OCI accepts display names directly

    async def initialize(self) -> None:
        """Initialize and validate OCI configuration."""
        if self.config.oci_auth_type not in VALID_OCI_AUTH_TYPES:
            raise ValueError(
                f"Invalid OCI authentication type: {self.config.oci_auth_type}."
                f"Valid types are one of: {VALID_OCI_AUTH_TYPES}"
            )

        if not self.config.oci_compartment_id:
            raise ValueError("OCI_COMPARTMENT_OCID is a required parameter. Either set in env variable or config.")

    def get_base_url(self) -> str:
        region = self.config.oci_region or DEFAULT_OCI_REGION
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/v1"

    def get_api_key(self) -> str | None:
        # OCI doesn't use API keys, it uses request signing
        return "<NOTUSED>"

    def get_extra_client_params(self) -> dict[str, Any]:
        """
        Get extra parameters for the AsyncOpenAI client, including OCI-specific auth and headers.
        """
        auth = self._get_auth()
        compartment_id = self.config.oci_compartment_id or ""

        return {
            "http_client": DefaultAsyncHttpxClient(
                auth=auth,
                headers={
                    "CompartmentId": compartment_id,
                },
            ),
        }

    def _get_oci_signer(self) -> oci.signer.AbstractBaseSigner | None:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return None

    def _get_oci_config(self) -> dict:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            config = {"region": self.config.oci_region}
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            config = oci.config.from_file(self.config.oci_config_file_path, self.config.oci_config_profile)
            if not config.get("region"):
                raise ValueError(
                    "Region not specified in config. Please specify in config or with OCI_REGION env variable."
                )

        return config

    def _get_auth(self) -> httpx.Auth:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return OciInstancePrincipalAuth()
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            return OciUserPrincipalAuth(
                config_file=self.config.oci_config_file_path, profile_name=self.config.oci_config_profile
            )
        else:
            raise ValueError(f"Invalid OCI authentication type: {self.config.oci_auth_type}")

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        List available models from OCI Generative AI service and store OCID mappings.
        """
        oci_config = self._get_oci_config()
        oci_signer = self._get_oci_signer()
        compartment_id = self.config.oci_compartment_id or ""

        if oci_signer is None:
            client = GenerativeAiClient(config=oci_config)
        else:
            client = GenerativeAiClient(config=oci_config, signer=oci_signer)

        models: ModelCollection = client.list_models(
            compartment_id=compartment_id, lifecycle_state="ACTIVE"
        ).data

        seen_models = set()
        model_ids = []
        for model in models.items:
            if model.time_deprecated or model.time_on_demand_retired:
                continue

            if "CHAT" not in model.capabilities or "FINE_TUNE" in model.capabilities:
                continue

            # Use display_name + model_type as the key to avoid conflicts
            model_key = (model.display_name, ModelType.llm)
            if model_key in seen_models:
                continue

            seen_models.add(model_key)
            # Framework automatically adds provider_id prefix
            model_ids.append(model.display_name)


        logger.info(f"Loaded {len(model_ids)} OCI models")
        return model_ids

    async def openai_embeddings(self, params: OpenAIEmbeddingsRequestWithExtraBody) -> OpenAIEmbeddingsResponse:
        # The constructed url is a mask that hits OCI's "chat" action, which is not supported for embeddings.
        raise NotImplementedError("OCI Provider does not (currently) support embeddings")

    async def openai_chat_completion(
        self, params: OpenAIChatCompletionRequestWithExtraBody
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """
        Override the OpenAI chat completion to use native OCI SDK.
        OCI doesn't have OpenAI-compatible endpoints, so we use the native API.
        """
        # Get the actual OCI model name from the model store
        # The model store lookup is synchronous so get model_id directly
        model_id = params.model
        # Strip the oci/ prefix if present to get the display name
        provider_model_id = model_id.replace("oci/", "") if model_id.startswith("oci/") else model_id

        # Use display name directly (OCI accepts display names)
        logger.info(f"Using model display name: {provider_model_id}")

        # Set up OCI client
        oci_config = self._get_oci_config()
        oci_signer = self._get_oci_signer()

        if oci_signer is None:
            inference_client = GenerativeAiInferenceClient(config=oci_config)
        else:
            inference_client = GenerativeAiInferenceClient(config=oci_config, signer=oci_signer)

        # Convert OpenAI messages to OCI format
        oci_messages = []
        for msg in params.messages:
            role = msg.role if hasattr(msg, 'role') else "user"
            content = msg.content if hasattr(msg, 'content') else ""

            # OCI uses uppercase roles
            oci_role = role.upper() if role.lower() in ["user", "assistant", "system"] else "USER"
            oci_messages.append({"role": oci_role, "content": [{"type": "TEXT", "text": content}]})

        # Build chat request using GenericChatRequest
        chat_request = GenericChatRequest()
        chat_request.messages = oci_messages
        chat_request.api_format = "GENERIC"
        chat_request.max_tokens = params.max_tokens or 512
        chat_request.temperature = params.temperature if params.temperature is not None else 1.0
        chat_request.top_p = params.top_p if params.top_p is not None else 1.0
        chat_request.is_stream = params.stream or False

        # Create ChatDetails with OCID
        chat_details = ChatDetails()
        chat_details.serving_mode = OnDemandServingMode(model_id=provider_model_id)  # Use display name
        chat_details.compartment_id = self.config.oci_compartment_id
        chat_details.chat_request = chat_request

        try:
            if params.stream:
                return self._handle_streaming_response(inference_client, chat_details, params.model)
            else:
                return await self._handle_non_streaming_response(inference_client, chat_details, params.model)
        except Exception as e:
            logger.error(f"Error during OCI chat completion: {e}", exc_info=True)
            raise

    async def _handle_non_streaming_response(
        self, client: GenerativeAiInferenceClient, chat_details: ChatDetails, model_id: str
    ) -> OpenAIChatCompletion:
        """Handle non-streaming chat completion response from OCI."""
        import time
        import asyncio

        # OCI SDK is synchronous, so run in thread pool
        response = await asyncio.to_thread(client.chat, chat_details)

        # Transform OCI response to OpenAI format
        oci_response = response.data.chat_response

        choices = []
        if hasattr(oci_response, "choices") and oci_response.choices:
            for idx, choice in enumerate(oci_response.choices):
                message_content = ""
                if hasattr(choice, "message") and choice.message:
                    if hasattr(choice.message, "content") and choice.message.content:
                        for content_part in choice.message.content:
                            if hasattr(content_part, "text"):
                                message_content += content_part.text

                choices.append({
                    "index": idx,
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                    },
                    "finish_reason": "stop",
                })

        return OpenAIChatCompletion(
            id=f"oci-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=model_id,
            choices=choices,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    async def _handle_streaming_response(
        self, client: GenerativeAiInferenceClient, chat_details: ChatDetails, model_id: str
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        """Handle streaming chat completion response from OCI."""
        import time
        import asyncio
        import queue
        import threading

        chunk_id = f"oci-{int(time.time())}"
        event_queue = queue.Queue()

        # Run the OCI streaming in a thread
        def _stream_events():
            try:
                response = client.chat(chat_details)
                for event in response.data.events():
                    event_queue.put(("event", event))
            except Exception as e:
                event_queue.put(("error", e))
            finally:
                event_queue.put(("done", None))

        # Start streaming thread
        thread = threading.Thread(target=_stream_events, daemon=True)
        thread.start()

        # Yield chunks as they arrive
        while True:
            # Get event from queue with timeout
            try:
                msg_type, data = await asyncio.to_thread(event_queue.get, timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if msg_type == "done":
                break
            elif msg_type == "error":
                logger.error(f"Error in streaming thread: {data}")
                break
            elif msg_type == "event":
                event = data
                try:
                    # Parse the SSE event
                    if hasattr(event, "data"):
                        event_data = json.loads(event.data)

                        # Extract delta content from OCI's format
                        delta_content = ""

                        # OCI format: {"message": {"content": [{"type": "TEXT", "text": "..."}]}}
                        if "message" in event_data:
                            message = event_data["message"]
                            if "content" in message and message["content"]:
                                for content_part in message["content"]:
                                    if "text" in content_part:
                                        delta_content += content_part["text"]

                        # Yield chunk
                        if delta_content:  # Only yield if there's content
                            chunk = OpenAIChatCompletionChunk(
                                id=chunk_id,
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model_id,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {"content": delta_content},
                                        "finish_reason": "",
                                    }
                                ],
                            )
                            yield chunk
                except Exception as e:
                    logger.error(f"Error parsing streaming event: {e}")
                    continue

        # Send final chunk with finish_reason
        final_chunk = OpenAIChatCompletionChunk(
            id=chunk_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model_id,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        )
        yield final_chunk
