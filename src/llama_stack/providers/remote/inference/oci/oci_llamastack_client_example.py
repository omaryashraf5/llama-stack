#!/usr/bin/env python3
"""
Example script demonstrating how to use Oracle OCI inference with llama stack client.

Prerequisites:
1. Install OCI SDK: pip install oci
2. Install llama-stack-client: pip install llama-stack-client
3. Configure OCI credentials at ~/.oci/config
4. Set OCI_COMPARTMENT_OCID environment variable
5. Start llama stack server with OCI config: llama stack run oci_config.yaml
"""

import os

from llama_stack_client import LlamaStackClient


def main():
    # Connect to llama stack server running with OCI inference
    client = LlamaStackClient(base_url="http://localhost:8321")

    print("=" * 80)
    print("OCI Inference with Llama Stack Client Example")
    print("=" * 80)

    # 1. List available models from OCI
    print("\n1. Listing available OCI models...")
    try:
        models = client.models.list()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model.id}")
            print(f"    Provider: {model.owned_by}")
            if hasattr(model, "custom_metadata") and model.custom_metadata:
                print(f"    Metadata: {model.custom_metadata}")
    except Exception as e:
        print(f"Error listing models: {e}")
        print("\nMake sure:")
        print("  1. OCI credentials are configured")
        print("  2. OCI_COMPARTMENT_OCID is set")
        print("  3. Generative AI service is enabled in your compartment")
        return

    # 2. Simple inference example
    print("\n2. Running inference with OCI model...")

    # Use a model from OCI Generative AI service
    if len(models) == 0:
        print("⚠️  No models available, skipping inference examples")
        return

    model_id = models[0].id  # Use first available model

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "What is Oracle Cloud Infrastructure?"}
            ],
            temperature=0.7,
            max_tokens=512,
        )

        print(f"Model: {model_id}")
        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Error during inference: {e}")
        print(
            f"\nMake sure the model '{model_id}' is available in your OCI compartment"
        )

    # 3. Streaming inference example
    print("\n3. Running streaming inference with OCI model...")

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "List 3 benefits of using OCI for AI workloads.",
                }
            ],
            temperature=0.7,
            max_tokens=256,
            stream=True,
        )

        print(f"Model: {model_id}")
        print("Streaming response: ", end="", flush=True)

        for chunk in stream:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    print(delta.content, end="", flush=True)

        print("\n")

    except Exception as e:
        print(f"Error during streaming inference: {e}")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Check for required environment variable
    if not os.getenv("OCI_COMPARTMENT_OCID"):
        print("WARNING: OCI_COMPARTMENT_OCID environment variable not set")
        print(
            "Please set it with: export OCI_COMPARTMENT_OCID='ocid1.compartment.oc1..xxx'"
        )
        print("\nContinuing anyway, but API calls may fail...\n")

    main()
