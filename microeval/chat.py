#!/usr/bin/env python3
"""
Interactive chat loop with LLM providers.
"""

import asyncio

from microeval.llm import LLMService, get_llm_client


async def setup_async_exception_handler():
    loop = asyncio.get_event_loop()

    def silence_event_loop_closed(loop, context):
        if "exception" not in context or not isinstance(
            context["exception"], (RuntimeError, GeneratorExit)
        ):
            loop.default_exception_handler(context)

    loop.set_exception_handler(silence_event_loop_closed)


async def amain(service: LLMService):
    await setup_async_exception_handler()
    async with get_llm_client(service) as client:
        print(f"Chat loop with {service}-{client.model}")
        messages = []
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            messages.append({"role": "user", "content": user_input})
            result = await client.get_completion(messages)
            response_text = result.get("text", "")
            print(f"\nResponse: {response_text}")
            messages.append({"role": "assistant", "content": response_text})


def main(service: LLMService = "openai"):
    try:
        asyncio.run(amain(service))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
