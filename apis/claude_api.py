import time

import anthropic
import imghdr

api = "YOUR_API"

client = anthropic.Anthropic(
    api_key=api,
)


def chat_generation_with_apeer(prompt, image_pack):
    image_blocks = image_pack
    last_prompt = []
    last_prompt.append({
        "type": "text",
        "text": prompt[-1]
    })
    if image_pack is not None:
        for image_block in image_blocks:
            last_prompt.append(image_block)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241122",
        max_tokens=8192,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[0]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[1]
                    }
                ]
            },
            {
                "role": "user",
                "content": last_prompt
            },
        ]
    )
    response = (message.content[0].text)

    return response


def chat_generation_with_urial(prompt, image_pack):
    image_blocks = image_pack
    last_prompt = []
    last_prompt.append({
        "type": "text",
        "text": prompt[-1]
    })

    if image_pack is not None:
        for image_block in image_blocks:
            last_prompt.append(image_block)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241122",
        max_tokens=8192,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[0]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[1]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[2]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[3]
                    }
                ]
            },
            {
                "role": "user",
                "content": last_prompt
            },
        ]
    )
    response = (message.content[0].text)

    return response


def chat_generation_with_polish_urial(prompt, image_pack):
    image_blocks = image_pack
    last_prompt = []
    last_prompt.append({
        "type": "text",
        "text": prompt[-3]
    })

    if image_pack is not None:
        for image_block in image_blocks:
            last_prompt.append(image_block)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241122",
        max_tokens=8192,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[0]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[1]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[2]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[3]
                    }
                ]
            },
            {
                "role": "user",
                "content": last_prompt
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[-2]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[-1]
                    }
                ]
            }
        ]
    )
    response = (message.content[0].text)

    return response


def single_chat_generation(prompt):

    message = client.messages.create(
        model="claude-3-5-sonnet-20241122",
        max_tokens=8192,
        temperature=0,
        top_p=0.1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    response = (message.content[0].text)

    return response


