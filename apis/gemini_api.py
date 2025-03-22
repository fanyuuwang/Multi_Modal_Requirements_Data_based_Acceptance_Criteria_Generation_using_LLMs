api_key = "YOUR API"
from google.api_core import exceptions
import time
import google.generativeai as genai
from google.genai import types

genai.configure(api_key=api_key)
client = genai.GenerativeModel('gemini-2.0-flash')

def chat_generation_with_apeer(prompt, image_pack):
    contents = []

    # Add the first user turn
    contents.append({
        "role": "user",
        "parts": [{"text": prompt[0]}]
    })

    # Add the assistant's turn
    contents.append({
        "role": "assistant",
        "parts": [{"text": prompt[1]}]
    })

    # Construct the parts for the final user prompt and images
    last_prompt_parts = []
    last_prompt_parts.append({"text": prompt[-1]})  # Add the final text prompt
    if image_pack is not None:
        for image_block in image_pack:
            image_data = {
                "mime_type": "image/jpeg",  # Or image/png, based on your image
                "data": image_block["source"]["data"]
            }
            last_prompt_parts.append({"inline_data": image_data})  # Add image as inline_data

    # Add the final user turn with the combined text and image parts
    contents.append({
        "role": "user",
        "parts": last_prompt_parts
    })


    model = genai.GenerativeModel('gemini-2.0-flash')  # Initialize model here
    try:
        response = model.generate_content(contents=contents)
    except exceptions.ResourceExhausted as e:
        print(f"Quota error: {e}")
        print("Retrying in 60 seconds")
        time.sleep(60)
        try:
            response = model.generate_content(contents=contents)
            # print(response.text)
        except exceptions.ResourceExhausted as e:
            print("Retry Failed, Quota still exceeded")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return response.text  # Access the text directly


def chat_generation_with_urial(prompt, image_pack):
    contents = []

    # Add the first user turn
    contents.append({
        "role": "user",
        "parts": [{"text": prompt[0]}]
    })

    # Add the assistant's turn
    contents.append({
        "role": "assistant",
        "parts": [{"text": prompt[1]}]
    })

    contents.append({
        "role": "user",
        "parts": [{"text": prompt[2]}]
    })

    contents.append({
        "role": "assistant",
        "parts": [{"text": prompt[3]}]
    })

    # Construct the parts for the final user prompt and images
    last_prompt_parts = []
    last_prompt_parts.append({"text": prompt[-1]})  # Add the final text prompt

    if image_pack is not None:
        for image_block in image_pack:
            image_data = {
                "mime_type": "image/jpeg",  # Or image/png, based on your image
                "data": image_block["source"]["data"]
            }
            last_prompt_parts.append({"inline_data": image_data})  # Add image as inline_data

    # Add the final user turn with the combined text and image parts
    contents.append({
        "role": "user",
        "parts": last_prompt_parts
    })

    model = genai.GenerativeModel('gemini-2.0-flash')  # Initialize model here
    try:
        response = model.generate_content(contents=contents)
    except exceptions.ResourceExhausted as e:
        print(f"Quota error: {e}")
        print("Retrying in 60 seconds")
        time.sleep(60)
        try:
            response = model.generate_content(contents=contents)
            # print(response.text)
        except exceptions.ResourceExhausted as e:
            print("Retry Failed, Quota still exceeded")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return response.text  # Access the text directly

def chat_generation_with_polish_urial(prompt, image_pack):
    contents = []

    # Add the first user turn
    contents.append({
        "role": "user",
        "parts": [{"text": prompt[0]}]
    })

    # Add the assistant's turn
    contents.append({
        "role": "assistant",
        "parts": [{"text": prompt[1]}]
    })

    contents.append({
        "role": "user",
        "parts": [{"text": prompt[2]}]
    })

    contents.append({
        "role": "assistant",
        "parts": [{"text": prompt[3]}]
    })

    # Construct the parts for the final user prompt and images
    last_prompt_parts = []
    last_prompt_parts.append({"text": prompt[-3]})  # Add the final text prompt

    for image_block in image_pack:
        image_data = {
            "mime_type": "image/jpeg",  # Or image/png, based on your image
            "data": image_block["source"]["data"]
        }
        last_prompt_parts.append({"inline_data": image_data})  # Add image as inline_data

    # Add the final user turn with the combined text and image parts
    contents.append({
        "role": "user",
        "parts": last_prompt_parts
    })

    contents.append({
        "role": "assistant",
        "parts": prompt[-2]
    })

    contents.append({
        "role": "user",
        "parts": prompt[-1]
    })

    model = genai.GenerativeModel('gemini-2.0-flash')  # Initialize model here
    try:
        response = model.generate_content(contents=contents)
    except exceptions.ResourceExhausted as e:
        print(f"Quota error: {e}")
        print("Retrying in 60 seconds")
        time.sleep(60)
        try:
            response = model.generate_content(contents=contents)
            # print(response.text)
        except exceptions.ResourceExhausted as e:
            print("Retry Failed, Quota still exceeded")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return response.text  # Access the text directly


def single_chat_generation(prompt):
    contents = []
    generation_config = genai.GenerationConfig(
        temperature=0,  # Set the temperature here
        top_p=0.1
    )
    model = genai.GenerativeModel('gemini-2.0-flash')  # Initialize model here
    contents.append({
        "role": "user",
        "parts": prompt
    })
    try:
        response = model.generate_content(contents=contents, generation_config=generation_config)
    except exceptions.ResourceExhausted as e:
        print(f"Quota error: {e}")
        print("Retrying in 60 seconds")
        time.sleep(60)
        try:
            response = model.generate_content(contents=contents, generation_config=generation_config)
            # print(response.text)
        except exceptions.ResourceExhausted as e:
            print("Retry Failed, Quota still exceeded")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return response.text
