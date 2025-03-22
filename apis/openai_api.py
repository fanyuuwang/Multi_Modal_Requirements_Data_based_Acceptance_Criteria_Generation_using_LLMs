from openai import OpenAI

api = "YOUR_API"
client = OpenAI(api_key=api)


def chat_generation_with_apeer(prompt, image_pack):
    image_blocks = image_pack
    last_prompt = []

    last_prompt.append({
        "type": "text",
        "text": prompt[-1]
    })
    if image_pack is not None:
        for image_block in image_blocks:
            new_image_block = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_block['source']['data']}"
                }
            }
            last_prompt.append(new_image_block)

    completion = client.chat.completions.create(
      model="gpt-4o",
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
    response = (completion.choices[0].message.content)

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
            new_image_block = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_block['source']['data']}"
                }
            }
            last_prompt.append(new_image_block)

    completion = client.chat.completions.create(
      model="gpt-4o",
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
    response = (completion.choices[0].message.content)

    return response

def chat_generation_with_polish_urial(prompt, image_pack):
    image_blocks = image_pack
    last_prompt = []
    last_prompt.append({
        "type": "text",
        "text": prompt[-3]
    })
    for image_block in image_blocks:
        new_image_block = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_block['source']['data']}"
            }
        }
        last_prompt.append(new_image_block)

    completion = client.chat.completions.create(
      model="gpt-4o",
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
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                      "text": prompt[-1]
                  }
              ]
          }
        ]
    )
    response = (completion.choices[0].message.content)

    return response

def single_chat_generation(prompt):
    contents = []
    contents.append({
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt
                  }
              ]
          },)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=contents,
        temperature=0,
        top_p=0.1,
    )
    response = (completion.choices[0].message.content)
    return response
