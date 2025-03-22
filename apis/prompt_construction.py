import base64
import requests

import requests
import base64
from io import BytesIO
import imghdr


def read_images(images):
    """
    Reads images from GitHub repository and prepares them for Claude API

    Args:
        images: List of image names without extension

    Returns:
        List of dicts containing image content blocks ready for Claude API
    """
    image_content_blocks = []

    for image in images:
        # Construct the URL
        url = f"https://raw.githubusercontent.com/fanyuuwang/esolution/main/pages/{image}.jpg"

        # Fetch the image
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch image {image}: {response.status_code}")
            continue

        image_data = response.content

        # Auto-detect the actual image format
        img_format = imghdr.what(None, h=image_data)
        if img_format == 'jpeg':
            media_type = 'image/jpeg'
        elif img_format == 'png':
            media_type = 'image/png'
        elif img_format == 'gif':
            media_type = 'image/gif'
        elif img_format == 'webp':
            media_type = 'image/webp'
        else:
            media_type = 'image/jpeg'  # Default fallback

        # Encode to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Create content block in Claude's expected format
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image
            }
        }

        image_content_blocks.append(image_block)

    return image_content_blocks


def urial_construction(us, use, bg, images):
    examples = {
        "User Story": "As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic, so that I can quickly locate the most relevant content.",
        "User Story Extension": "The indicator will automatically update based on the current week functionality.",
        "Background Knowledge": "As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic within the dropdown menu",
        "Considering": "Please investigate options available to highlight the learning section (bold letters or using blue colour) see UI design screenshots. Note: (Current week functionality not required until teaching starts - Phase 2)",
        "Acceptance Criteria": {
            "ACriteria": [
                {
                    "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                    "When": "The date is within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                    "Then": "'Current Week' (or current Topic, current Module, or Current Day) is highlighted within the dropdown menu as per UI designs (attachment 1):\n* Include a Book icon to indicate the current week.\n* Investigate options to highlight the learning section (e.g., bold letters or blue color)."
                },
                {
                    "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                    "When": "The date is not within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                    "Then": "Do not highlight any Learning section (e.g., week) in the dropdown menu."
                }
            ]
        }
    }
    instruction = """You are a software requirements analyst.
Your task is to generate well-defined acceptance criteria from user stories.
Each user story comes with additional details such as:
- User Story Extension: Additional context for the feature.
- Background Knowledge: Important information related to the user story.
- Considering: Additional factors or constraints to take into account.

Please note that, the user story and its extension should be primarily relied, but the background knowledge, considering and images may have some irrelevant information. You can make use of it if there are related information provided in the background knowledge, considering and images. 

In the acceptance criteria generation, you need to consider the following factors:
- Relevance assesses how well the generated acceptance criteria aligns with the given information.
- Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information.
- Correctness evaluates the accuracy of acceptance criteria and its steps.
- Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.
- Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup.

The acceptance criteria should be structured in a Given-When-Then format and returned as a JSON object.
Ensure clarity, completeness, and testability of each criterion."""

    answer1 = "Please provide the user story along with any relevant extensions, background knowledge, and considerations. I’ll generate well-defined acceptance criteria in the Given-When-Then format and return them as a structured JSON object."

    query1 = f"""User Story:\n{examples['User Story']}
            \nUser Story Extension:\n{examples['User Story Extension']}
            \nBackground Knowledge:\n{examples['Background Knowledge']}
            \nConsidering:\n{examples['Considering']}"""

    answer2 = """"ACriteria": [
                    {
                        "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                        "When": "The date is within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                        "Then": "'Current Week' (or current Topic, current Module, or Current Day) is highlighted within the dropdown menu as per UI designs (attachment 1):\n* Include a Book icon to indicate the current week.\n* Investigate options to highlight the learning section (e.g., bold letters or blue color)."
                    },
                    {
                        "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                        "When": "The date is not within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                        "Then": "Do not highlight any Learning section (e.g., week) in the dropdown menu."
                    }
                ]"""

    query2 = f"""User Story:\n{us}
            \nUser Story Extension:\n{use}
            \nBackground Knowledge:\n{bg}
            \nScreenshots:\nThe corresponding screenshots of the UI are also included in the image file. Please refer the screenshots to generate your acceptance criteria.
            
            The response must be in JSON format with the following structure:

    {{
      "acceptanceCriteria": [
        {{
          "GIVEN": <string>,
          "WHEN": <string>,
          "THEN": <string>
        }},
        ...
      ]
    }}
    
    Your response:"""

    if images is not None:
        image_blocks = read_images(images)
        return [instruction, answer1, query1, answer2, query2], image_blocks
    return [instruction, answer1, query1, answer2, query2], None


def apeer_construction(us, use, bg, images):
    instruction = f"""You are an AI model trained to assist in software development by generating acceptance criteria from user stories and related background knowledge. Your task is to generate a well-structured JSON object containing acceptance criteria based on the provided input.
    
    You will be given the following background knowledge, please comprehensively understand these background knowledge to help us generate acceptance criteria based on user story.
    \nBackground Knowledge:\n{bg}"""

    answer1 = "Okay, please provide the user story and rubric in generation, I will generate acceptance criteria in the well-structured JSON format based on the provided input."

    query1 = f"""Using the detailed description provided below, please generate acceptance criteria that captures all user story narratives and the background information. Note that the description may include multiple user stories or multiple sets of acceptance criteria. Please ensure that every detail, including all user stories and supporting information, is incorporated into your final generation
    Your response will be evaluated based on the following rubrics:
	1.	Relevance: Ensure that the acceptance criteria align with the given user story, description, background knowledge, and constraints.
	2.	Coverage: The acceptance criteria should comprehensively address all relevant aspects of the given information.
	3.	Correctness: The acceptance criteria should be factually accurate and logically structured.
	4.	Understandability: The generated acceptance criteria should be clear, concise, and free from redundancy.
	5.	Feasibility: The acceptance criteria should be practical and executable with the available resources in the project setup.

Instructions:
	•	Ensure that each acceptance criterion is atomic (focused on a single, testable requirement).
	•	Avoid redundancy while ensuring comprehensive coverage.
	•	Use precise language so that software developers and testers can implement and verify it easily.
	•	Follow the given JSON format strictly to maintain consistency.
	•	Please note that, the user story and its extension should be primarily relied, but the background knowledge, considering and images may have some irrelevant information. You can make use of it if there are related information provided in the background knowledge, considering and images.  
    
    Here is my given information:

    ** The user story: {us}

    ** The user story description: {use}
    
    ** The corresponding screenshots of the UI are also included in the image file. Please refer the screenshots to generate your acceptance criteria
    
    The response must be in JSON format with the following structure:

    {{
      "acceptanceCriteria": [
        {{
          "GIVEN": <string>,
          "WHEN": <string>,
          "THEN": <string>
        }},
        ...
      ]
    }}
    
Please give your response:"""
    if images is not None:
        image_blocks = read_images(images)
        return [instruction, answer1, query1], image_blocks

    return [instruction, answer1, query1], None


def polish_urial_prompt_construction(user_story, user_story_extension, bg, images, ac_to_refine, all_ac):
    examples = {
        "User Story": "As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic, so that I can quickly locate the most relevant content.",
        "User Story Extension": "The indicator will automatically update based on the current week functionality.",
        "Background Knowledge": "As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic within the dropdown menu",
        "Considering": "Please investigate options available to highlight the learning section (bold letters or using blue colour) see UI design screenshots. Note: (Current week functionality not required until teaching starts - Phase 2)",
        "Acceptance Criteria": {
            "ACriteria": [
                {
                    "Given": "I am logged in as a learner.",
                    "When": "I access the course dashboard.",
                    "Then": "(1) I should see a button at the top-right labeled ‘Learning Support?’, along with an accompanying icon. (2) Clicking the button should navigate me to the ‘Support Resources’ section within the Help Center tab"
                },
                {
                    "Given": "I am logged in as a user with course viewing privileges (learner)",
                    "When": "I access the Course Dashboard.",
                    "Then": "(1) I should see a widget labeled ‘Your Learning Journey’. (2) The widget should have three learning modes — own-time, real-time and wrap-up. (3) The widget should show the current status of tasks completed and pending for each learning mode."
                }
            ]
        }
    }
    instruction = """You are a software requirements analyst.
    Your task is to generate well-defined acceptance criteria from user stories.
    Each user story comes with additional details such as:
    - User Story Extension: Additional context for the feature.
    - Background Knowledge: Important information related to the user story.
    - Considering: Additional factors or constraints to take into account.

    Please note that, the user story and its extension should be primarily relied, but the background knowledge, considering and images may have some irrelevant information. You can make use of it if there are related information provided in the background knowledge, considering and images. 

    In the acceptance criteria generation, you need to consider the following factors:
    - Relevance assesses how well the generated acceptance criteria aligns with the given information.
    - Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information.
    - Correctness evaluates the accuracy of acceptance criteria and its steps.
    - Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.
    - Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup.

    The acceptance criteria should be structured in a Given-When-Then format and returned as a JSON object.
    Ensure clarity, completeness, and testability of each criterion."""

    answer1 = "Please provide the user story along with any relevant extensions, background knowledge, and considerations. I’ll generate well-defined acceptance criteria in the Given-When-Then format and return them as a structured JSON object."

    query1 = f"""User Story:\n{examples['User Story']}
                \nUser Story Extension:\n{examples['User Story Extension']}
                \nBackground Knowledge:\n{examples['Background Knowledge']}"""

    answer2 = """"ACriteria": [
                        {
                            "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                            "When": "The date is within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                            "Then": "'Current Week' (or current Topic, current Module, or Current Day) is highlighted within the dropdown menu as per UI designs (attachment 1):\n* Include a Book icon to indicate the current week.\n* Investigate options to highlight the learning section (e.g., bold letters or blue color)."
                        },
                        {
                            "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                            "When": "The date is not within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                            "Then": "Do not highlight any Learning section (e.g., week) in the dropdown menu."
                        }
                    ]"""

    query2 = f"""User Story:\n{user_story}
                \nUser Story Extension:\n{user_story_extension}
                \nBackground Knowledge:\n{bg}
                \nScreenshots:\nThe corresponding screenshots of the system are also included in the image file.

                The response must be in JSON format with the following structure:

        {{
          "acceptanceCriteria": [
            {{
              "GIVEN": <string>,
              "WHEN": <string>,
              "THEN": <string>
            }},
            ...
          ]
        }}

        Your response:"""

    answer3 = f"{all_ac}"

    prompt = (
        "There is one specific acceptance criteria, which is believed as not completely related to the other given information or missed some significant points. Your task is to refine this specific acceptance criteria to find if there are some points missed and include the missed points in the revised version. But note that you should primarily base on the user story and its extension.\n\n"
        "The acceptance criteria to be refined:\n"
        f"{ac_to_refine}\n\n"
    )

    prompt += (
        "Instructions:\n"
        "1. Based on the historical context provided, revise the acceptance criteria to ensure it meets the standards of a high-quality set of acceptance criteria, including relevance, coverage, correctness, understandability, and feasibility.\n"
        "2. Ensure that your revised acceptance criteria are unique and do not duplicate or closely resemble any of the other existing acceptance criteria.\n"
        "3. Reevaluate the test objectives within this acceptance criteria. If the test objective is accurate, you may retain its current phrasing.\n"
        "4. Do not introduce any new terminology outside the scope of the provided information. Maintain alignment with the overall user story and any domain-specific language.\n"
        """The response must be in JSON format with the following structure without any other explanation:

    {{
      "acceptanceCriteria": [
        {{
          "GIVEN": <string>,
          "WHEN": <string>,
          "THEN": <string>
        }}
        
        Your response: """
    )

    query3 = prompt

    if images is not None:
        image_blocks = read_images(images)
        return [instruction, answer1, query1, answer2, query2, answer3, query3], image_blocks
    return [instruction, answer1, query1, answer2, query2, answer3, query3], None

def ablation_prompt_construction(us, use):

    instruction = f"""You are an AI model trained to assist in software development by generating acceptance criteria from user stories. Your task is to generate a well-structured JSON object containing acceptance criteria based on the provided input."""

    answer1 = "Okay, please provide the user story and rubric in generation, I will generate acceptance criteria in the well-structured JSON format based on the provided input."

    query1 = f"""Using the detailed description provided below, please generate acceptance criteria that captures all user story narratives. Note that the description may include multiple user stories or multiple sets of acceptance criteria. Please ensure that every detail, including all user stories and supporting information, is incorporated into your final generation
        
        Your response will be evaluated based on the following rubrics:
    	1.	Relevance: Ensure that the acceptance criteria align with the given user story, description, and constraints.
    	2.	Coverage: The acceptance criteria should comprehensively address all relevant aspects of the given information.
    	3.	Correctness: The acceptance criteria should be factually accurate and logically structured.
    	4.	Understandability: The generated acceptance criteria should be clear, concise, and free from redundancy.
    	5.	Feasibility: The acceptance criteria should be practical and executable with the available resources in the project setup.

    Instructions:
    	•	Ensure that each acceptance criterion is atomic (focused on a single, testable requirement).
    	•	Avoid redundancy while ensuring comprehensive coverage.
    	•	Use precise language so that software developers and testers can implement and verify it easily.
    	•	Follow the given JSON format strictly to maintain consistency.
    	•	Please note that, the user story and its extension should be primarily relied.

        Here is my given information:

        ** The user story: {us}
        
        ** The user story extension: {use}

        ** The corresponding screenshots of the system are also included in the image file.

        The response must be in JSON format with the following structure:

        {{
          "acceptanceCriteria": [
            {{
              "GIVEN": <string>,
              "WHEN": <string>,
              "THEN": <string>
            }},
            ...
          ]
        }}

    Please give your response:"""

    return [instruction, answer1, query1], None


def urial_ablation_construction(us, use, bg):
    examples = {
        "User Story": "As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic, so that I can quickly locate the most relevant content.",
        "User Story Extension": "The indicator will automatically update based on the current week functionality.",
        "Background Knowledge": "As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic within the dropdown menu",
        "Considering": "Please investigate options available to highlight the learning section (bold letters or using blue colour) see UI design screenshots. Note: (Current week functionality not required until teaching starts - Phase 2)",
        "Acceptance Criteria": {
            "ACriteria": [
                {
                    "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                    "When": "The date is within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                    "Then": "'Current Week' (or current Topic, current Module, or Current Day) is highlighted within the dropdown menu as per UI designs (attachment 1):\n* Include a Book icon to indicate the current week.\n* Investigate options to highlight the learning section (e.g., bold letters or blue color)."
                },
                {
                    "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                    "When": "The date is not within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                    "Then": "Do not highlight any Learning section (e.g., week) in the dropdown menu."
                }
            ]
        }
    }
    instruction = """You are a software requirements analyst.
Your task is to generate well-defined acceptance criteria from user stories.
Each user story comes with additional details such as:
- User Story Extension: Additional context for the feature.
- Background Knowledge: Important information related to the user story.
- Considering: Additional factors or constraints to take into account.

Please note that, the user story and its extension should be primarily relied, but the background knowledge, considering and images may have some irrelevant information. You can make use of it if there are related information provided in the background knowledge, considering and images. 

In the acceptance criteria generation, you need to consider the following factors:
- Relevance assesses how well the generated acceptance criteria aligns with the given information.
- Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information.
- Correctness evaluates the accuracy of acceptance criteria and its steps.
- Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.
- Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup.

The acceptance criteria should be structured in a Given-When-Then format and returned as a JSON object.
Ensure clarity, completeness, and testability of each criterion."""

    answer1 = "Please provide the user story along with any relevant extensions, background knowledge, and considerations. I’ll generate well-defined acceptance criteria in the Given-When-Then format and return them as a structured JSON object."

    query1 = f"""User Story:\n{examples['User Story']}
            \nUser Story Extension:\n{examples['User Story Extension']}
            \nBackground Knowledge:\n{examples['Background Knowledge']}
            \nConsidering:\n{examples['Considering']}"""

    answer2 = """"ACriteria": [
                    {
                        "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                        "When": "The date is within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                        "Then": "'Current Week' (or current Topic, current Module, or Current Day) is highlighted within the dropdown menu as per UI designs (attachment 1):\n* Include a Book icon to indicate the current week.\n* Investigate options to highlight the learning section (e.g., bold letters or blue color)."
                    },
                    {
                        "Given": "I am logged in as a user with a role that allows course creation, course editing, and course viewing (refer to roles matrix).",
                        "When": "The date is not within the Learning section date range, I am either in edit mode or not in edit mode, and I click on the Learning section tab to view the Learning section.",
                        "Then": "Do not highlight any Learning section (e.g., week) in the dropdown menu."
                    }
                ]"""

    query2 = f"""User Story:\n{us}
            \nUser Story Extension:\n{use}
            \nBackground Knowledge:\n{bg}
            \nScreenshots:\nThe corresponding screenshots of the system are also included in the image file.

            The response must be in JSON format with the following structure:

    {{
      "acceptanceCriteria": [
        {{
          "GIVEN": <string>,
          "WHEN": <string>,
          "THEN": <string>
        }},
        ...
      ]
    }}

    Your response:"""

    return [instruction, answer1, query1, answer2, query2], None
