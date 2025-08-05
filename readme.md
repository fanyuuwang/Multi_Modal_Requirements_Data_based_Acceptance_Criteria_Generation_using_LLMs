**Annomynous Repo of submission "Multi-Modal-Requirements-Data-based-Acceptance-Criteria-Generation-using-LLMs"**

*Introduction*

ROOT
- source: main python file
- utils: some information extraction functions
- RAG: visual and textual RAG methods
- reward_models: global and local reward models
- apis: the LLMs' api, prompt construction
- Testset: The anonymized testset from the project
  	- Screenshots are the UI/UX designs
  	- anonymized_esolutions are the user story and acceptance testing data

**Excecuting Configuration**

We provided our example in source/main.py. You can specify your user story in this file to generate your acceptance criteria.

**Apeer Prompt Template**

User:
You are an AI model trained to assist in software development by generating acceptance criteria from user stories and related background knowledge. Your task is to generate a well-structured JSON object containing acceptance criteria based on the provided input.

You will be given the following background knowledge, please comprehensively understand this background knowledge to help us generate acceptance criteria based on user story.

Background Knowledge:
{INPUT BACKGROUND}

Considering:
{INPUT CONSIDERING}

⸻

Assistant:
Okay, please provide the user story and rubric in generation, I will generate acceptance criteria in the well-structured JSON format based on the provided input.

⸻

User:
Using the detailed description provided below, please generate acceptance criteria that capture all user story narratives and the background information. Note that the description may include multiple user stories or multiple sets of acceptance criteria. Please ensure that every detail, including all user stories and supporting information, is incorporated into your final generation.

Your response will be evaluated based on the following rubrics:
  - Relevance assesses how well the generated acceptance criteria aligns with the given information.
  - Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information.
  - Correctness evaluates the accuracy of acceptance criteria and its steps.
  - Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.
  - Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup.

Here are some instructions for you to follow:
	1.	Ensure that each acceptance criterion is atomic (focused on a single, testable requirement).
	2.	Avoid redundancy while ensuring comprehensive coverage.
	3.	Use precise language so that software developers and testers can implement and verify it easily.
	4.	Follow the given JSON format strictly to maintain consistency.
	5.	Please note that the user story and its extension should be primarily relied on, but the background knowledge, consideration, and images may have some irrelevant information. You can make use of it if there is related information provided in the background knowledge, considering, and images.

Here is my given information:
	•	The user story: {INPUT USER STORY}
	•	The user story description: {INPUT STORY DESCRIPTION}
	•	The corresponding screenshots of the system are also included in the image file.

**URIAL Prompt Template**

User:
You are a software requirements analyst.
Your task is to generate well-defined acceptance criteria from user stories.
Each user story comes with additional details such as:
- User Story Extension: Additional context for the feature.
- Background Knowledge: Important information related to the user story.

Please note that, the user story and its extension should be primarily relied, but the background knowledge, considering and images may have some irrelevant information. You can make use of it if there are related information provided in the background knowledge and images. 

In the acceptance criteria generation, you need to consider the following factors:
  - Relevance assesses how well the generated acceptance criteria aligns with the given information.
  - Coverage evaluates the ‘completeness’ of the generated acceptance criteria, i.e., the extent to which the acceptance criteria encompasses all relevant aspects of the given information.
  - Correctness evaluates the accuracy of acceptance criteria and its steps.
  - Understandability assesses whether the generated acceptance criteria is clear and comprehensible to the experts, without any redundancies.
  - Feasibility evaluates whether the acceptance criteria can be executed with the available resources in the project setup.

The acceptance criteria should be structured in a Given-When-Then format and returned as a JSON object.
Ensure clarity, completeness, and testability of each criterion.

⸻

Assistant:
Please provide the user story along with any relevant extensions, background knowledge, and considerations. I’ll generate well-defined acceptance criteria in the Given-When-Then format and return them as a structured JSON object.

⸻

User:
User Story: As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic, so that I can quickly locate the most relevant content.
User Story Extension: The indicator will automatically update based on the current week functionality.
Background Knowledge: As an academic or a student, when I click on learning section tab to view learning sections I want to have a visual indicator to highlight the current week/ topic within the dropdown menu.

⸻

Assistant:
ACriteria": [
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

⸻

User:
User Story: {INPUT USER STORY}
User Story Extension: {INPUT USER STORY EXTENSION}
Background Knowledge: {INPUT BACKGROUND KNOWLEDGE}
Screenshots: The corresponding screenshots of the UI are also included in the image file. Please refer the screenshots to generate your acceptance criteria.
            
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

**Polishing Prompt Template**

{USE PREVIOUS CHAT HISTORY}

User:
There is one specific acceptance criteria, which is believed as not completely related to the other given information or missed some significant points. Your task is to refine this specific acceptance criteria to find if there are some points missed and include the missed points in the revised version. But note that you should primarily base on the user story and its extension.
The acceptance criteria to be refined: {INPUT THE AC NEEDED POLISHING}

Instructions:
  1. Based on the historical context provided, revise the acceptance criteria to ensure it meets the standards of a high-quality set of acceptance criteria, including relevance, coverage, correctness, understandability, and feasibility.
  2. Ensure that your revised acceptance criteria are unique and do not duplicate or closely resemble any of the other existing acceptance criteria.
  3. Reevaluate the test objectives within this acceptance criteria. If the test objective is accurate, you may retain its current phrasing.
  4. Do not introduce any new terminology outside the scope of the provided information. Maintain alignment with the overall user story and any domain-specific language.
The response must be in JSON format with the following structure without any other explanation:

    {{
      "acceptanceCriteria": [
        {{
          "GIVEN": <string>,
          "WHEN": <string>,
          "THEN": <string>
        }}

**LLMs as a Judge Prompt Template**

User:
Given the following acceptance criteria and reference information, please determine whether ac2 effectively covers the point established in ac1.
User Story: {INPUT USER STORY}
User Story Description: {INPUT USER STORY EXTENSION}
ac1 (Ground Truth):
{INPUT AC1}
ac2 (Set of Acceptance Criteria):
{INPUT AC2}
Task:
  - Understand and identify the testing objectives in ac1 in the context of the user story and description.
  - Analyze the testing objectives provided in ac2 (multiple acceptance criteria in ac2) and determine if they cover the testing objectives specified in ac1.

You don't need to provide any explanation in your answer. Please choose your answer from the following options for the question 'Does ac2 cover the testing objectives in ac1?':
  - Yes, the test objectives in ac1 are fully covered in ac2.
  - Partially, the test objectives in ac1 are mentioned in ac2 but not fully covered.
  - No, ac2 failed to fully cover any testing objectives in ac1.
Your answer:

**Pairwise Comparison Prompt Template**

User:
You are given a user story, its description, and two sets of acceptance criteria.
Your task is to evaluate which acceptance criteria set is better based on the following rubric:
- Relevance: How well the acceptance criteria align with the given information.
- Coverage: The completeness of the acceptance criteria, i.e., whether they encompass all relevant aspects of the given information.
- Correctness: The accuracy of the acceptance criteria and their steps.
- Understandability: The clarity and comprehensibility of the acceptance criteria, ensuring no redundancies.
- Feasibility: Whether the acceptance criteria can be executed with the available resources in the project setup.
User Story: {INPUT USER STORY}
User Story Description: {INPUT USER STORY EXTENSION}
Acceptance Criteria 1:
{INPUT AC1}
Acceptance Criteria 2:
{INPUT AC2}
Based on the above, respond with only one word:
• 'Acceptance 1' if Acceptance Criteria 1 is better based on the rubric.
• 'Acceptance 2' if Acceptance Criteria 2 is better based on the rubric.
