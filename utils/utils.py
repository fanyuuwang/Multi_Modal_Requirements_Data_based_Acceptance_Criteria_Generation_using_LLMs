import re
import json
import requests
import glob
from bs4 import BeautifulSoup
import os
from json import JSONDecoder


# URL of the directory (ensure it ends with a slash)
def load_acceptance_criteria(file_path):
    acs = []
    uss = []
    use = []
    ids = []
    bg1 = []
    bg2 = []
    img = []
    with open(file_path, "r") as f:
        generated_acs = json.load(f)
    for single_data in generated_acs:
        acs.append(single_data["acceptanceCriteria"])
        uss.append(single_data["User_Story"])
        use.append(single_data["US_Description"])
        ids.append(single_data["Ground_ID"])
        bg1.append(single_data["BG"])
        bg2.append(single_data["Con"])
        img.append(single_data["Images"])

    return uss, uss, acs, ids, bg1, bg2, img

def clean_response(response):
    try:
        extracted_list = re.search(r'\[(.*?)\]', response, re.DOTALL).group(1).strip()
    except:
        try:
            extracted_list = re.search(r'\[(.*?)\n\n', response, re.DOTALL).group(1).strip()
        except:
            try:
                extracted_list = re.search(r'\n\n(.*?)\n\n', response, re.DOTALL).group(1).strip()
            except:
                try:
                    decoder = JSONDecoder()
                    pos = 0
                    while pos < len(response):
                        try:
                            json_obj, index = decoder.raw_decode(response, pos)
                            return json_obj
                        except json.JSONDecodeError:
                            pos += 1  # Move one character ahead and try again.
                # print("error")
                # print(response)
                except:
                    extracted_list = "****ERROR****" + response
    return extracted_list

def json_clean_response(response):
    try:
        extracted_list = re.search(r'\{(.*?)\}', response, re.DOTALL).group(1).strip()
    except:
        try:
            extracted_list = re.search(r'\{(.*?)\n\n', response, re.DOTALL).group(1).strip()
        except:
            try:
                extracted_list = re.search(r'\n\n(.*?)\n\n', response, re.DOTALL).group(1).strip()
            except:
                # print("error")
                # print(response)
                extracted_list = "****ERROR****" + response

    return extracted_list

def load_information(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    backgroundKnowledge = []
    for us in data:
        backgroundKnowledge.append(us["Background Knowledge"])

    return backgroundKnowledge

def get_images(page_url):
    """
    Returns a list of full image URLs found on the given webpage.

    Args:
        page_url (str): The URL of the webpage to scan for images.

    Returns:
        List[str]: A list of absolute image URLs.
    """
    image_urls = []

    try:
        response = requests.get(page_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")

    for img in img_tags:
        img_src = img.get("src")
        if not img_src:
            continue

        full_img_url = urljoin(page_url, img_src)
        image_urls.append(full_img_url)

    return image_urls

def get_image_html(file_path):
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))  # Get all .txt files
    file_names = []
    content_list = []

    for file_path in txt_files:
        file_names.append(os.path.basename(file_path))  # Extract file name
        with open(file_path, "r", encoding="utf-8") as file:
            content_list.append(file.read())  # Read file content

    return content_list
