import copy
import json
import os
import sys
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
import pandas as pd
import json
import apis.gemini_api as api
from apis.prompt_construction import *
from reward_models.prometheus import *
from reward_models.generative_verifier import *
from reward_models.ur3 import *
from RAG.similarity_rag import *
from RAG.icRALM import *
from RAG.vrag import *
from RAG.html_rag import *
from utils.utils import *


def polish_acs_gen(file_path):
    model = initiate_gen_veri()
    prom = initiate_prometheus()
    with open(file_path, 'r') as f:
        data = json.load(f)
    uss, use, acs, ids, bg1, bg2, images = load_acceptance_criteria(file_path)
    for case_num, (ustory, ustory_ext, acriteria) in enumerate(zip(uss, use, acs)):
        last_total_score = 0
        polish_count = 0
        polished_acs = []
        polished_scores = []
        while True:
            story = '\n'.join(ustory)
            story_ext = '\n'.join(ustory_ext)
            prompt = f"""User Story: {story}

                                    User Story Description: {story_ext}"""
            total_score = run_absoluty_judge(prompt, acriteria, prom)
            # total_score = model.gen_verifier(["\n".join(ustory), "\n".join(ustory_ext)], acriteria)
            print(f"{ids[case_num]} Score: ", total_score)
            if total_score < last_total_score or polish_count > 3:
                acriteria = polished_acs[polished_scores.index(max(polished_scores))]
                break
            if total_score == 5:
                break
            else:
                polished_scores.append(total_score)
                print(f"{ids[case_num]} Polishing... Current Iteration: {polish_count}")
                polish_count += 1
                case_score = []
                for ac in acriteria:
                    predict_value = model.gen_verifier([story, story_ext], ac)
                    case_score.append(predict_value)
                refine_ac = acriteria[case_score.index(min(case_score))]
                other_ac = copy.deepcopy(acriteria)
                other_ac.remove(refine_ac)
                polish_prompt, image_pack = polish_urial_prompt_construction(story, story_ext, bg1[case_num], bg2[case_num], images[case_num], refine_ac, other_ac, acriteria)
                ori_response = api.chat_generation_with_polish_urial(polish_prompt, image_pack)
                try:
                    response = clean_response(ori_response)
                    response = json.loads(f'[{response.strip()}]')
                    acriteria = other_ac + response
                    polished_acs.append(acriteria)
                except:
                    print("Error in polishing response:", ids[case_num])
                    print(refine_ac)
                    print(ori_response)
                    break
        data[case_num]["acceptanceCriteria"] = acriteria
        with open(file_path.replace(".json", "_final2_polished_prome+gen.json"), 'w') as f:
            json.dump(data, f)


def polish_acs_ur3(file_path):
    ur3_model = initiate_ur3_reranker()
    prom = initiate_prometheus()
    with open(file_path, 'r') as f:
        data = json.load(f)
    uss, use, acs, ids, bg1, bg2, images = load_acceptance_criteria(file_path)
    for case_num, (ustory, ustory_ext, acriteria) in enumerate(zip(uss, use, acs)):
        last_total_score = 0
        polish_count = 0
        polished_acs = []
        polished_scores = []
        while True:
            story = '\n'.join(ustory)
            story_ext = '\n'.join(ustory_ext)
            prompt = f"""User Story: {story}

                                        User Story Description: {story_ext}"""
            total_score, feedback = run_absoluty_judge(prompt, acriteria, prom)
            print(feedback)
            # total_score = model.gen_verifier(["\n".join(ustory), "\n".join(ustory_ext)], acriteria)
            print(f"{ids[case_num]} Score: ", total_score)
            if total_score < last_total_score or polish_count > 3:
                acriteria = polished_acs[polished_scores.index(max(polished_scores))]
                break
            if total_score == 5:
                break
            else:
                polished_scores.append(total_score)
                print(f"{ids[case_num]} Polishing... Current Iteration: {polish_count}")
                polish_count += 1
                case_score = []
                for ac in acriteria:
                    predict_value = ur3_model.rerank([story, story_ext], ac)
                    case_score.append(predict_value)
                refine_ac = acriteria[case_score.index(max(case_score))]
                print(refine_ac)
                other_ac = copy.deepcopy(acriteria)
                other_ac.remove(refine_ac)
                polish_prompt, image_pack = polish_urial_prompt_construction(story, story_ext, bg1[case_num], bg2[case_num], images[case_num], refine_ac, other_ac, acriteria)
                ori_response = api.chat_generation_with_polish_urial(polish_prompt, image_pack)
                print(ori_response)
                try:
                    response = clean_response(ori_response)
                    response = json.loads(f'[{response.strip()}]')
                    acriteria = other_ac + response
                    polished_acs.append(acriteria)
                except:
                    print("Error in polishing response:", ids[case_num])
                    print(refine_ac)
                    print(ori_response)
                    break
        data[case_num]["acceptanceCriteria"] = acriteria
        with open(file_path.replace(".json", "_final2_polished_prome+ur3.json"), 'w') as f:
            json.dump(data, f)


def retrieval_with_vrag(user_story, images, topk):
    model = initiate_vrag(images)
    selected_pics = model.query(user_story, topk)

    return selected_pics


def retrieval_with_icralm(input_information, ground_knowledge, topk=2):
    model = initiate_TRAG(ground_knowledge)
    bg_results, bg_labels = model.query(input_information, topk)

    return bg_results


def retrieval_with_similarity(input_information, ground_knowledge, topk=2):
    model = initiate_simiRAG(ground_knowledge)
    bg_results, bg_labels = model.query(input_information, topk)

    return bg_results


def retrieval_html(input_information, html, topk=2):
    model = initiate_html(html)
    html_results, html_labels = model.query(input_information, topk)
    return html_results


def do_rag(user_story, us_description, args):
    """
    args.story_path, bg_path, html_path, image_dir
    """
    total_rag_results = []
    ground_knowledges = load_bg(args.bg_path)
    html = get_image_html(args.html_path)
    # TRAG with ICRALM
    ic_background = retrieval_with_icralm([user_story, us_description], [ground_knowledges], args.text_top_k)
    bg_results = ic_background

    # TRAG with SentBERT with Similarity
    # simi_background = retrieval_with_similarity([user_story, us_description], [ground_knowledges], args.text_top_k)
    # bg_results = simi_background

    # VRAG with HTML (You can use the simplified HTML directory to initialize this method)
    html_screen_shots = retrieval_html(
        [user_story, us_description],
        html, args.image_top_k)

    image_labels = [(x.replace(".jpg", "").replace(args.image_dir, "")) for x in html_screen_shots]
    
    # VRAG with DSE
    # dse_screen_shots = retrieval_with_vrag([user_story, us_description], images, topk=args.image_top_k)
    # image_labels = [(x.replace(".jpg", "").replace(args.image_dir, "")) for x in dse_screen_shots]
    single_result = {
        "BG": bg_labels,
        "Images": image_labels,
        "US": user_story,
        "US_extension": us_description
    }
    return single_result


def generate_ac(single_result):
    prompt, image_pack = apeer_construction(single_data["US"], single_data["US_extension"],
                                            single_data["BG"], single_data["Images"])
    response = api.chat_generation_with_apeer(prompt, image_pack)
    response = clean_response(response)
    response = json.loads(f'[{response.strip()}]')
    single_data["acceptanceCriteria"] = response
    return single_data


def main(user_story, user_story_extension):
    args = None
    rag_result = do_rag(user_story, user_story_extension, args)
    generated_result = generate_ac(rag_result)
    final_result = polish_acs_ur3(generated_result)


if __name__ == '__main__':
    main()
