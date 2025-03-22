# Absolute Grading: Outputs score of 1 to 5
import json
import os
import sys
import random
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
from itertools import combinations
import networkx as nx
from collections import defaultdict
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT
import numpy as np
import torch
import numpy as np

def absolute_judge(user_story, acceptance_criteria, judge):
    instruction = "Given the software requirements (user story) and corresponding background knowledge, you are required to identify correct acceptance criteria like an expert in software engineering."
    response = f"Acceptance Criteria: {acceptance_criteria}"
    reference_answer = f"{user_story}",
    rubric_data = {
        "criteria": """A good acceptance criterion clearly and concisely states what needs to be true for a feature to be accepted. It should be:
        1.	Relevance: Ensure that the acceptance criteria align with the given user story, description, and constraints.
    	2.	Coverage: The acceptance criteria should comprehensively address all relevant aspects of the given information.
    	3.	Correctness: The acceptance criteria should be factually accurate and logically structured.
    	4.	Understandability: The generated acceptance criteria should be clear, concise, and free from redundancy.
    	5.	Feasibility: The acceptance criteria should be practical and executable with the available resources in the project setup.""",
        "score1_description": "Vague/Minimal: Criteria: Barely defined; statements like “The system should work properly.”Quality: Ambiguous and untestable. Provides minimal guidance.",
        "score2_description": "Basic: Criteria: Clearly written but lacks measurable details. Quality: Understandable but may lead to interpretation differences in testing since the “what” is clear yet the “how much” is missing.",
        "score3_description": "Satisfactory: Criteria: Clear, concise, and includes measurable outcomes (e.g., “User must be able to log in within 3 seconds under normal conditions.”).Quality: Testable and reduces ambiguity, ensuring consistency between teams.",
        "score4_description": "Comprehensive: Criteria: Includes measurable outcomes, edge cases, and conditions (e.g., “For 95% of logins under standard network conditions, the response time should not exceed 3 seconds. Error messages must be displayed for failed attempts.”).Quality: Thorough and robust, allowing detailed test case design and better risk mitigation.",
        "score5_description": "Exemplary/Best Practice: Criteria: Fully detailed with clear, measurable, and testable conditions. It covers positive paths, negative scenarios, performance benchmarks, usability factors, and even security aspects when relevant. Often expressed in a behavior-driven style (e.g., “Given a valid user, when they attempt to log in, then they should access the dashboard in less than 3 seconds, and invalid attempts should trigger a clear error message.”). Quality: Provides full transparency and serves as a solid foundation for automated acceptance testing, leaving little room for misinterpretation.",
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    feedback, score = judge.single_absolute_grade(
        instruction=instruction,
        response=response,
        rubric=score_rubric,
        reference_answer=reference_answer
    )

    # print("Feedback:", feedback)
    # print("Score:", score)

    return feedback, score


def relative_judge(user_story, acceptance_criterias, judge):
    data = {
        "instruction": "Given the software requirements (user story) and corresponding background knowledge, you are required to identify correct acceptance criteria like an expert in software engineering.",
        "response_A": acceptance_criterias[0],
        "response_B": acceptance_criterias[1],
        "reference_answer": f"User Story and Background Knowledge: {user_story}",
        "rubric": """Evaluation Rubric
  1.  Clarity
  • Questions to Consider:
  • Is the language simple, direct, and free of ambiguity?
  • Can every team member (developers, testers, product owners, etc.) understand the criteria without multiple interpretations?
  • Comments/Examples:
E.g., “The phrase ‘confirmation message’ is clear, but ‘immediate response’ may need further definition.”
  2.  Testability
  • Questions to Consider:
  • Can each criterion be directly translated into one or more test cases with clear pass/fail outcomes (e.g., using a Given/When/Then format)?
  • Is it evident how to verify that the criteria have been met?
  • Comments/Examples:
E.g., “The criteria are written in a way that testers can simulate scenarios to check outcomes.”
  3.  Completeness
  • Questions to Consider:
  • Do the acceptance criteria cover all aspects of the user story, including main flows, edge cases, and failure scenarios?
  • Is there any critical condition or business rule that is missing?
  • Comments/Examples:
E.g., “While positive scenarios are covered, the failure cases (e.g., invalid input) are not addressed.”
  4.  Consistency
  • Questions to Consider:
  • Are the acceptance criteria aligned with the user story and consistent with other related stories or business rules?
  • Is there any conflicting or overlapping information?
  • Comments/Examples:
E.g., “The criteria do not conflict with other documented requirements, ensuring a consistent definition of done.”
  5.  Feasibility
  • Questions to Consider:
  • Are the acceptance criteria realistic and achievable given current technical constraints and available resources?
  • Do they avoid setting requirements that are technically or operationally impractical?
  • Comments/Examples:
E.g., “The criteria are ambitious but achievable within the current project constraints.”
  6.  Value Orientation
  • Questions to Consider:
  • Do the acceptance criteria clearly articulate the value or benefit delivered to the end user or stakeholder?
  • Are they outcome-focused rather than just a set of technical specifications?
  • Comments/Examples:
E.g., “The criteria link well to user benefits, such as improved usability or performance enhancements.”"""
    }

    feedback, score = judge.single_relative_grade(**data)

    # print("Feedback:", feedback)
    # print("Score:", score)
    return feedback, score

def run_absoluty_judge(context, issues, judge):

    feedback, score = absolute_judge(context, issues, judge)

    return score, feedback


def compute_ranking(pairwise_results):
    """
    Compute ranking based on pairwise preferences using Copeland Score and Rank Centrality.

    :param pairwise_results: List of tuples (index_a, index_b, preferred_index)
    :return: Two rankings (Copeland Score ranking, Rank Centrality ranking)
    """
    # Extract unique candidates
    candidates = set()
    for a, b, _ in pairwise_results:
        candidates.update([a, b])
    candidates = sorted(candidates)  # Ensure consistent ordering

    # Initialize adjacency matrix for Rank Centrality
    n = len(candidates)
    index_map = {c: i for i, c in enumerate(candidates)}
    win_matrix = np.zeros((n, n))

    # Compute Copeland Score and win counts
    win_count = defaultdict(int)
    match_count = defaultdict(int)

    for a, b, winner in pairwise_results:
        if winner == a:
            win_count[a] += 1
        else:
            win_count[b] += 1
        match_count[a] += 1
        match_count[b] += 1

        # Update Rank Centrality matrix
        win_matrix[index_map[winner], index_map[a if winner == b else b]] += 1

    # Compute Copeland Score Ranking
    copeland_ranking = sorted(win_count.keys(), key=lambda x: -win_count[x])

    # Normalize the win matrix for Rank Centrality
    for i in range(n):
        total = win_matrix[i].sum()
        if total > 0:
            win_matrix[i] /= total

    # Use Markov Chain approach for Rank Centrality
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if win_matrix[i, j] > 0:
                G.add_edge(candidates[i], candidates[j], weight=win_matrix[i, j])

    rank_centrality_scores = nx.pagerank(G, alpha=0.85)
    rank_centrality_ranking = sorted(rank_centrality_scores.keys(), key=lambda x: -rank_centrality_scores[x])

    return copeland_ranking, rank_centrality_ranking

def run_relative_judge(context, issues, model):
    judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)
    total_prefers = []
    issue_indexs = [u for u in range(len(issues))]
    pairwise_issue_indexs = [list(connection) for connection in combinations(issue_indexs, 2)]
    for pair_issue in pairwise_issue_indexs:
        feedback, score = relative_judge(context, [issues[pair_issue[0]], issues[pair_issue[1]]], judge)
        if score == "A":
            total_prefers.append((pair_issue[0], pair_issue[1], pair_issue[0]))
        else:
            total_prefers.append((pair_issue[0], pair_issue[1], pair_issue[1]))

    rank1, rank2 = compute_ranking(total_prefers)
    print("Sorted Issue Index:", rank2)
    return None, rank2

def initiate_prometheus():
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    return judge
