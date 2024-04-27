import os
import json
from openai import OpenAI
client = OpenAI()

EVALUATION_PROMPT = """
Your job is to evaluate the performance of an AI-powered question answering system. You will be given a query, a ground truth answer, and the answer given by the AI. Your task is to grade the AI's answer on a scale of 0-10. A score of 0 means the AI's answer is completely wrong. A score of 10 means the AI's answer is completely correct. A score of 5 means the AI's answer is partially correct.

Your response must ONLY be an integer between 0 and 10 (inclusive). Do not include any other text in your response.

GUIDELINES FOR GRADING
- The ground truth answers are often lacking in detail, so if the AI's answer is more detailed than the ground truth answer, then that's generally a good sign.
- Be wary of overly broad or general AI answers. If the AI's answer lacks specifics, then it probably isn't a good answer.
- If a grading rubric is included in the GRADING RUBRIC section, then pay close attention to it. The rubric will tell you specific things to look for in the AI's answer.

QUERY
{query}

GROUND TRUTH ANSWER
{ground_truth_answer}

GRADING RUBRIC
{rubric}

AI-GENERATED ANSWER
{model_answer}

GRADE
""".strip()

# we'll use this to run the evaluation prompt
def openai_api_call(chat_messages: list[dict], model_name: str = "gpt-4-1106-preview") -> str:
    assert model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-1106-preview"]

    # call the OpenAI API
    client.api_key = os.getenv("OPENAI_API_KEY")
    response = client.chat.completions.create(
        model=model_name,
        messages=chat_messages,
        max_tokens=int(1),
        temperature=float(0.0),
    )
    llm_output = response.choices[0].message.content.strip()
    return llm_output

# evaluate the model's predictions against the ground truth answers
def evaluate_response(query, gt_answer, rubric, model_answer):
    prompt = EVALUATION_PROMPT.format(query=query, ground_truth_answer=gt_answer, rubric=rubric, model_answer=model_answer)
    chat_messages = [{"role": "user", "content": prompt}]
    response = openai_api_call(chat_messages, model_name="gpt-4-1106-preview")
    return response

def run_evaluation(eval_set: list[dict]):
    """
    - eval_set is a list of dictionaries, where each dictionary contains:
        - "query": the query
        - "gt_answer": the ground truth answer, i.e. what you want the model to output
        - "rubric": additional notes for the evaluator
        - "model_response": the model's response to the query
    """
    # run the evaluation
    eval_results = []
    for eval_item in eval_set:
        query = eval_item["query"]
        gt_answer = eval_item["gt_answer"]
        rubric = eval_item["rubric"]
        model_answer = eval_item["model_answer"]

        grade = evaluate_response(query, gt_answer, rubric, model_answer)
        print (query)
        print (grade)
        print ("")

        # save the results
        eval_results.append({
            "query": query,
            "gt_answer": gt_answer,
            "rubric": rubric,
            "model_answer": model_answer,
            "grade": grade,
        })

    # export eval results to a json file
    with open(f"eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

# load in the eval set from a json file
with open("eval_set_w_responses.json", "r") as f:
    eval_set = json.load(f)

# run the evaluation
run_evaluation(eval_set)