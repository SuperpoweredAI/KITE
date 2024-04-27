from superpowered import create_chat_thread, get_chat_response

def get_response_sp_chat(query: str, config: dict):
    """
    Get response from the Superpowered Chat endpoint
    - config is a dictionary containing:
        - "use_rse": whether to use the RSE or not
        - "kb_id": the ID of the knowledge base to use
        - "system_message": the system message to use
        - "model_name": the name of the model to use (e.g. "gpt-4")
    """
    use_rse = config["use_rse"]
    kb_id = config["kb_id"]
    system_message = config["system_message"]
    model_name = config["model_name"]
    
    thread_id = create_chat_thread([kb_id], use_rse=use_rse, system_message=system_message, model=model_name)["id"] # create a chat thread with the provided config parameters
    chat_response = get_chat_response(thread_id, query)
    chat_response = chat_response["interaction"]["model_response"]["content"]
    return chat_response

# define the config
SYSTEM_MESSAGE = """
You are an AI company assistant for a startup called Sourcegraph. You have been paired with a search system that will provide you with relevant information from the company handbook to help you answer user questions. You will see the results of these searches below. Since this is the only information about the company you have access to, if the necessary information to answer the user's question is not contained there, you should tell the user you don't know the answer. You should NEVER make things up just try to provide an answer.
""".strip()

config = {
    "use_rse": True,
    "kb_id": "8606a749-80e1-49e2-84b9-368c11d980e9", #"INSERT_KB_ID_HERE",
    "system_message": SYSTEM_MESSAGE,
    "model_name": "gpt-3.5-turbo",
}

# load in the eval set
#from supreme_court_opinions import eval_set
from sourcegraph import eval_set

# get the model's response for each query in the eval set and put it in the eval set dictionary
for eval_item in eval_set:
    query = eval_item["query"]
    model_answer = get_response_sp_chat(query, config)
    eval_item["model_answer"] = model_answer

# save eval_set to json (but don't overwrite the original eval_set.json file)
import json
with open("eval_set_w_responses.json", "w") as f:
    json.dump(eval_set, f, indent=4)