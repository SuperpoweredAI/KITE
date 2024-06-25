from anthropic import Anthropic
import os

PERSONA_TASK_PROMPT = """
I'm building an evaluation set for a retrieval augmented generation (RAG) system. The first step is to generate a few realistic personas and tasks for each dataset. Each persona should have a specific task associated with it. In other words, the task should be something that the persona would want to accomplish by using this RAG system connected to this specific dataset.

Dataset: {dataset_name}
{dataset_description}

Please generate 10 unique persona/task combinations for this dataset. Each persona should be a short paragraph (3-5 sentences) describing a fictional person, and each task should be a short paragraph (3-5 sentences) describing a specific goal or task that the persona wants to accomplish using the RAG system connected to this dataset.
""".strip()

tmp = "The task should be something the user would do on a regular basis with the RAG system, not a one-off task."

SAMPLE_GENERATION_INSTRUCTIONS = """

""".strip()

def make_llm_call(chat_messages: list[dict], model) -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system_message = ""
    num_system_messages = 0
    normal_chat_messages = []
    for message in chat_messages:
        if message["role"] == "system":
            if num_system_messages > 0:
                raise ValueError("ERROR: more than one system message detected")
            system_message = message["content"]
            num_system_messages += 1
        else:
            normal_chat_messages.append(message)

    response = client.messages.create(
        system=system_message,
        messages=normal_chat_messages,
        model=model,
        max_tokens=4000,
        temperature=0.7,
    )
    return response.content[0].text


#dataset_name = "AI Papers"
#dataset_description = "A collection of ~100 research papers published on arXiv, mostly from the last two years, focused on LLMs and RAG."

#dataset_name = "BVP Cloud Index 10-Ks"
#dataset_description = "A collection of the most recent 10-K filings for each of the companies in the BVP Cloud Index, a stock index that tracks the performance of public cloud companies."

#dataset_name = "Sourcegraph Company Handbook"
#dataset_description = "The Sourcegraph Company Handbook, a public document that outlines the company's policies, procedures, and culture."

#dataset_name = "Supreme Court Opinions 2023"
#dataset_description = "All Supreme Court opinions from the year 2023."

dataset_name = ""
dataset_description = ""

model = "claude-3-5-sonnet-20240620"
prompt = PERSONA_TASK_PROMPT.format(dataset_name=dataset_name, dataset_description=dataset_description)
chat_messages = [{"role": "user", "content": prompt}]

llm_output = make_llm_call(chat_messages=chat_messages, model=model)
print (llm_output)