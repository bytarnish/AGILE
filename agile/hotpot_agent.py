import json
import re
import torch
import torch.nn.functional as F
from generate import PlayGround
import os
from utils import embed_model, call_search_azure

model_file = "checkpoints/hotpotqa/agile"
playground = PlayGround(
    ckpt_path=model_file, 
    config_path=os.path.join(model_file, "config.json"),
    tokenizer_path=model_file
)

summary_file = "checkpoints/hotpotqa/summary"
summary_playground = PlayGround(
    ckpt_path=summary_file, 
    config_path=os.path.join(summary_file, "config.json"),
    tokenizer_path=summary_file,
    model_num=1
)

def text2emb(text):
    results = embed_model([text])
    return results[0]

def search_memory(param, lib_emb):
    param_emb = text2emb(param)
    lib = [i[0] for i in lib_emb]
    lib_embs = [i[1] for i in lib_emb]
    dis = F.cosine_similarity(torch.tensor(lib_embs), torch.tensor(param_emb).unsqueeze(0), dim=1).cpu().numpy().tolist() # [len(mem)]
    idx = range(len(dis))
    dis = list(zip(idx, dis))
    dis = sorted(dis, key=lambda x: x[1], reverse=True)
    data_sort = [lib[x[0]] for x in dis]
    return data_sort[0]

answer_pattern = r"Action: \[(.*?)\] \((.*?)\)"
def get_action(res):
    matches = re.findall(answer_pattern, res, re.DOTALL)
    if len(matches) < 1:
        return None, "", ""
    prefix = res.split("[{}] ({})".format(matches[0][0], matches[0][1]))[0]
    if matches[0][0].lower() == "search":
        return "Search", matches[0][1], prefix
    elif matches[0][0].lower() == "predictanswer":
        return "PredictAnswer", matches[0][1], prefix
    else:
        return "SeekAdvice", "", prefix

prompt = """You are an intelligent agent with the ability to search knowledge. Please answer the following questions.

You can analyze the solution steps based on the problem and known information. 
For missing information, you can use search tools by output `[Search] ([entity])`. If there is enough information, you can output `[PredictAnser] ([answer])` to answer the question directly or output `[Seekadvice] ()` if you are not sure and need to seek advice. Please note that the answer must be the span in the observation sentences.

[Question]: {}

Thought1:"""

gpt4_eval_prompt = """Based on the provided question and reference answer, please determine if the response is correct or incorrect. Begin by articulating your rationale, and conclude with a single word judgment: 'Yes' for correct or 'No' for incorrect.

question: {question}
reference answer: {reference}
response: {response}"""

test_file = "data/hotpotqa/test/data.jsonl"
with open(test_file) as f, open("results/hotpotqa/agile-vic13b-ppo.jsonl", "w") as f1:
    for line in f.readlines():
        data = json.loads(line)
        data["agent_output"] = ""
        lib = {i[0]: i[1] for i in data["context"]}
        lib_emb = [[i, text2emb(i)] for i in lib.keys()]
        supporting_facts = {}
        for i in data["supporting_facts"]:
            if i[0] not in lib or i[1] >= len(lib[i[0]]):
                continue
            if i[0] in supporting_facts:
                supporting_facts[i[0]].append(lib[i[0]][i[1]].strip())
            else:
                supporting_facts[i[0]] = [lib[i[0]][i[1]].strip()]
        supporting_facts_order = list(supporting_facts.keys())
        searched = set()
        
        prompt_, idx = prompt.format(data["question"]), 0
        while idx < 5:
            res = playground.generate(prompt_).strip('\n').strip('response:').strip(' ').strip('</s>')
            action, param, prefix = get_action(res)
            
            if action == "Search":
                try:
                    search_entity = search_memory(param, lib_emb)
                except:
                    break
                idx += 1
                prefix += "[{}] ({})".format(action, param)
                if search_entity not in searched:
                    # First search, call the summary model
                    suffix = "\nObservation{}: Search Result - {} (Summary version)\n".format(idx, search_entity)
                    train_prompt = prompt_ + prefix + "\nObservation{}: Search Result - {}\n".format(idx, search_entity) + "\n".join(lib[search_entity])
                    train_prompt = "You are an intelligent agent with the ability to search knowledge. Please summarize the searched content based on the question and historical records, and extract the most relevant parts to the question. If there is no relevant content, return [no information]\n\n" + train_prompt.split("Please note that the answer must be the span in the observation sentences.")[1].strip() + "\n\nSummary:"
                    observe = summary_playground.generate(train_prompt).strip('\n').strip('response:').strip(' ').strip('<s>').strip('</s>')
                    print(train_prompt)
                    print(observe)
                    print("---")
                    searched.add(search_entity)
                else:
                    # Second search, no summary, provide all information
                    suffix = "\nObservation{}: Search Result - {} (Full version)\n".format(idx, search_entity)
                    observe = "\n".join(lib[search_entity])
                    lib = {k: v for k, v in lib.items() if k != search_entity} # Delete entity to avoid being searched for the third time
                    lib_emb = [i for i in lib_emb if i[0] != search_entity]
                prompt_ += " " + prefix + suffix + observe + "\n\nThought{}:".format(idx + 1)
                data["agent_output"] = prompt_

            elif action == "SeekAdvice":
                prompt_ += " " + prefix + "[{}] ()".format(action)
                data["agent_output"] = prompt_
                break
            
            else:
                data["eval"] = data["answer"].strip().lower() == param.strip().lower()
                data["part_eval"] = (data["answer"].strip().lower() in param.strip().lower() or param.strip().lower() in data["answer"].strip().lower())
                gpt4_eval = call_search_azure(gpt4_eval_prompt.format(question=data["question"], reference=data["answer"], response=param))
                if 'Yes' in gpt4_eval:
                    data["gpt_eval"] = True
                else:
                    data["gpt_eval"] = False
                prefix += "[{}] ({})".format(action, param)
                prompt_ += " " + prefix
                data["agent_output"] = prompt_
                break
        
        if "eval" not in data:
            data["eval"] = data["part_eval"] = data["gpt_eval"] = True
            data["seekadvice"] = True
            data["agent_output"] += "\n\nSeek advice"
        else:
            data["seekadvice"] = False
        f1.write(json.dumps(data, ensure_ascii=False) + '\n')