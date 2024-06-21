import time

import httpx
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import copy
import random
import openai
import json
# from transformers import GPT2Tokenizer, RobertaTokenizer
from collections import deque, defaultdict
from logic_rules import checkrules
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
BIDIRECTIONAL_REL = ["SIMULTANEOUS", "BEGINS-ON"]
TEXT2LABEL = {",":",", "N":"NONE", "C":"COREFERENCE", "F":"BEFORE", "O":"OVERLAP", "T":"CONTAINS", "S":"SIMULTANEOUS", "E":"ENDS-ON", "G":"BEGINS-ON","P":"PRECONDITION", "U":"CAUSE","V":"SUBEVENT"}
LABEL2TEXT = {v:k for k, v in TEXT2LABEL.items()}
EACH_NONE = {"NO_COREFERENCE":"NONE", "NO_TEMPORAL":"NONE", "NO_CAUSAL":"NONE", "NO_SUBEVENT":"NONE"}
ALL_RELATION_TYPES = {
    0:["NO_COREFERENCE","COREFERENCE"],
    1:["NO_TEMPORAL", "BEFORE","OVERLAP","CONTAINS","SIMULTANEOUS","ENDS-ON","BEGINS-ON"],
    2:["NO_CAUSAL", "PRECONDITION","CAUSE"],
    3:["NO_SUBEVENT", "SUBEVENT"]
}
ALL_RELATION_TYPES_CTB = {
    0:["NO_TEMPORAL", "BEFORE","OVERLAP","CONTAINS","SIMULTANEOUS","ENDS-ON","BEGINS-ON"],
    1:["NO_CAUSAL", "PRECONDITION","CAUSE"],
}
ONE_PAIR_RULES = {
"COREFERENCE": "If two events are COREFERENCE, then they won’t have temporal, causal, and subevent relations.",
"BEFORE": "If event A happens BEFORE event B, then they won't have coreference and subevent relations.",
"OVERLAP": "If event A happens OVERLAP with event B, then they won't have coreference and subevent relations.",
"CONTAINS": "If event A's time CONTAINS event B's time, then they won't have coreference and causal relations.",
"SIMULTANEOUS": "If events A and event B happen SIMULTANEOUSly, then they won’t have coreference, causal, and subevent relations.",
"ENDS-ON": "If event A ENDS-ON event B, then they won’t have coreference, causal and subevent relations.",
"BEGINS-ON": "If event A BEGINS-ON event B, then they won’t have coreference, causal and subevent relations.",
"CAUSE":"If event A CAUSEs event B, then event A happens BEFORE or OVERLAP with event B, and they won't have coreference and subevent relations.",
"PRECONDITION":"If event A is event B’s PRECONDITION, then event A happens BEFORE or OVERLAP with event B, and they won't have coreference and subevent relations.",
"SUBEVENT":"If event B is a SUBEVENT of event A, then they won’t have coreference and causal relations, and event A’s time should CONTAINS event B’s time.",
"NO_TEMPORAL":"If event A and event B do not have a temporal relation, then they won't have causal and subevent relations.",
"COREFERENCE_double":"If event A and event B are COREFERENCE, then event B and event A should be COREFERENCE (COREFERENCE relation is bidirectional)",
"BEFORE_double":"If event A happens BEFORE event B, then event B has NO_TEMPORAL relation with event A.",
"OVERLAP_double":"If event A happens OVERLAP with event B, then event B has NO_TEMPORAL relation with event A.",
"CONTAINS_double": "If event A's time CONTAINS event B's time, then event B has NO_TEMPORAL relation with event A.",
"SIMULTANEOUS_double": "If event A and event B happen SIMULTANEOUSly, then event B and event A happen SIMULTANEOUSly (SIMULTANEOUS relation is bidirectional).",
"ENDS-ON_double": "If event A ENDS-ON event B, then event B has NO_TEMPORAL relation with event A.",
"BEGINS-ON_double": "If event A BEGINS-ON event B, then event B BEGINS-ON event A (BEGINS-ON relation is bidirectional).",
"CAUSE_double":"If event A CAUSEs event B, then event B has NO_TEMPORAL relation with event A.",
"PRECONDITION_double":"If event A is event B's PRECONDITION, then event B has NO_TEMPORAL relation with event A.",
"SUBEVENT_double":"If event B is event A's subevent, then event B has NO_TEMPORAL relation with event A.",

}
SYMBOL_FACTS = {
"COREFERENCE": "event %s and event %s are COREFERENCE",
"BEFORE": "event %s happens BEFORE event %s",
"OVERLAP": "event %s happens OVERLAP with event %s",
"CONTAINS": "event %s's time CONTAINS event %s's time",
"SIMULTANEOUS": "events %s and event %s happen SIMULTANEOUSly",
"ENDS-ON": "event %s ENDS-ON event %s",
"BEGINS-ON": "event %s BEGINS-ON event %s",
"CAUSE":"event %s CAUSEs event %s",
"PRECONDITION":"event %s is event %s’s PRECONDITION",
"SUBEVENT":"event %s is a SUBEVENT of event %s",
}
ONE_PAIR_PRED = {
"COREFERENCE": [["COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"]],
"BEFORE": [["NO_COREFERENCE", "BEFORE", "NO_CAUSAL", "NO_SUBEVENT"]],
"OVERLAP": [["NO_COREFERENCE", "OVERLAP", "NO_CAUSAL", "NO_SUBEVENT"]],
"CONTAINS": [["NO_COREFERENCE", "CONTAINS", "NO_CAUSAL", "NO_SUBEVENT"]],
"SIMULTANEOUS": [["NO_COREFERENCE", "SIMULTANEOUS", "NO_CAUSAL", "NO_SUBEVENT"]],
"ENDS-ON": [["NO_COREFERENCE", "ENDS-ON", "NO_CAUSAL", "NO_SUBEVENT"]],
"BEGINS-ON": [["NO_COREFERENCE", "BEGINS-ON", "NO_CAUSAL", "NO_SUBEVENT"]],
"CAUSE": [["NO_COREFERENCE", "BEFORE", "CAUSE", "NO_CAUSAL"],["NONE", "OVERLAP", "CAUSE", "NO_SUBEVENT"]],
"PRECONDITION":[["NO_COREFERENCE", "BEFORE", "PRECONDITION", "NO_SUBEVENT"],["NONE", "OVERLAP", "PRECONDITION", "NO_SUBEVENT"]],
"SUBEVENT":[["NO_COREFERENCE", "CONTAINS", "NO_CAUSAL", "SUBEVENT"]],
}
class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = []
        self.eid2mentions = {}
        if "events" in data:
            for e in data["events"]:
                if len(e["mention"]) > 0:
                    e["mention"][0]["eid"] = e["id"]
                    self.events += e["mention"]
            for e in data["events"]:
                if len(e["mention"]) > 0:
                    self.eid2mentions[e["id"]] = e["mention"]
        else:
            self.events = copy.deepcopy(data['event_mentions'])

        self.clusters = data["coreference_relations"]
        self.sort_events()
        self.get_pairs()

        self.get_relations(data["coreference_relations"], "coref", data["pred_coreference_relations"])
        self.get_relations(data["temporal_relations"], "temporal", data["pred_temporal_relations"])
        self.get_relations(data["causal_relations"], "causal", data["pred_causal_relations"])
        self.get_relations(data["subevent_relations"], "subevent", data["pred_subevent_relations"])

    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_pairs(self):
        self.all_pairs = []
        for e1 in self.events:
            for e2 in self.events:
                if e1["id"] == e2["id"]:
                    continue
                self.all_pairs.append((e1["id"], e2["id"]))

    def get_coref_relations(self, coref_rels, pred_coref_rels):
        self.pair2rel = {}
        for e1 in self.events:
            for e2 in self.events:
                if e1["id"] == e2["id"]: continue
                self.pair2rel[str((e1["id"], e2["id"]))]={"coref": [],
                                                    "temporal": [],
                                                    "causal": [],
                                                    "subevent": [],
                                                    }
        for cluster in coref_rels:
            for e1_id in cluster:
                for e2_id in cluster:
                    if e1_id==e2_id:continue
                    self.pair2rel[str((e1_id, e2_id))].setdefault("coref", []).append(("COREFERENCE", 1.0, "ground-truth"))

        self.pred_pair2rel = {}
        if pred_coref_rels:
            for e1 in self.events:
                for e2 in self.events:
                    if e1["id"] == e2["id"]: continue
                    self.pred_pair2rel[str((e1["id"], e2["id"]))] = {"coref": [],
                                                                "temporal": [],
                                                                "causal": [],
                                                                "subevent": [],
                                                                }
            for cluster in pred_coref_rels:
                for e1_id in cluster:
                    for e2_id in cluster:
                        if e1_id == e2_id: continue
                        self.pred_pair2rel[str((e1_id, e2_id))].setdefault("coref", []).append(("COREFERENCE", 1.0, "pred"))

    def get_relations(self, relations, rel_type, pred_rels):
        if pred_rels:
            if rel_type == "coref":
                self.pred_pair2rel = {}
                for e1 in self.events:
                    for e2 in self.events:
                        if e1["id"] == e2["id"]: continue
                        self.pred_pair2rel[str((e1["id"], e2["id"]))] = {"coref": [],
                                                                         "temporal": [],
                                                                         "causal": [],
                                                                         "subevent": [],
                                                                         }
            for rel in pred_rels:
                for pair in pred_rels[rel]:
                    e1_id = pair[0]
                    e2_id = pair[1]
                    if len(pair) == 3: prob = pair[2]
                    else: prob = 1.0
                    if e1_id.startswith("TIME") or e2_id.startswith("TIME"):
                        self.pred_pair2rel[str((e1_id, e2_id))] = {"temporal": [(rel, prob, "pred")]}
                    else:
                        self.pred_pair2rel[str((e1_id, e2_id))].setdefault(rel_type, []).append((rel, prob, "pred"))


        if rel_type == "coref":
            self.pair2rel = {}
            for e1 in self.events:
                for e2 in self.events:
                    if e1["id"] == e2["id"]: continue
                    self.pair2rel[str((e1["id"], e2["id"]))] = {"coref": [],
                                                                "temporal": [],
                                                                "causal": [],
                                                                "subevent": [],
                                                                }
            for cluster in relations:
                for e1_id in cluster:
                    for e2_id in cluster:
                        if e1_id == e2_id: continue
                        self.pair2rel[str((e1_id, e2_id))].setdefault("coref", []).append(
                            ("COREFERENCE", 1.0, "ground-truth"))
        else:
            for rel in relations:
                for pair in relations[rel]:
                    if pair[0] in self.eid2mentions and pair[1] in self.eid2mentions:
                        for e1 in self.eid2mentions[pair[0]]:
                            for e2 in self.eid2mentions[pair[1]]:
                                self.pair2rel[str((e1["id"], e2["id"]))].setdefault(rel_type, []).append((rel, 1.0, "ground-truth"))
                                if rel in BIDIRECTIONAL_REL:
                                    self.pair2rel[str((e2["id"], e1["id"]))].setdefault(rel_type, []).append((rel, 1.0, "ground-truth"))

def check_rules_with_engine(preds, engine, event_ids):
    combinations = [(event_ids[i], event_ids[j]) for i in range(len(event_ids)) for j in range(i + 1, len(event_ids))]
    for com in combinations:
        engine.assert_('relation', 'not_equal', (com[0], com[1]))
    for p_i, p in enumerate(preds):
        for i in p:
            engine.assert_('relation', i.lower().replace("-", "_"), combinations[p_i])
    engine.activate('rules_two_events')
    engine.activate('rules_three_events')
    wrong_num1, wrong_num2 = 0, 0
    rules1, rules2 = set(), set()
    inconsistent_anwer, engine_rule = checkrules.inconsistent_answers, checkrules.engine_rules
    for i_a_key in inconsistent_anwer:
        if i_a_key[0] in event_ids:
            for i_a in inconsistent_anwer[i_a_key]:
                if len(i_a) == 2:
                    wrong_num1+=1
                elif len(i_a) == 3:
                    wrong_num2+=1
    for e_r in engine_rule:
        if e_r[0] in event_ids:
            for r in engine_rule[e_r]:
                if len(e_r) == 2:
                    rules1.add(r)
                elif len(e_r) == 3:
                    rules2.add(r)
    engine.reset()
    checkrules.inconsistent_answers ={}
    checkrules.engine_rules = {}
    return rules1, rules2, wrong_num1, wrong_num2

def check_rules_with_engine_double(preds, engine, event_ids):
    combinations = [(event_ids[i], event_ids[j]) for i in range(len(event_ids)) for j in range(len(event_ids))if i!=j]
    for com in combinations:
        engine.assert_('relation', 'not_equal', (com[0], com[1]))
    for p_i, p in enumerate(preds):
        for i in p:
            engine.assert_('relation', i.lower().replace("-", "_"), combinations[p_i])
    engine.activate('rules_two_events')
    engine.activate('rules_three_events')
    wrong_num1, wrong_num2 = 0, 0
    rules1, rules2 = set(), set()
    inconsistent_anwer, engine_rule = checkrules.inconsistent_answers, checkrules.engine_rules
    for i_a_key in inconsistent_anwer:
        if i_a_key[0] in event_ids:
            for i_a in inconsistent_anwer[i_a_key]:
                if len(i_a) == 2:
                    wrong_num1+=1
                elif len(i_a) == 3:
                    wrong_num2+=1
    for e_r in engine_rule:
        if e_r[0] in event_ids:
            for r in engine_rule[e_r]:
                if len(e_r) == 2:
                    rules1.add(r)
                elif len(e_r) == 3:
                    rules2.add(r)
    engine.reset()
    checkrules.inconsistent_answers ={}
    checkrules.engine_rules = {}
    return rules1, rules2, wrong_num1, wrong_num2

def sort_option(lst, relation_types):
    sorted_arr = []
    for i in range(len(relation_types)):
        for j in relation_types[i]:
            if j in lst:
                sorted_arr.append(j)
    return sorted_arr

def random_select_consistent_answers(preds):
    with open("./../logic_rules/rules.json", "r", encoding="utf-8") as f:
        rules = json.load(f)
    f.close()
    two_events_rules = rules["two_events"]
    rule_option = set()
    for pred in preds:
        for j in pred:
            option = []
            flag = [0, 0, 0, 0]
            if j.startswith("NO") and j!="NO_TEMPORAL":
                option = ("NO_COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT")
                rule_option.add(option)
                continue
            if j == "NO_TEMPORAL":
                rule_option.add(("NO_COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"))
                rule_option.add(("COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"))
                continue
            else:
                option.append(j)
                must = two_events_rules[j]["must"]
                should_not = two_events_rules[j]["should_not"]
                for m in must:
                    option.append(m)
                for rel in ALL_RELATION_TYPES:
                    if any (elem in set(ALL_RELATION_TYPES[rel]) for elem in option):
                        flag[rel]=1
                if flag[0] == flag[1] == flag[2] == flag[3] == 1:
                    sorted_arr = sort_option(option, ALL_RELATION_TYPES)
                    rule_option.add(tuple(sorted_arr))
                else:
                    for f_i,f in enumerate(flag):
                        candidate = []
                        if f!=1:
                            for r in ALL_RELATION_TYPES[f_i]:
                                if r not in should_not:
                                    candidate.append(r)
                            for can in candidate:
                                rule_option.add(tuple(sort_option(option+[can], ALL_RELATION_TYPES)))

    return random.choice(list(rule_option))

def random_select_consistent_answers_ctb(preds):
    with open("./../logic_rules/rules.json", "r", encoding="utf-8") as f:
        rules = json.load(f)
    f.close()
    two_events_rules = rules["two_events"]
    rule_option = set()
    for pred in preds:
        for j in pred:
            option = []
            flag = [0, 0]
            if j.startswith("NO") :
                option = ("NO_TEMPORAL", "NO_CAUSAL")
                rule_option.add(option)
                continue
            else:
                option.append(j)
                must = two_events_rules[j]["must"]
                should_not = two_events_rules[j]["should_not"]
                for m in must:
                    option.append(m)
                for rel in ALL_RELATION_TYPES_CTB:
                    if any (elem in set(ALL_RELATION_TYPES_CTB[rel]) for elem in option):
                        flag[rel]=1
                if flag[0] == flag[1]  == 1:
                    sorted_arr = sort_option(option, ALL_RELATION_TYPES_CTB)
                    rule_option.add(tuple(sorted_arr))
                else:
                    for f_i,f in enumerate(flag):
                        candidate = []
                        if f!=1:
                            for r in ALL_RELATION_TYPES_CTB[f_i]:
                                if r not in should_not:
                                    candidate.append(r)
                            for can in candidate:
                                rule_option.add(tuple(sort_option(option+[can], ALL_RELATION_TYPES_CTB)))

    return random.choice(list(rule_option))

def shuffle_words(sentence):
    words = sentence.split()
    random.shuffle(words)
    shuffled_sentence = ' '.join(words)
    return shuffled_sentence

def prepare_rule_text(rules, shuffle=False):
    rules = set(rules)
    if len(rules)==0: return "\n"
    all_rules = "There are some rules among the relations, you can select some of them to reason or check your answers:\n"
    for i, rule in enumerate(rules):
        if shuffle:
            rule = shuffle_words(rule)
        all_rules += '(' + str(i+1) + ') ' + rule + '\n'
    return  all_rules


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def run_openai_gpt(prompt, pair_num, model_name, add_rule, tokenizer=None, device="cpu", local_model=None, temperature=0, stop_words="Text:"):
    client = OpenAI(api_key="",
                    http_client=httpx.Client(),
                    )
    max_token = max(pair_num) * 30
    if add_rule == 'new' or add_rule == 'self':
        max_token += 200
    if model_name == "GPT3":
        deployment_name = "text-davinci-003"
        sample = openai.Completion.create(model=deployment_name,
                                          prompt=prompt,
                                          max_tokens=max_token,
                                          temperature=0,
                                          stream=False,
                                          stop=[stop_words])
        time.sleep(0.1)
    elif model_name == "ChatGPT" or model_name == "GPT4":
        if model_name == "ChatGPT":
            deployment_name = "gpt-3.5-turbo-0301"
        else:
            deployment_name = "gpt-4"
        message = []
        if add_rule.startswith('retri'):
            for p in prompt:
                message=[{"role": "user", "content": p[0]}, {"role": "assistant", "content": p[1]}, {"role": "user", "content": p[2]}]
        else:
            for p in prompt:
                message=[{"role": "user", "content": p}]

        chat_sample = client.chat.completions.create(
            model=deployment_name,
            messages=message,
            max_tokens=max_token,
            temperature=0,
            stream=False,
            stop = [stop_words]
        )
        sample = {"choices":[{"message":{"content": chat_sample.choices[0].message.content}}]}

        time.sleep(0.2)
    else:
        if temperature <= 0.0:
            do_sample = False
            temperature += 0.1
        stop_word_id = 13  # '\n'
        sample = defaultdict(list)
        for inputs in prompt:
            inputs = tokenizer(inputs, return_tensors="pt")
            input_ids_length = inputs.input_ids.size(1)
            generate_ids = local_model.generate(
                inputs.input_ids.to(device),
                max_new_tokens=max_token,
                do_sample=do_sample,
                early_stopping=True
            )
            response = tokenizer.batch_decode(generate_ids[:, input_ids_length:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
            sample["choices"].append(response.split(stop_words)[0])

    uncalibrated_prediction = []
    for b_i, b in enumerate(sample["choices"]):
        preds = []
        if model_name == "ChatGPT" or model_name == "GPT4":
            if add_rule == 'self':
                txt = b['message']['content'].split('Answers:\n')[-1].split('\n')
            elif add_rule == 'new':
                txt = b['message']['content'].split('Therefore, the answer is:\n')[-1].split('\n')
            else:
                txt = b['message']['content'].split('\n')
        elif model_name == "GPT3":
            if add_rule == 'self':
                txt = b['text'].split('Answers:\n')[-1].split('\n')
            elif add_rule == 'new':
                txt = b['text'].split('Therefore, the answer is:\n')[-1].split('\n')
            else:
                txt = b['text'].split('\n')
        else:
            if add_rule == 'self':
                txt = b.split('Answers:\n')[-1].split('\n')
            elif add_rule == 'new':
                txt = b.split('Therefore, the answer is:\n')[-1].split('\n')
            else:
                txt = b.split('\n')
        for text in txt:
            tmp_prediction = [i.strip() for i in text.replace(".", "").split(",")]
            if len(tmp_prediction) > 4:
                tmp_prediction = tmp_prediction[:4]
            set_prediction = set(tmp_prediction)
            prediction = []
            for key in ALL_RELATION_TYPES:
                set_key = set(ALL_RELATION_TYPES[key])
                common = list(set_prediction.intersection(set_key))
                if common:
                    prediction.append(common[0])
                else:
                    prediction.append(ALL_RELATION_TYPES[key][0])
            preds.append(prediction)
        p_num = pair_num[b_i]
        if len(preds)>p_num:
            preds = preds[:p_num]
        while len(preds)<p_num:
            preds.append(['NO_COREFERENCE', 'NO_TEMPORAL', 'NO_CAUSAL', 'NO_SUBEVENT'])
        uncalibrated_prediction.append(preds)
    return sample, uncalibrated_prediction

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def run_openai_gpt_ctb(prompt, pair_num, model_name, add_rule, tokenizer=None, device="cpu", local_model=None, temperature=0):
    client = OpenAI(api_key="",
                    http_client=httpx.Client(),
                    )
    max_token = max(pair_num) * 30
    if add_rule == 'new' or add_rule == 'self':
        max_token += 200
    if model_name == "GPT3":
        deployment_name = "text-davinci-003"
        sample = openai.Completion.create(model=deployment_name,
                                          prompt=prompt,
                                          max_tokens=max_token,
                                          temperature=0,
                                          stream=False,
                                          stop=["Text:\n"])
        time.sleep(0.1)
    elif model_name == "ChatGPT" or model_name == "GPT4":
        if model_name == "ChatGPT":
            deployment_name = "gpt-3.5-turbo-0301"
        else:
            deployment_name = "gpt-4"
        message = []
        if add_rule.startswith('retri'):
            for p in prompt:
                message=[{"role": "user", "content": p[0]}, {"role": "assistant", "content": p[1]}, {"role": "user", "content": p[2]}]
        else:
            for p in prompt:
                message=[{"role": "user", "content": p}]

        chat_sample = client.chat.completions.create(
            model=deployment_name,
            messages=message,
            max_tokens=max_token,
            temperature=0,
            stream=False,
            stop=["Text:\n"]
        )
        sample = {"choices":[{"message":{"content": chat_sample.choices[0].message.content}}]}
        time.sleep(0.2)
    else:
        if temperature <= 0.0:
            do_sample = False
            temperature += 0.1
        stop_word_id = 13  # '\n'
        sample = defaultdict(list)
        for inputs in prompt:
            inputs = tokenizer(inputs, return_tensors="pt")
            input_ids_length = inputs.input_ids.size(1)
            generate_ids = local_model.generate(
                inputs.input_ids.to(device),
                max_new_tokens=max_token,
                do_sample=do_sample,
                early_stopping=True
            )
            response = tokenizer.batch_decode(generate_ids[:, input_ids_length:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
            sample["choices"].append(response.split('Text:')[0])

    uncalibrated_prediction = []
    for b_i, b in enumerate(sample["choices"]):
        preds = []
        if model_name == "ChatGPT" or model_name == "GPT4":
            if add_rule == 'self':
                txt = b['message']['content'].split('Answers:\n')[-1].split('\n')
            elif add_rule == 'new':
                txt = b['message']['content'].split('Therefore, the answer is:\n')[-1].split('\n')
            else:
                txt = b['message']['content'].split('\n')
        elif model_name == "GPT3":
            if add_rule == 'self':
                txt = b['text'].split('Answers:\n')[-1].split('\n')
            elif add_rule == 'new':
                txt = b['text'].split('Therefore, the answer is:\n')[-1].split('\n')
            else:
                txt = b['text'].split('\n')
        else:
            if add_rule == 'self':
                txt = b.split('Answers:\n')[-1].split('\n')
            elif add_rule == 'new':
                txt = b.split('Therefore, the answer is:\n')[-1].split('\n')
            else:
                txt = b.split('\n')
        for text in txt:
            tmp_prediction = [i.strip() for i in text.replace(".", "").split(",")]
            if len(tmp_prediction) > 2:
                tmp_prediction = tmp_prediction[:2]
            set_prediction = set(tmp_prediction)
            prediction = []
            for key in ALL_RELATION_TYPES_CTB:
                set_key = set(ALL_RELATION_TYPES_CTB[key])
                common = list(set_prediction.intersection(set_key))
                if common:
                    prediction.append(common[0])
                else:
                    prediction.append(ALL_RELATION_TYPES_CTB[key][0])
            preds.append(prediction)
        p_num = pair_num[b_i]
        if len(preds)>p_num:
            preds = preds[:p_num]
        while len(preds)<p_num:
            preds.append(['NO_TEMPORAL', 'NO_CAUSAL'])
        uncalibrated_prediction.append(preds)
    return sample, uncalibrated_prediction


