import os
import random
import json

from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, RobertaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import logging
import datetime
import argparse
import torch

from src.utils import ONE_PAIR_RULES, random_select_consistent_answers, prepare_rule_text, \
    run_openai_gpt, ALL_RELATION_TYPES, \
    check_rules_with_engine_double
from pyke import knowledge_engine
date_string = datetime.datetime.now().strftime("%Y-%m-%d")
logger = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
# roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_path = {
    "llama2-13b": "/cpfs01/shared/CausalAI/HuggingfaceModels/Llama-2-13b-hf",
    "vicuna-13b": "/cpfs01/shared/CausalAI/HuggingfaceModels/vicuna-13b-v1.3"
}
instruction = '''There is a piece of text with some events marked by < and > symbols, and your task is to identify the relations between each two events. There are four types of relations that need to identify.
(1) Coreference relation. If events A and B refer to the same event semantically, choose "COREFERENCE". Otherwise, choose "NO_COREFERENCE".
(2) Temporal relation. "NO_TEMPORAL": if there is no clear temporal relation between event A and B. "BEFORE": if event A happened completely before event B. "OVERLAP": if event A has an overlap with event B. "CONTAINS": if event A's time contains event B's time. "SIMULTANEOUS": if events A and B happen at the same time. "ENDS-ON": if event A ends when event B starts. "BEGINS-ON": if event A and event B starts at the same time, but ends at different times. 
(3) Causal relation. "NO_CAUSAL": if there is no clear causal relation between event A and B. "PRECONDITION": if event B would not have happened if event A had not happened. "CAUSE": if event B was inevitable given event A. 
(4) Subevent relation. If event A spatiotemporally CONTAINs event B and event B is a component part of event A, choose "SUBEVENT". Otherwise, choose "NO_SUBEVENT". 
Note that each relation above refers to the direction from event A to event B, so you need to judge both the relations between event A and event B and the relations between event B and event A.
'''
instruction_new = '''
We need to identify four types of relations. Let's think in this way:
1) First extract the obvious relations between events.
2) Then choose the relevant rules above according to the extracted known relations.
3) Finally, infer the remaining relations according to the selected rules and the known relations.
'''
def evaluate(args):
    engine = knowledge_engine.engine(".")
    evaluate_events(args, engine)

def evaluate_events(args, engine):
    train_sample = []
    with open(os.path.join(args.cache_path, "train_sent_prompts_events.jsonl"), "r", encoding="utf-8") as f0:
        lines = f0.readlines()
        for line in lines:
            data = json.loads(line.strip())
            train_sample.append(data)
    f0.close()
    if args.add_rule.startswith('retri') or args.add_rule == 'post':
        # run "none" rule before run "retri" rule!
        if args.add_rule == 'retri1' or args.add_rule == 'post':
            input_dir = os.path.join("output/42/MAVEN-ERE/", args.model_name, "test_events_none_res.jsonl")
        else:
            input_dir = os.path.join("output/42/MAVEN-ERE/", args.model_name,
                                 "test_events_"+args.add_rule[:-1]+str(int(args.add_rule[-1])-1)+"_res.jsonl")
    else:
        input_dir = os.path.join(args.cache_path, "test_sent_prompts_events.jsonl")
    sent_prompts = []
    with open(input_dir, "r", encoding="utf-8") as f1:
        lines = f1.readlines()
        for line in lines:
            data = json.loads(line.strip())
            sent_prompts.append(data)
    f1.close()

    batch_prompts, batch_samples, batch_sent, batch_pair_num = [], [], [], []
    predictions = [[], [], [], []]
    labels = [[], [], [], []]
    all_num = 0
    wrong_num1, wrong_num2 = 0, 0
    rule_num1, rule_num2 = 0, 0
    if args.add_rule == 'post':
        for sent_i, sent_prompt in enumerate(tqdm(sent_prompts)):
            gt = sent_prompt["label"]
            pred = sent_prompt["pred"]
            new_pred = []
            for p in pred:
                new_pred.append(list(random_select_consistent_answers([p])))
            for i in range(4):
                labels[i] += [g[i] for g in gt]
                predictions[i] += [p[i] for p in new_pred]
            # wrong_num1 += sent_prompt["wrong_num"][0]
            # wrong_num2 += sent_prompt["wrong_num"][1]
    else:
        if args.model_name in ['ChatGPT', 'GPT3', 'GPT4']:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            local_model = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path[args.model_name])
            local_model = AutoModelForCausalLM.from_pretrained(
                model_path[args.model_name],
                device_map="auto",
                torch_dtype=torch.float16,
            )
        output_path = os.path.join("output/42/MAVEN-ERE/", args.model_name,
                                   "test_events_" + args.add_rule + "_res.jsonl")
        with open(output_path, "w", encoding="utf-8") as f2:
            for sent_i, sent_prompt in enumerate(tqdm(sent_prompts)):
                num_pairs = len(sent_prompt["label"])
                if args.add_rule == 'new':
                    instruction_prompt = '''Text:
Before her death , Todd posted a video on YouTube in which she used a series of flash cards to < tell > her experience of being blackmailed into exposing her breasts via webcam , and of being < bullied > and physically assaulted .
Event Pairs:
< tell > and < bullied >
< bullied > and < tell >
From the text, we could first get:
< bullied > happens before < seized >, and being < bullied > causes her to < tell >
Reasoning: based on the context and due to the logic rule: BEFORE A B, happens relations subevent and A or event and event then won’t if OVERLAP CAUSEs B, have event they event coreference.
Therefore, the answer is:
NO_COREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.
NO_COREFERENCE, BEFORE, PRECONDITION, NO_SUBEVENT.
\n'''
                elif args.add_rule == 'self':
                    instruction_prompt = '''Text:
Before her death , Todd posted a video on YouTube in which she used a series of flash cards to < tell > her experience of being blackmailed into exposing her breasts via webcam , and of being < bullied > and physically assaulted .
Event Pairs:
< tell > and < bullied >
< bullied > and < tell >
Reasoning: < bullied > happens before < seized >, and being < bullied > causes her to < tell >.
Answers:
NO_COREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.
NO_COREFERENCE, BEFORE, PRECONDITION, NO_SUBEVENT.
\n'''
                elif args.add_rule.startswith('retri'):
                    instruction_prompt = sent_prompt["sample"]
                else:
                    instruction_prompt = '''Text:
Before her death , Todd posted a video on YouTube in which she used a series of flash cards to < tell > her experience of being blackmailed into exposing her breasts via webcam , and of being < bullied > and physically assaulted .
Event Pairs:
< tell > and < bullied >
< bullied > and < tell >
Answers:
NO_COREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.
NO_COREFERENCE, BEFORE, PRECONDITION, NO_SUBEVENT.
\n'''
                if args.add_rule == 'none':
                    gpt3_input_text = instruction + instruction_prompt + sent_prompt["prompt"] + sent_prompt["pair"] + "\nAnswers:\n"
                elif args.add_rule.startswith('retri'):
                    if len(sent_prompt["retrieved_rule"]) == 0 or (sent_prompt["wrong_num"][0]+sent_prompt["wrong_num"][1]==0) :
                        gt = sent_prompt["label"]
                        pred = sent_prompt["pred"]
                        for i in range(4):
                            labels[i] += [g[i] for g in gt]
                            predictions[i] += [p[i] for p in pred]
                        wrong_num1 += sent_prompt["wrong_num"][0]
                        wrong_num2 += sent_prompt["wrong_num"][1]
                        f2.write(json.dumps(sent_prompt) + '\n')
                        continue
                    else:
                        rule_prompt = prepare_rule_text(sent_prompt["retrieved_rule"], shuffle=True)
                        if args.model_name == "ChatGPT" or args.model_name == "GPT4":
                            gpt3_input_text = (instruction + instruction_prompt + sent_prompt["prompt"] + sent_prompt["pair"] + "\nAnswers:\n", ".\n".join([", ".join(p) for p in sent_prompt["pred"]]) + '.\n', "Your answers are logically inconsistent. " + rule_prompt + sent_prompt["pair"] + "\nRevised Answers:\n")
                        else:
                            gpt3_input_text = instruction + rule_prompt + instruction_prompt + sent_prompt["prompt"] + sent_prompt["pair"] + "\nAnswers:\n"
                elif args.add_rule == "all":
                    rule_prompt = prepare_rule_text([ONE_PAIR_RULES[r] for r in ONE_PAIR_RULES], shuffle=True)
                    gpt3_input_text = instruction + rule_prompt + instruction_prompt + sent_prompt["prompt"] + sent_prompt["pair"] + "\nAnswers:\n"
                elif args.add_rule == 'self':
                    gpt3_input_text = instruction + instruction_prompt + sent_prompt["prompt"] + sent_prompt["pair"] + "\nReasoning:\n"
                elif args.add_rule == 'new':
                    rule_prompt = prepare_rule_text([ONE_PAIR_RULES[r] for r in ONE_PAIR_RULES])
                    gpt3_input_text = instruction + instruction_new + instruction_prompt + sent_prompt["prompt"] + sent_prompt["pair"] + "\nFrom the text, we could first get:\n"
                length = len(tokenizer.encode(gpt3_input_text)) + 30 * num_pairs
                if args.add_rule == 'new' or args.add_rule == 'self': length += 200
                if length > args.max_length:
                    gt = sent_prompt["label"]
                    pred = sent_prompt["pred"]
                    for i in range(4):
                        labels[i] += [g[i] for g in gt]
                        predictions[i] += [p[i] for p in pred]
                    wrong_num1 += sent_prompt["wrong_num"][0]
                    wrong_num2 += sent_prompt["wrong_num"][1]
                    f2.write(json.dumps(sent_prompt) + '\n')
                    continue
                if len(batch_samples) < args.batch_size:
                    batch_prompts.append(gpt3_input_text)
                    batch_samples.append(instruction_prompt)
                    batch_sent.append(sent_i)
                    batch_pair_num.append(num_pairs)
                    all_num += len(sent_prompt["label"])
                if len(batch_samples) == args.batch_size:
                    openai_samples, preds = run_openai_gpt(batch_prompts, batch_pair_num, args.model_name, args.add_rule, tokenizer=tokenizer, device=args.device, local_model=local_model)
                    for sample, s_i, pred, s_j in zip(batch_samples, batch_sent, preds, range(20)):
                        gt = sent_prompts[s_i]["label"]
                        for i in range(4):
                            labels[i] += [g[i] for g in gt]
                            predictions[i] += [p[i] for p in pred] #[[],[],[]]

                        if args.add_rule == 'new':
                            if args.model_name == 'ChatGPT' or args.model_name == "GPT4":
                                logging.info('\n\nText:\n' + openai_samples["choices"][s_j]['message'][
                                    'content'] + '\nExpected Answers:\n' + ".\n".join([", ".join(g) for g in gt]))
                            elif args.model_name == 'GPT3':
                                logging.info('\n\nText:\n' + openai_samples["choices"][s_j][
                                    'text'] + '\nExpected Answers:\n' + ".\n".join([", ".join(g) for g in gt]))
                            else:
                                logging.info('\n\nText:\n' + openai_samples["choices"][s_j] + '\nExpected Answers:\n' + ".\n".join([", ".join(g) for g in gt]))
                        rule_text1, rule_text2, wn1, wn2 = check_rules_with_engine_double(pred, engine, sent_prompts[s_i]["event_ids"])
                        rule = list(rule_text1.union(rule_text2))
                        result = {
                            "id":sent_prompts[s_i]["id"],
                            "sample": sample,
                            "prompt": sent_prompts[s_i]["prompt"],
                            "pair":  sent_prompts[s_i]["pair"],
                            "label": gt,
                            "answers": sent_prompts[s_i]["answers"],
                            "event_ids":sent_prompts[s_i]["event_ids"],
                            "pred": pred,
                            "wrong_num":[wn1, wn2],
                            "retrieved_rule": rule
                        }
                        wrong_num1 += wn1
                        wrong_num2 += wn2
                        rule_num1+=len(rule_text1)
                        rule_num2+=len(rule_text2)
                        f2.write(json.dumps(result) + '\n')
                    batch_prompts, batch_samples, batch_sent, batch_pair_num = [], [], [], []

                if (sent_i == len(sent_prompts) - 1) and len(batch_samples) > 0:
                    openai_samples, preds = run_openai_gpt(batch_prompts, batch_pair_num, args.model_name,
                                                           args.add_rule, tokenizer=tokenizer, device=args.device, local_model=local_model)
                    for sample, s_i, pred, s_j in zip(batch_samples, batch_sent, preds, range(20)):
                        gt = sent_prompts[s_i]["label"]
                        for i in range(4):
                            labels[i] += [g[i] for g in gt]
                            predictions[i] += [p[i] for p in pred]  # [[],[],[]]

                        if args.add_rule == 'new':
                            if args.model_name == 'ChatGPT' or args.model_name == "GPT4":
                                logging.info('\n\nText:\n' + openai_samples["choices"][s_j]['message'][
                                    'content'] + '\nExpected Answers:\n' + ".\n".join([", ".join(g) for g in gt]))
                            elif args.model_name == 'GPT3':
                                logging.info('\n\nText:\n' + openai_samples["choices"][s_j][
                                    'text'] + '\nExpected Answers:\n' + ".\n".join([", ".join(g) for g in gt]))
                            else:
                                logging.info('\n\nText:\n' + openai_samples["choices"][s_j] + '\nExpected Answers:\n' + ".\n".join([", ".join(g) for g in gt]))

                        rule_text1, rule_text2, wn1, wn2 = check_rules_with_engine_double(pred, engine,
                                                                                   sent_prompts[s_i]["event_ids"])
                        rule = list(rule_text1.union(rule_text2))
                        result = {
                            "id": sent_prompts[s_i]["id"],
                            "sample": sample,
                            "prompt": sent_prompts[s_i]["prompt"],
                            "pair": sent_prompts[s_i]["pair"],
                            "label": gt,
                            "answers": sent_prompts[s_i]["answers"],
                            "event_ids": sent_prompts[s_i]["event_ids"],
                            "pred": pred,
                            "wrong_num": [wn1, wn2],
                            "retrieved_rule": rule
                        }
                        wrong_num1 += wn1
                        wrong_num2 += wn2
                        rule_num1 += len(rule_text1)
                        rule_num2 += len(rule_text2)
                        f2.write(json.dumps(result) + '\n')
                    batch_prompts, batch_samples, batch_sent, batch_pair_num = [], [], [], []

        f2.close()
    for i in range(4):
        logging.info(classification_report(labels[i], predictions[i], labels=ALL_RELATION_TYPES[i][1:], zero_division=0))
    logging.info("All Counts:{}, WN1:{}, WN2:{}, R1:{}, R2:{}.".format(all_num, wrong_num1, wrong_num2, rule_num1, rule_num2))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default='./cached_prompts', type=str)
    parser.add_argument("--output_dir", default='./output', type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--prompt_size", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_name", default='GPT4', type=str, choices=['ChatGPT', 'GPT3', 'GPT4', 'vicuna-13b', 'llama2-13b'])
    parser.add_argument("--add_rule", default='none', type=str, choices=['none', 'self', 'new', 'retri', 'all'])
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # sys.stdout = open(os.path.join(date_string+"_log.txt"), 'w')
    filename = os.path.join(args.output_dir, args.model_name,
                            date_string + "_events_" + args.add_rule + '_log.txt')
    logging.basicConfig(
        filename=filename, \
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
    )
    logging.info("Parameters:{}\n".format(args))
    random.seed(args.seed)
    if args.add_rule == 'none':
        print("Not Using Rules.")
    else:
        print("Using Rules.")
    evaluate(args)


if __name__ == "__main__":
    main()