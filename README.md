# Teach-LLM-LR
Official implementation of the Paper [*Learning To Teach Large Language Models Logical Reasoning*](https://arxiv.org/abs/2310.09158).

Authors: **Meiqi Chen, Yubo Ma, Kaitao Song, Yixin Cao, Yan Zhang, and Dongsheng Li.**

---
![method](https://github.com/chenmeiqii/Teach-LLM-LR/assets/113371834/9c1e18d4-26a2-4ba7-9771-924ce7e78094)


## Requirements
```shell
pip install -r requirements.txt
```
Besides, install [pyke](https://pyke.sourceforge.net/) package to conduct logic programming.


## How to run experiments
```shell
python evaluate_maven.py --model_name ChatGPT --add_rule none
```

### Output Format
```JSON5
{
	"id": "fd6e73c8c60cf8c8c6007d86bedbf54c_0_40",
	"sample": "Text:\nThe Bear River < Massacre > , or the < Battle > of Bear River or Massacre at Boa Ogoi , took place in present-day Idaho on January 29 , 1863 .\n< Massacre > and < Battle >\n< Battle > and < Massacre >\nAnswers:\nCOREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.\nCOREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.\n",
	"prompt": "Text:\nThe men 's ice hockey < tournament > at the 1924 Winter Olympics in Chamonix , France , was the 2nd Olympic Championship , also serving as the 2nd World < Championships > .",
	"pair": "\n< Championships > and < tournament >\n< tournament > and < Championships >",
	"label": [
		["COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"],
		["COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"]
	],
	"answers": "COREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.\nCOREFERENCE, NO_TEMPORAL, NO_CAUSAL, NO_SUBEVENT.\n",
	"event_ids": ["4778d9ffb01bd86cc7030a3260f9557e", "aa99e9b8fa9be9777458a1d19bf6852e"],
	"pred": [
		["COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"],
		["COREFERENCE", "NO_TEMPORAL", "NO_CAUSAL", "NO_SUBEVENT"]
	],
	"wrong_num": [0, 0],
	"retrieved_rule": ["If event A and event B are COREFERENCE, then event B and event A should be COREFERENCE (COREFERENCE relation is bidirectional).", "If event A and event B are COREFERENCE, then the relations between event B and event C should be the same as that between event A and event C."]
}
```

