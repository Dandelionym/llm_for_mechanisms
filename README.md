> The official repo for the paper "Leveraging large language models for nanaosynthesis mechanisms explanation"


## Abstract

With the rapid development of artificial intelligence (AI), large language models (LLMs) such as GPT-4 have garnered significant attention in the scientific community, demonstrating great potential in advancing scientific discovery. This progress raises a critical question: are these LLMs well-aligned with real-world physicochemical principles? Current evaluation strategies largely emphasize fact-based knowledge, such as material property prediction or name recognition, but they often lack an understanding of fundamental physicochemical mechanisms that require logical reasoning. To bridge this gap, our study developed a benchmark consisting of 775 multiple-choice questions focusing on the mechanisms of gold nanoparticle synthesis. By reflecting on existing evaluation metrics, we question whether a direct true-or-false assessment merely suggests conjecture. Hence, we propose a novel evaluation metric, the confidence-based score (c-score), which probes the output logits to derive the precise probability for the correct answer. Based on extensive experiments, our results show that in the context of gold nanoparticle synthesis, LLMs understand the underlying physicochemical mechanisms rather than relying on conjecture. This study underscores the potential of LLMs to grasp intrinsic scientific mechanisms and sets the stage for developing more reliable and effective AI tools across various scientific domains.


## About Data

We manually created the dataset for evaluation. It is in the format of FastChat or ShareGPT, both are popular in LLMs area. If you are interested in using this dataset, please reference our paper (see below).


## How to run

Before running, the deployment of [🚀FastChat](https://github.com/lm-sys/FastChat) is recommended for inferencing LLMs with OpenAI API fashion. It is an open platform for training, serving, and evaluating large language model based chatbots.

After deployed LLMs, you may start the inference API in LAN, e.g., http://10.11.50.197:7860, and IP address `10.11.50.197` should be your machine's actual IP. The port can also be configured in FastChat. Then you should change the configuration of address and port in `eval_opensourced_llms.py`, it is easy to config. 

Finally, you could run each evaluation by simply run `python eval_opensourced_llms.py`

If you have `OpenAI API`, you could change the API Key in the code. And the `Claude API` configuration is in similar.


## Notes

This study focus on the evaluation of LLMs in science mechanisms understanding, trying to open a new perspective for AI for Science Research.  If anyone is interested, please see our manuscript.


## Contact

Email to: `yingmingpu@gmail.com`. 

## Citation

