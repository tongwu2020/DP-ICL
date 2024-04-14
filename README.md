# Privacy-Preserving In-Context Learning for Large Language Models

 
This repository hosts the code and resources for replicating the paper "Privacy-Preserving In-Context Learning for Large Language Models" on ICLR 2024. [[Paper]](https://openreview.net/pdf?id=x4OPJ7lHVU)


### Abstract 
In-context learning (ICL) is an important capability of Large Language Models (LLMs), enabling these models to dynamically adapt based on specific, in-context exemplars, thereby improving accuracy and relevance. However, LLM's responses may leak the sensitive private information contained in in-context exemplars. To address this challenge, we propose Differentially Private In-context Learning (DP-ICL), a general paradigm for privatizing ICL tasks. The key idea for DP-ICL paradigm is generating differentially private responses through a noisy consensus among an ensemble of LLM's responses based on disjoint exemplar sets. Based on the general paradigm of DP-ICL, we instantiate several techniques showing how to privatize ICL for text classification and language generation. We experiment on four text classification benchmarks and two language generation tasks, and our empirical findings suggest that our DP-ICL achieves a strong utility-privacy tradeoff.



### Please check the jupyter Notebook

Important Note: Due to OpenAI's deprecation of the model previously utilized in our experiments, direct replication of some results may no longer be feasible. Nonetheless, we have archived the individual predictions, which are accessible for experiments with varying privacy parameters.


[[Google Drive Link]](https://drive.google.com/drive/u/1/folders/0AJCvzTWYIQ14Uk9PVA)



## Cite our paper
```
@inproceedings{
wu2024privacypreserving,
title={Privacy-Preserving In-Context Learning for Large Language Models},
author={Tong Wu and Ashwinee Panda and Jiachen T. Wang and Prateek Mittal},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=x4OPJ7lHVU}
}
```
