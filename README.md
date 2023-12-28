# tcl2023-term_project
This work proposes an approach for compressing prompts to reduce the cost of using LLMs without fine-tuning. We assume that tokens with lower probabilities have higher importance to the sentence. Therefore, we can reduce the length of the prompt by removing high probability tokens. Also, we expect that less privileged languages can remove more tokens to achieve fairness of AI.

  
mlm_compress.py: Uses Masked Language Modeling to predict probabilities. It includes parallel and sequential removal, and the model can be freely changed.  
clm_compress.py: On the other hand, uses Causal Language Modeling to predict probabilities.
evaluation.ipynb: Evaluate the results of compressed prompts using BERTScore and Perplexity.
