## Examples of data evaluation

We take the commonly-used SummEval dataset of the summarization task and Topical-Chat dataset of the dialogue response generation task as examples. For each dataset, we organize their different subsets or evaluation aspects into separate files to facilitate the calculation of correlation coefficients. 

Each sample includes the required content of the task, aspect, source and its description, and target and its description. Additional information, such as system ids, segment ids, and human ratings, is optionally included for calculating correlation coefficients. Moreover, although references are not used in our evaluation, they are still included to facilitate the calculation of other reference-based evaluation metrics.

Specifically, the evaluation aspect of knowledge use in Topical-Chat involves additional factual information. When the evaluation needs extra information beyond the source and target, it can be similarly added.
