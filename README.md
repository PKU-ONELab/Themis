# Themis: A Reference-free NLG Evaluation Language Model with Flexibility and Interpretability

This is the official repository for our EMNLP 2024 paper [Themis: A Reference-free NLG Evaluation Language Model with Flexibility and Interpretability](https://aclanthology.org/2024.emnlp-main.891.pdf).

## Introduction

We propose **Themis**, an 8B-parameter large language model (LLM) specifically designed and trained for NLG evaluation with more comprehensive capabilities. 

Our Themis can evaluate various NLG tasks, including uncommon ones like question-answering evaluation (**Versatility**), in a reference-free manner (**Independence**). Moreover, it allows for specific and customized evaluation aspects and criteria, including overall quality and more fine-grained aspects (**Flexibility**), and its evaluation contains corresponding analysis and explanation together with the rating (**Interpretability**). 

We believe that an ideal evaluator should be convenient to use and possess these characteristics. The comparison between related methods and Themis is shown in the table below.

|      Method       | Versatility | Independence | Flexibility | Interpretability | Open-source |
| :---------------: | :---------: | :----------: | :---------: | :--------------: | :---------: |
|      UniEval      |      ❌      |      ❌       |      ✔️      |        ❌         |      ✔️      |
|      G-Eval       |      ✔️      |      ✔️       |      ✔️      |        ✔️         |      ❌      |
|      X-Eval       |      ✔️      |      ❌       |      ✔️      |        ❌         |      ❌      |
|    Prometheus     |      ✔️      |      ❌       |      ✔️      |        ✔️         |      ✔️      |
|      Auto-J       |      ✔️      |      ✔️       |      ❌      |        ✔️         |      ✔️      |
|   InstructScore   |      ✔️      |      ❌       |      ❌      |        ✔️         |      ✔️      |
|    TIGERScore     |      ✔️      |      ✔️       |      ❌      |        ✔️         |      ✔️      |
| **Themis (Ours)** |      ✔️      |      ✔️       |      ✔️      |        ✔️         |      ✔️      |

## Performance

We implement experiments on several common NLG evaluation tasks and datasets to compare our Themis with other methods, including SummEval for summarization, Topical-Chat for dialogue response generation, SFRES&SFHOT for data-to-text, QAGS for factuality, MANS for story generation, and WMT23 zh-en for machine translation. Experimental results show that our Themis achieves better overall evaluation performance over other evaluation models, including GPT-4.

| Method               | SummEval  | Topical-Chat | SFHOT& SFRES |  QAGS  |  MANS  |  WMT23  | Average $\rho$ |
| -------------------- | :-------: | :----------: | :---------: | :-------: | :-------: | :-------: | :------------: |
| BLEU                 |   0.075   |    0.388     |    0.024    |     -     |   0.032   |   0.021   |       -        |
| ROUGE                |   0.152   |    0.412     |    0.101    |     -     |  -0.002   |   0.151   |       -        |
| BARTScore            |   0.329   |    0.086     |    0.208    |   0.425   |   0.350   |   0.118   |     0.253      |
| BERTScore            |   0.231   |    0.394     |    0.139    |     -     |   0.285   |   0.219   |       -        |
| BLEURT               |   0.152   |    0.388     |    0.244    |     -     |   0.138   |   0.263   |       -        |
| CometKiwi            |   0.228   |    0.340     |    0.251    |   0.094   |   0.251   |   0.343   |     0.251      |
| UniEval              |   0.474   |    0.577     |    0.282    |     -     |     -     |     -     |       -        |
| G-Eval (GPT-3.5)     |   0.409   |    0.585     |      -      |   0.461   |     -     |     -     |       -        |
| G-Eval (GPT-4)       |   0.523   |    0.588     |      -      |   0.611   |     -     |     -     |       -        |
| GPT-3.5 Turbo        |   0.416   |    0.578     |    0.306    |   0.431   |   0.328   |   0.347   |     0.401      |
| GPT-4 Turbo          |   0.511   |  **0.746**   |    0.320    |   0.637   |   0.473   | **0.437** |     0.521      |
| X-Eval               |   0.480   |    0.605     |    0.303    |   0.578   |     -     |     -     |       -        |
| Prometheus-13B       |   0.163   |    0.434     |    0.173    |     -     |   0.007   |   0.129   |       -        |
| Auto-J-13B           |   0.198   |    0.425     |    0.141    |   0.226   |   0.380   |   0.104   |     0.246      |
| TIGERScore-13B       |   0.384   |    0.346     |    0.200    |   0.504   |   0.231   |   0.248   |     0.319      |
| InstructScore-7B     |   0.258   |    0.241     |    0.247    |     -     |   0.298   |   0.219   |       -        |
| **Themis-8B (ours)** | **0.553** |    0.725     |  **0.333**  | **0.684** | **0.551** |   0.405   |   **0.542**    |

We further conduct more in-depth analyses, including generalization tests on unseen tasks like the instruction-following evaluation as well as aspect-targeted perturbation tests, and our Themis also exhibits superior evaluation performance. For more experimental results and details, please refer to our paper.

## Usage

### Environment

We use the [vllm](https://github.com/vllm-project/vllm) library to accelerate model inference and generating evaluation results. Other required libraries will be installed alongside vllm:

```
pip install vllm
```

### Model

Our Themis is now available on huggingface hub: [🤗 Themis](https://huggingface.co/PKU-ONELab/Themis)

### Data Format

The format of the data to be evaluated should be a list, where each sample is in dictionary format and includes at least the following contents:

```python
{
  "task": ""  # Which NLG task does the sample belongs to, e.g. Summarization
  "aspect": ""  # The criterion of the evaluation aspect, e.g. Fluency: Measure the quality of individual sentences of the summary...
  "source_des": ""  # The description of the source, e.g. Article
  "source": ""  # The source content
  "target_des": ""  # The description of the target, e.g. Summary
  "target": "" # The target content
}
```

More specifically, there is optional information that can be added for evaluation, and please refer to the examples in `eval_data`.

### Evaluation

Our evaluation based on Themis can be directly performed with `eval.py`, and the evaluation results will be automatically parsed for the corresponding analyses and ratings, which is simple and convenient to use.

We support the evaluation of multiple datasets at once, the specification of the temperature and the sampling number for the evaluation, as well as the calculation of the correlation coefficient between the evaluation results and human scores.

In addition, you need to specify the paths to the model and evaluation data, and the directory where the output evaluation results will be saved. More hyperparameters for vllm can be modified in `eval.py`. By default, the temperature is set to 0, and the sampling number is set to 1. Significantly, due to the implementation of vllm, even if the temperature is set to 0, using different inference settings such as GPU numbers and max_num_seqs will still result in different results.

```
CUDA_VISIBLE_DEVICES=<GPU_ids> python eval.py \
    --model "Themis" \
    --test_dir "./eval_data" \
    --output_dir "./eval_out"
```

The format of the evaluation results is as follows:

```python
{
  "Correlation": # Optional
  "Evaluation": [
      {
        "Evaluation Outputs": [
          {
            "Analysis": # The evaluation analysis
            "Rating": # The evaluation rating
          }, ...
        ],
        "Final Rating":  # The average of multiple evaluation ratings if the sampling number is more than 1
      }, ...
  ]
}
```

You can also set the temperature to greater than 0, e.g. 0.9, and the sampling number to greater than 1, such as 10, to obtain multiple evaluation results. When human scores are included in the input data to be evaluated, you can additionally set `correlation=True` to calculate and output the Pearson, Spearman, and Kendall correlation coefficients between the evaluation results and human scores. When the sampling number is greater than 1, the average evaluation rating is used for calculation.

```
CUDA_VISIBLE_DEVICES=<GPU_ids> python eval.py \
    --model "Themis" \
    --test_dir "./eval_data" \
    --output_dir "./eval_out" \
    --sampling_n 10 \
    --temperature 0.9 \
    --correlation True
```

## Citation

```
@inproceedings{hu2024themis,
  title={Themis: A Reference-free NLG Evaluation Language Model with Flexibility and Interpretability},
  author={Hu, Xinyu and Lin, Li and Gao, Mingqi and Yin, Xunjian and Wan, Xiaojun},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={15924--15951},
  year={2024}
}
```
