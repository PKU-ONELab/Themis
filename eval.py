import os
import sys
import json
import argparse
import logging
import math 

from tqdm import tqdm
from stats import Correlation

PROMPT_W_ADD = "###Instruction###\n\
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.\n\
Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n\
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n\
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.\n\
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.\n\
\n\
###Evaluation Criterion###\n\
{aspect}\n\
\n\
###Example###\n\
{source_des}:\n\
{source}\n\
\n\
{addition_des}:\n\
{addition}\n\
\n\
{target_des}:\n\
{target}\n\
\n\
###Your Evaluation###\n"

PROMPT = "###Instruction###\n\
Please act as an impartial and helpful evaluator for natural language generation (NLG), and the audience is an expert in the field.\n\
Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n\
Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, start with \"Rating:\" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n\
You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues and errors involved; otherwise, you will be penalized.\n\
Make sure you read and understand these instructions, as well as the following evaluation criterion and example content, carefully.\n\
\n\
###Evaluation Criterion###\n\
{aspect}\n\
\n\
###Example###\n\
{source_des}:\n\
{source}\n\
\n\
{target_des}:\n\
{target}\n\
\n\
###Your Evaluation###\n"

SEP = "<sep of eval.py>"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class Namespace(argparse.Namespace):
    model: str
    test_dir: str
    output_dir: str

    ## the config of vllm
    max_new_tokens: int
    temperature: float
    sampling_n: int
    tensor_parallel_size: int
    max_num_seqs: int

    ## whether output the correlation between human ratings and model evaluations
    correlation: bool

def generate(engine, sampling_params, test_prompts, use_tqdm: bool = True):
    
    for ex in test_prompts:
        test_prompt = ex[0]
        request_id = ex[1]
        engine.add_request(request_id, test_prompt, sampling_params)

    if use_tqdm:
        num_requests = engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests, 
                    desc="Processed prompts", 
                    dynamic_ncols=True)
    
    outs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()

        for step_output in step_outputs:
            if step_output.finished:
                outs.append(step_output)
                if use_tqdm:
                    pbar.update(1)
    
    if use_tqdm:
        pbar.close()

    return [([ex.text for ex in out.outputs], out.request_id) for out in outs]


def get_prompt(ex):
    if ex.get("addition", None) is not None:
        return PROMPT_W_ADD.format_map(ex)
    return PROMPT.format_map(ex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True, type=str, help="Model name or path")
    parser.add_argument('--max_new_tokens', '-mx', default=2048, type=int, help="Max new tokens")
    parser.add_argument('--max_num_seqs', '-ms', default=1024, type=int, help="Max number of sequences")
    parser.add_argument('--test_dir', '-i', required=True, type=str, help="Testset directory")
    parser.add_argument('--output_dir', '-o', required=True, type=str, help="Output directory")
    parser.add_argument('--temperature', '-t', default=0, type=float, help="Temperature for sampling")
    parser.add_argument('--tensor_parallel_size', '-tp', default=4, type=int, help="Number of tensor parallel replicas")
    parser.add_argument('--sampling_n', '-n', default=1, type=int, help="Number of sampled sequences to return")
    parser.add_argument('--correlation', '-c', default=False, type=bool, help="Whether output the correlation between human ratings and model evaluations")
    args = parser.parse_args(namespace=Namespace())

    test_files = []
    for prefix, _, files in os.walk(args.test_dir):
        for file in files:
            file_name = file
            if file.endswith(".json"):
                file = os.path.join(prefix, file)
                sep = os.path.sep
                test_files.append((file, file_name))

    test_files.sort()
    logger.info(f"Test files: {test_files}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_test_prompts = []
    all_outputs = {}

    for file, _ in test_files:
        test_dataset = json.load(open(file, encoding='utf-8'))
        assert isinstance(test_dataset, list) , "type of testset must be list"
        assert len(test_dataset) > 0, "len of testset must larger than 0"
        assert isinstance(test_dataset[0], dict), "type of samples in testset must be dict"

        all_test_prompts.extend([ 
            (get_prompt(ex), file + SEP + str(i))
            for i, ex in enumerate(test_dataset)
        ])
        all_outputs[file] = [None for _ in range(len(test_dataset))]
            
    
    def process(inputs):
        from vllm import EngineArgs, LLMEngine, SamplingParams
        engine_args = EngineArgs(model=args.model, 
                                 tensor_parallel_size=args.tensor_parallel_size,
                                 max_num_seqs=args.max_num_seqs,
                                 max_num_batched_tokens=max(args.max_num_seqs, args.max_new_tokens),
                                 gpu_memory_utilization=0.98,
                                 swap_space=16)
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature, n=args.sampling_n)
        return generate(engine, sampling_params, inputs)
        
    outs = process(all_test_prompts)

    for ex in outs:
        text, id = ex
        file, i = id.split(SEP)
        all_outputs[file][int(i)] = text

    def parse(out: str):
        last_line = out.split('\n')[-1]
        if last_line.startswith("Rating: "):
            try: 
                rating = float(last_line[8:])
                if math.isfinite(rating):
                    return {"Analysis": '\n'.join(out.split('\n')[:-1]), "Rating": rating}
            except:
                pass

        return {"Analysis": out, "Rating": 0}

    for file, name in test_files:
        for i, ex in enumerate(all_outputs[file]):
            outs = []
            for out in ex:
                outs.append(parse(out))
            all_outputs[file][i] = {"Evaluation Outputs": outs,
                                 "Final Rating": sum(out["Rating"] for out in outs) / len(outs)}
            

        try:
            json.dump({"Evaluation": all_outputs[file]},
                    open(os.path.join(args.output_dir, "evaluation_" + name), 'w', encoding='utf-8'), 
                    indent=4, ensure_ascii=False)
        except:
            logger.info(f"Evaluation {file}")
        
    if args.correlation:
        overall_stats = {}
        for file, name in test_files:
            test_dataset = json.load(open(file, encoding='utf-8'))

            num_sys = 1

            if test_dataset[0].get("sys_id", None) is None:
                examples = [{"gold": ex["human_rating"], "pred": ex_out["Final Rating"], "sys_id": None} 
                            for ex, ex_out in zip(test_dataset, all_outputs[file])]
            else:
                examples = [{"gold": ex["human_rating"], "pred": ex_out["Final Rating"], 
                             "seg_id": ex["seg_id"], "sys_id": ex["sys_id"]} 
                            for ex, ex_out in zip(test_dataset, all_outputs[file])]
                
                num_sys = len(set(ex["sys_id"] for ex in examples))
                num_seg = len(set(ex["seg_id"] for ex in examples))
                num_samples = len(set((ex["sys_id"], ex["seg_id"]) for ex in examples))
                assert len(examples) == num_samples and num_seg * num_sys == num_samples, \
                    "Must number of system mul number of segment is equal number of samples"
                examples.sort(key=lambda x: (x["sys_id"], x["seg_id"]))

            gold = [ex['gold'] for ex in examples]
            pred = [ex['pred'] for ex in examples]
            corr = Correlation(num_sys, gold, pred)
            stats = {}
            if examples[0]['sys_id'] is not None:
                for average_by in ['item']:
                    for metric in ['Pearson', 'Spearman', 'Kendall']:
                        stats[f"{average_by}_{metric}"] = getattr(corr, metric)(average_by)[0]
            else:
                for metric in ['Pearson', 'Spearman', 'Kendall']:
                    stats[f"{metric}"] = getattr(corr, metric)()[0]
            
            overall_stats[name] = stats
            try:
                json.dump({"Correlation": stats, "Evaluation": all_outputs[file]}, 
                    open(os.path.join(args.output_dir, "evaluation_" + name), 'w', encoding='utf-8'), 
                    indent=4, ensure_ascii=False)
            except:
                logger.info(f"Evaluation {file}")

        json.dump(overall_stats, open(os.path.join(args.output_dir, "overall_stats.json"), 'w', encoding='utf-8'), indent=4)
    