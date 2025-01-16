import os
import re
import string
import yaml
from collections import defaultdict 
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import tqdm
from ollama import chat
from ollama import ChatResponse
import textwrap

from generators import Generator, Data
from rendering import autorender_data
from prompting import load_prompts_and_renders

from openai import OpenAI

client = OpenAI(api_key="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")


EXTRACT_NUMERIC_PROMPT = """You are a helpful assistant that extracts numerical values from text. Given a response, return ONLY the primary numerical value being discussed.

If the answer contains approximations like "approximately 0.5" or "about 100", return that number.
If no clear numerical answer is found, return None.

Response to analyze:
{text}

Return only the number, nothing else. Do not include any text, units, or explanations."""

STATISTICAL_REASONING_PROMPT = """You are a helpful assistant analyzing statistical data visualizations. Please carefully examine the {render_type} and answer the following question:

{question}

After your explanation, clearly state your final answer in one word on a new line starting with "ANSWER:".\n\nResponse: """

DEGREE_MAP = {
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "quartic": 4,
    "quintic": 5,
    "sextic": 6,
    "septic": 7,
    "octic": 8,
    "nonic": 9, 
    "decic": 10
}

MODEL_MAP = {
    "llama3.2-vision": "Llama 3.2 Vision 11b",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o": "GPT-4o",
}


def extract_numeric_answer(response_text, answer_type=float):
    # Find answer pattern and capture the content after it
    pattern = r'(?:\*\*)?(?:final )?answer:(?:\*\*)?\s*(.*?)(?:\n|$)'
    matches = list(re.finditer(pattern, response_text.lower().strip()))
    
    if matches:
        # Get the last match
        match = matches[-1]
        answer = match.group(1).strip().replace('**', '')
        if "unknown" in answer.lower() or "possibly" in answer.lower():
            return "unknown"
        if "impossible" in answer.lower():
            return "impossible"
        # Clean up the answer, keeping minus sign for negatives
        answer = ''.join(char for char in answer if char.isdigit() or char in ['.', '%', '/', '-'])
        if answer and answer[-1] == '.':
            answer = answer[:-1]
        if "%" in answer:
            return float(answer.replace("%", "")) / 100
        try:
            if answer in DEGREE_MAP:
                answer = DEGREE_MAP[answer]
                return answer
            return float(eval(answer))
        except:
            return None

    return None

def extract_qualitative_answer(response_text, options):
    # Find answer pattern and capture the content after it
    pattern = r'(?:\*\*)?(?:final )?answer:(?:\*\*)?\s*(.*?)(?:\n|$)'
    matches = list(re.finditer(pattern, response_text.lower().strip()))
    
    if matches:
        # Get the last match
        match = matches[-1]
        answer = match.group(1).strip().replace('**', '')
        # Convert answer and options to lowercase for case-insensitive matching
        answer_lower = answer.lower()
        answer_lower = ''.join(char for char in answer_lower if char not in string.punctuation)
        try:
            options_lower = [opt.lower() if isinstance(opt, str) else opt for opt in options]
        except TypeError:
            options_lower = [opt for opt in options]
        
        # Sort options by length in descending order to check longer options first
        sorted_pairs = sorted(zip(options, options_lower), key=lambda x: len(str(x[1])), reverse=True)
        # Try to match the answer with one of the options, checking longer options first
        for opt, opt_lower in sorted_pairs:
            if opt_lower == answer_lower:
                return opt

        return answer_lower

    # If no match found, return None
    return None


def evaluate_model(model_name='llama3.2-vision', dataset_names=None, prompts=None, accuracy_threshold=0.9):

    def get_model_response(message):
        if model_name == "llama3.2-vision":
            response: ChatResponse = chat(model='llama3.2-vision', messages=[message])
            return response.message.content
        elif model_name == "gpt-4o-mini":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[message],
            )
            return response.choices[0].message.content
    
    results = {}
    get_results = []

    if dataset_names is not None:
        for dataset_name in dataset_names:
            if os.path.exists(f'results/{model_name}/{dataset_name}_results.yaml'):
                with open(f'results/{model_name}/{dataset_name}_results.yaml', 'r') as f:
                    results.update(yaml.load(f, Loader=yaml.FullLoader))
            else:
                assert prompts is not None, "Prompts must be provided if results file does not exist."
                get_results.append(dataset_name)
    else:
        assert prompts is not None, "Prompts must be provided if dataset_names is not provided."
        if os.path.exists(f'results/{model_name}/all_prompts_results.yaml'):
            with open(f'results/{model_name}/all_prompts_results.yaml', 'r') as f:
                results.update(yaml.load(f, Loader=yaml.FullLoader))
        else:
            get_results.append("")

    total_prompts = 0
    total_errors = 0
    
    for dataset_name in get_results:
        if dataset_names is not None:
            dataset_prompts = [p for p in prompts if p['dataset'] == dataset_name]
        else:
            dataset_prompts = prompts

        total_prompts += len(dataset_prompts)
        errors = []
        
        for q in tqdm.tqdm(dataset_prompts, desc="Evaluating...", total=len(dataset_prompts)):
            render_id = q['render_id']
            results[render_id] = {
                'metadata': {
                    'prompt': q['prompt'],
                    'answer': q['answer'],
                    'prompt_type': q['prompt_type'],
                    'prompt_tier': q['prompt_tier'],
                    'options': q.get('options', None),
                    'dataset': q['dataset'],
                }
            }

            for render_type in ['plot', 'table']:
                try:
                    question = f"Question: According to the {render_type}, {q['prompt']}"

                    if render_type == 'table':
                        with open(q['renders']['table'], 'r') as f:
                            table_string = f.read()
                        question = f"Table:\n{table_string}\n\n{question}"
                        prompt = STATISTICAL_REASONING_PROMPT.format(render_type=render_type, question=question)
                        response = get_model_response({"role": "user", "content": prompt})
                        
                    else:
                        prompt = STATISTICAL_REASONING_PROMPT.format(render_type=render_type, question=question)
                        response = get_model_response({"role": "user", "content": prompt, "images": [q['renders']['plot']]})

                    results[render_id][render_type] = {}
                    
                    if q['prompt_type'] == 'quantitative':
                        model_answer = extract_numeric_answer(response)
                        
                        if model_answer is not None and model_answer != "unknown" and model_answer != "impossible":
                            qanswer = float(f"{float(q['answer']):.2f}")
                            
                            #is_correct = float(model_answer) == qanswer
                            try:
                                #relative_acc = (qanswer - abs(float(model_answer) - qanswer)) / qanswer
                                relative_acc = 1.0 - (abs(qanswer - float(model_answer)) / qanswer)
                            except ZeroDivisionError:
                                relative_acc = 1.0 if is_correct else 0.0  # TODO: Fix accuracy for 0 answer
                            is_correct = relative_acc >= accuracy_threshold
                            
                            results[render_id][render_type].update({
                                'is_correct': is_correct,
                                'relative_accuracy': relative_acc
                            })
                        else:
                            results[render_id][render_type].update({
                                'is_correct': False,
                                'relative_accuracy': 0
                            })
                    else:
                        model_answer = extract_qualitative_answer(response, options=q['options'])
                        
                        if model_answer is not None:
                            is_correct = model_answer.lower() == q['answer'].replace("'", "").lower()
                            results[render_id][render_type].update({
                                'is_correct': is_correct,
                                'relative_accuracy': 1.0 if is_correct else 0.0
                            })
                        else:
                            results[render_id][render_type].update({
                                'is_correct': False, 
                                'relative_accuracy': 0.0
                            })
                    # Store response data
                    results[render_id][render_type].update({
                        'long_response': response,
                        'response_length': len(response),
                        'extracted_answer': model_answer,
                    })
                except Exception as e:
                    del results[render_id]
                    errors.append((render_id, render_type, e))
                    break

        if len(errors) > 0:
            total_errors += len(errors)
            for render_id, render_type, e in errors:
                print(f"Error for {render_id} ({render_type}): {e}")

        print(f"Answered {total_prompts - total_errors} out of {total_prompts} prompts without problems ({total_errors} errors raised).")

    return results


def analyze_results(results_dict, model_name, dataset_names, accuracy_threshold=0.9):
    """Analyze model results and compute various metrics."""

    # Helper function to compute metrics for a subset of results
    def compute_metrics(results_subset):
        rel_accs = [r.get('relative_accuracy', 0) for r in results_subset if 'relative_accuracy' in r]
        resp_lens = [r.get('response_length', 0) for r in results_subset if 'response_length' in r]
        null_count = sum(1 for r in results_subset if r.get('extracted_answer') is None)
        
        return {
            'accuracy': sum(1 for x in rel_accs if x >= accuracy_threshold) / len(rel_accs) if rel_accs else 0,
            'relative_accuracies': rel_accs,
            'avg_response_length': sum(resp_lens) / len(resp_lens) if resp_lens else 0,
            'response_lengths': resp_lens,
            'null_answers': null_count,
            'total_samples': len(rel_accs)
        }
    
    analyses = {
        'overall': {},
        'by_render_type': {'plot': {}, 'table': {}},
        'by_prompt_tier': {},
        'by_plot_type': {},
        'by_prompt_type': {'qualitative': {}, 'quantitative': {}}
    }
    
    # Initialize lists to store metrics
    all_rel_accuracies = []
    all_resp_lengths = []
    all_null_count = 0
    total_count = 0

    # Collect all results
    all_results = []
    for qid, qdata in results_dict.items():
        for render_type in ['plot', 'table']:
            if render_type in qdata:
                result = qdata[render_type]
                result['prompt_tier'] = qdata['metadata']['prompt_tier']
                result['prompt_type'] = qdata['metadata']['prompt_type']
                result['render_type'] = render_type
                #result['dataset'] = dataset_name  TODO: FIX
                all_results.append(result)
                
                # Update overall metrics
                all_rel_accuracies.append(result.get('relative_accuracy', 0))
                all_resp_lengths.append(result.get('response_length', 0))
                if result.get('extracted_answer') is None:
                    all_null_count += 1

    # Compute overall metrics
    analyses['overall'] = {
        'accuracy': sum(1 for x in all_rel_accuracies if x >= accuracy_threshold) / len(all_rel_accuracies),
        'relative_accuracies': all_rel_accuracies,
        'avg_response_length': sum(all_resp_lengths) / len(all_resp_lengths),
        'response_lengths': all_resp_lengths,
        'null_answers': all_null_count,
        'total_samples': total_count
    }

    # Group by render type
    for render_type in ['plot', 'table']:
        subset = [r for r in all_results if r['render_type'] == render_type]
        analyses['by_render_type'][render_type] = compute_metrics(subset)

    # Group by prompt tier
    prompt_tiers = set(r['prompt_tier'] for r in all_results)
    for tier in prompt_tiers:
        subset = [r for r in all_results if r['prompt_tier'] == tier]
        analyses['by_prompt_tier'][tier] = compute_metrics(subset)

    # Group by plot type
    plot_types = set(r['render_type'] for r in all_results)
    for plot_type in plot_types:
        subset = [r for r in all_results if r['render_type'] == plot_type]
        analyses['by_render_type'][plot_type] = compute_metrics(subset)

    # Group by prompt type
    for prompt_type in ['qualitative', 'quantitative']:
        subset = [r for r in all_results if r['prompt_type'] == prompt_type]
        analyses['by_prompt_type'][prompt_type] = compute_metrics(subset)

    # Save analyses to file
    if len(dataset_names) == 1:
        dataset_name = dataset_names[0]
    else:
        dataset_name = "all"
    os.makedirs(f'analyses/{model_name}', exist_ok=True)
    with open(f'analyses/{model_name}/{dataset_name}_analysis.yaml', 'w') as f:
        yaml.dump(analyses, f)
        
    return analyses


def analyze_prompt_distribution(prompts):
    """
    Analyze the distribution of prompts across different categories.
    
    Args:
        prompts: List of prompt dictionaries containing metadata
        
    Returns:
        dict: Statistics about the prompt distribution
    """
    total_prompts = len(prompts)
    
    # Count prompt tiers
    tier_counts = {}
    for prompt in prompts:
        tier = prompt['prompt_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
    # Count render types
    render_counts = {}
    for prompt in prompts:
        render_type = prompt['render_type']
        render_counts[render_type] = render_counts.get(render_type, 0) + 1
        
    # Count prompt types
    prompt_type_counts = {
        'qualitative': 0,
        'quantitative': 0
    }
    for prompt in prompts:
        prompt_type = prompt['prompt_type']
        prompt_type_counts[prompt_type] += 1
        
    return {
        'total_prompts': total_prompts,
        'by_tier': tier_counts,
        'by_render_type': render_counts,
        'by_prompt_type': prompt_type_counts
    }


def print_prompt_distribution(distribution):
    """
    Print the prompt distribution statistics in a readable format.
    
    Args:
        distribution: Dict containing prompt distribution statistics from analyze_prompt_distribution()
    """
    print("\nPrompt Distribution Analysis")
    print("===========================")
    
    print(f"\nTotal Prompts: {distribution['total_prompts']}")
    
    print("\nBy Tier:")
    for tier, count in distribution['by_tier'].items():
        percentage = (count / distribution['total_prompts']) * 100
        print(f"  {tier}: {count} ({percentage:.1f}%)")
        
    print("\nBy Render Type:") 
    for render_type, count in distribution['by_render_type'].items():
        percentage = (count / distribution['total_prompts']) * 100
        print(f"  {render_type}: {count} ({percentage:.1f}%)")
        
    print("\nBy Prompt Type:")
    for prompt_type, count in distribution['by_prompt_type'].items():
        percentage = (count / distribution['total_prompts']) * 100
        print(f"  {prompt_type}: {count} ({percentage:.1f}%)")


def process_datasets(dataset_names, clear_existing=True, subset=None):
    """
    Process multiple datasets by loading/generating data, rendering, and combining prompts.
    
    Args:
        dataset_names: List of dataset names to process
        clear_existing: Whether to clear existing renders. If True, only clears for first dataset.
        subset: A dictionary giving a property name and value to filter prompts by
    
    Returns:
        List of all prompts combined from the datasets
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        
    all_prompts = []
    
    for i, dataset_name in enumerate(dataset_names):
        # Load or generate dataset
        if not os.path.exists(f"data/metadata/datasets/{dataset_name}.yaml"):
            generator = Generator(config=f"data/generator_configs/generator_{dataset_name.split('_')[-1]}.yaml")
            data = generator(np.random.randint(50, 200))
            data.save(dataset_name)
        else:
            data = Data.load(dataset_name)
            
        # Render with clear_existing only for first dataset if enabled
        if clear_existing:
            should_clear = i == 0
            autorender_data(data, clear_existing=should_clear)

        # Load prompts for this dataset
        dataset_prompts = load_prompts_and_renders(dataset_name, load_renders_as_paths=True, clear_existing=clear_existing)[dataset_name]

        if subset is not None:
            dataset_prompts = [p for p in dataset_prompts if all(p[k] == v for k, v in subset.items())]

        all_prompts.extend(dataset_prompts)
        
    return all_prompts


def get_balanced_prompts(prompts, target_per_category=200):
    """
    Balance prompts across different categories to have roughly equal distribution.
    
    Args:
        prompts: List of prompts to balance
        target_per_category: Target number of prompts per category
        
    Returns:
        List of balanced prompts
    """
    # Group prompts by different categorizations
    by_prompt_type = defaultdict(list)
    by_render_type = defaultdict(list)
    by_tier = defaultdict(list)
    
    for prompt in prompts:
        by_prompt_type[prompt['prompt_type']].append(prompt)
        by_render_type[prompt['render_type']].append(prompt)
        by_tier[prompt['prompt_tier']].append(prompt)
    
    # Sample equal numbers from each category
    balanced_prompt_type = []
    for ptype in by_prompt_type:
        balanced_prompt_type.extend(
            np.random.choice(by_prompt_type[ptype], 
                           size=min(len(by_prompt_type[ptype]), target_per_category),
                           replace=False).tolist()
        )
        
    balanced_render_type = []
    for rtype in by_render_type:
        balanced_render_type.extend(
            np.random.choice(by_render_type[rtype],
                           size=min(len(by_render_type[rtype]), target_per_category),
                           replace=False).tolist()
        )
        
    balanced_tier = []
    for tier in by_tier:
        balanced_tier.extend(
            np.random.choice(by_tier[tier],
                           size=min(len(by_tier[tier]), target_per_category),
                           replace=False).tolist()
        )
    
    # Take union of balanced sets and update table paths
    balanced_prompts = []
    seen_prompts = set()
    for prompt in balanced_prompt_type + balanced_render_type + balanced_tier:
        if prompt['prompt'] not in seen_prompts:
            seen_prompts.add(prompt['prompt'])

            if 'renders' in prompt and 'table' in prompt['renders']:
                prompt['renders']['table'] = f"/home/alexart/vlm-benchmark/data/tables/{prompt['render_id']}.txt"
            if isinstance(prompt["answer"], (np.int64, np.float64)):
                prompt["answer"] = prompt["answer"].item()
            elif isinstance(prompt["answer"], np.ndarray):
                prompt["answer"] = prompt["answer"].tolist()
            elif isinstance(prompt["answer"], np.str_):
                prompt["answer"] = str(prompt["answer"])

            if "options" in prompt:
                converted_options = []
                for opt in prompt["options"]:
                    if isinstance(opt, (np.int64, np.float64, np.int32, np.float32)):
                        converted_options.append(opt.item())
                    elif isinstance(opt, np.ndarray):
                        converted_options.append(opt.tolist())
                    elif isinstance(opt, np.str_):
                        converted_options.append(str(opt))
                    else:
                        converted_options.append(opt)
                prompt["options"] = converted_options
            balanced_prompts.append(prompt)
    
    # Save balanced prompts
    os.makedirs('data/prompts', exist_ok=True)
    with open('data/prompts/all_prompts.yaml', 'w') as f:
        yaml.dump(balanced_prompts, f)
        
    return balanced_prompts


def get_diagnostic_prompts(prompts, target_per_category=1, vary_by=None):
    """
    Create a diagnostic set of prompts by taking one of each type of task.
    """
    if vary_by is None:
        vary_by = ["prompt_type", "prompt_tier", "render_type"]

    # Initialize dictionary to store all possible values for each category
    category_values = {category: set() for category in vary_by}
    
    # Collect all possible values for each category
    for prompt in prompts:
        for category in vary_by:
            if category in prompt:
                category_values[category].add(prompt[category])
    
    # Convert sets to sorted lists for deterministic behavior
    category_values = {k: sorted(list(v)) for k,v in category_values.items()}
    
    # Initialize diagnostic prompts list
    diagnostic_prompts = []
    
    # Get all combinations of category values
    category_combinations = list(itertools.product(*[category_values[cat] for cat in vary_by]))
    
    # For each combination, find matching prompts and randomly select one
    for combination in category_combinations:
        matching_prompts = []
        for prompt in prompts:
            matches = True
            for cat, val in zip(vary_by, combination):
                if prompt.get(cat) != val:
                    matches = False
                    break
            if matches:
                matching_prompts.append(prompt)
                
        if matching_prompts:
            # Randomly select one matching prompt
            diagnostic_prompts += np.random.choice(matching_prompts, size=target_per_category, replace=False).tolist()
        else:
            print(f"No matching prompts found for {combination}")
    
    return diagnostic_prompts


def plot_accuracy(data, category, title, model_name):
    plt.figure(figsize=(10, 6))
    
    # Extract accuracies
    if category == 'prompt_tier':
        labels = ["retrieval", "arithmetic", "boolean", "inference"]
    else:
        labels = list(data.keys())
    acc = [data[l]['accuracy'] for l in labels]
    
    x = range(len(labels))
    width = 0.35
    colors = sns.color_palette("husl", len(x))

    plt.bar(x, acc, width, color=colors)

    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for {MODEL_MAP[model_name]} by {title}')
    plt.xticks(x, [l.capitalize() for l in labels], rotation=0, fontsize=18)
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'analyses/{model_name}/accuracy_{category}.png')
    plt.close()


def plot_relative_accuracy(data, category, title, model_name):
    plt.figure(figsize=(10, 6))

    # Extract relative accuracies
    if category == 'prompt_tier':
        labels = ["retrieval", "arithmetic", "boolean", "inference"]
    else:
        labels = list(data.keys())
    acc = [data[l]['relative_accuracies'] for l in labels]

    colors = sns.color_palette("husl", len(labels))

    for i, label in enumerate(labels):
        # Create histogram for each label's relative accuracies
        plt.hist(acc[i], bins=20, alpha=0.5, label=label.capitalize(), color=colors[i])

    plt.xlabel('Relative Accuracy')
    plt.ylabel('Count')
    plt.title(f'Distribution of Relative Accuracies for {MODEL_MAP[model_name]} by {title}')
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(f'analyses/{model_name}/relative_accuracy_{category}.png')
    plt.close()


def plot_response_length(data, category, title, model_name):
    plt.figure(figsize=(10, 6))

    # Extract response lengths
    if category == 'prompt_tier':
        labels = ["retrieval", "arithmetic", "boolean", "inference"]
    else:
        labels = list(data.keys())
    acc = [data[l]['response_lengths'] for l in labels]

    colors = sns.color_palette("husl", len(labels))

    for i, label in enumerate(labels):
        # Create histogram for each label's response lengths
        plt.hist(acc[i], bins=20, alpha=0.5, label=label.capitalize(), color=colors[i])

    plt.xlabel('Response Length')
    plt.ylabel('Count')
    plt.title(f'Distribution of Response Lengths for {MODEL_MAP[model_name]} by {title}')
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(f'analyses/{model_name}/response_length_{category}.png')
    plt.close()


def plot_analysis(analyses, model_name):
    for category in ['render_type', 'prompt_tier', 'prompt_type']:
        plot_accuracy(analyses[f'by_{category}'], category, category.replace("_", " ").capitalize(), model_name)
        plot_relative_accuracy(analyses[f'by_{category}'], category, category.replace("_", " ").capitalize(), model_name)
        plot_response_length(analyses[f'by_{category}'], category, category.replace("_", " ").capitalize(), model_name)


if __name__ == "__main__":
    dataset = ["dataset_17", "dataset_5", "dataset_20", "dataset_21", "dataset_16", "dataset_7"]
    model = "llama3.2-vision"
    diagnostic = False

    #with open(f'results/{model}/all_prompts_results2.yaml', 'r') as f:
    #    results_llama = yaml.load(f, Loader=yaml.FullLoader)

    with open('data/prompts/all_prompts.yaml', 'r') as f:
        balanced_prompts = yaml.load(f, Loader=yaml.FullLoader)

    distribution = analyze_prompt_distribution(balanced_prompts)
    print_prompt_distribution(distribution)

    results = evaluate_model(model_name=model, prompts=balanced_prompts)

    with open(f'results/{model}/all_prompts_results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    analyses = analyze_results(results, model_name=model, dataset_names="all")
    print(f"OVERVIEW for {MODEL_MAP[model]}:")
    print(f"\t- Overall Accuracy: {analyses['overall']['accuracy']}")
    print(f"\t- Average Response Length: {analyses['overall']['avg_response_length']}")
    print(f"\t- Number Null Answers: {analyses['overall']['null_answers']}")
    plot_analysis(analyses, model)

    """
    prompts = process_datasets(dataset, clear_existing=True)

    if diagnostic:
        print("Running diagnostic set...")
        diagnostic_prompts = get_diagnostic_prompts(prompts, target_per_category=1)
        results = evaluate_model(model_name=model, prompts=diagnostic_prompts)
        with open('diagnostic_results.yaml', 'w') as f:
            yaml.dump(results, f)
    else:
        if os.path.exists('data/prompts/all_prompts.yaml'):
            with open('data/prompts/all_prompts.yaml', 'r') as f:
                balanced_prompts = yaml.load(f, Loader=yaml.FullLoader)
        else:
            balanced_prompts = get_balanced_prompts(prompts, target_per_category=10)
        
        # Check that the prompts are balanced
        distribution = analyze_prompt_distribution(balanced_prompts)
        print_prompt_distribution(distribution)

        results = evaluate_model(model_name=model, prompts=balanced_prompts)

        os.makedirs(f'results/{model}', exist_ok=True)
        with open(f'results/{model}/all_prompts_results.yaml', 'w') as f:
            yaml.dump(results, f)
    """


#dataset_names = ["dataset_17", "dataset_5", "dataset_20", "dataset_21", "dataset_16", "dataset_7"]
"""
prompts = process_datasets(dataset_names, clear_existing=True)
balanced_prompts = balance_prompts(prompts, target_per_category=10)
distribution = analyze_prompt_distribution(balanced_prompts)
print_prompt_distribution(distribution)

print(balanced_prompts[0])
"""

"""
results_llama = evaluate_model(model_name="llama3.2-vision", dataset_names=dataset_names)
analyses_llama = analyze_results(results_llama, model_name="llama3.2-vision", dataset_names="all")

print(analyses_llama['by_render_type'])

def plot_comparison(llama_data, category, title):
    plt.figure(figsize=(10, 6))
    
    # Extract accuracies for each model
    llama_acc = [data.get('accuracy', 0) for _, data in llama_data.items()]
    print(llama_acc)
    labels = list(llama_data.keys())
    
    x = range(len(labels))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], llama_acc, width, color='skyblue')
    
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for Llama 3.2 Vision 11b by {title}')
    plt.xticks(x, labels, rotation=0, fontsize=18)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'analyses/comparison_{category}.png')
    plt.close()

# Generate comparison plots
plot_comparison(
    analyses_llama['by_render_type'], 
    'render_type',
    'Render Type'
)

plot_comparison(
    analyses_llama['by_prompt_tier'],
    'prompt_tier',
    'Prompt Tier'
)

plot_comparison(
    analyses_llama['by_prompt_type'],
    'prompt_type', 
    'Prompt Type'
)

#data = Data.load("dataset_17")
#autorender_data(data, clear_existing=False)
"""

"""
dataset_names = ["dataset_5", "dataset_20", "dataset_21", "dataset_16", "dataset_7"]
prompts = process_datasets(dataset_names, clear_existing=True)

print(f"Loaded {len(prompts)} questions.")

distribution = analyze_prompt_distribution(prompts)
print_prompt_distribution(distribution)

balanced_prompts = balance_prompts(prompts, target_per_category=10)
"""

"""
balanced_prompts = load_prompts_and_renders("all", load_renders_as_paths=True)
distribution = analyze_prompt_distribution(balanced_prompts["all"])
print_prompt_distribution(distribution)

#dataset_name = "dataset_20"
#model = "llama3.2-vision"
model = "gpt-4o-mini"
results_gpt = evaluate_model(model_name=model, prompts=balanced_prompts, dataset_name="all")

for render_id, qdata in results_gpt.items():
    for render_type in ['plot', 'table']:
        response = qdata[render_type]['long_response']
        q = qdata['metadata']

        if q['prompt_type'] == 'quantitative':
            model_answer = extract_numeric_answer(response)

            results_gpt[render_id][render_type]['extracted_answer'] = model_answer
            
            if model_answer is not None:
                try:
                    qanswer = float(f"{float(q['answer']):.2f}")
                except ValueError:
                    qanswer = float(f"{float(q['answer'].split(',')[0].replace('(', '')):.2f}")
                is_correct = float(model_answer) == qanswer
                try:
                    relative_acc = (qanswer - abs(float(model_answer) - qanswer)) / qanswer
                except ZeroDivisionError:
                    relative_acc = 1.0 if is_correct else 0.0
                
                results_gpt[render_id][render_type].update({
                    'is_correct': is_correct,
                    'relative_accuracy': relative_acc
                })
            else:
                results_gpt[render_id][render_type].update({
                    'is_correct': False,
                    'relative_accuracy': 0
                })
        else:
            model_answer = extract_qualitative_answer(response, options=q['options'])
            results_gpt[render_id][render_type]['extracted_answer'] = model_answer
            
            if model_answer is not None:
                is_correct = model_answer == q['answer']
                results_gpt[render_id][render_type].update({
                    'is_correct': is_correct,
                    'relative_accuracy': 1.0 if is_correct else 0.0
                })
            else:
                results_gpt[render_id][render_type].update({
                    'is_correct': False, 
                    'relative_accuracy': 0.0
                })

analyses_gpt = analyze_results(results_gpt, model_name=model, dataset_name="all")

model_name = "llama3.2-vision"

results_llama = evaluate_model(model_name=model, prompts=balanced_prompts, dataset_name="all")

for render_id, qdata in results_llama.items():
    for render_type in ['plot', 'table']:
        response = qdata[render_type]['long_response']
        q = qdata['metadata']

        if q['prompt_type'] == 'quantitative':
            model_answer = extract_numeric_answer(response)

            results_llama[render_id][render_type]['extracted_answer'] = model_answer
            
            if model_answer is not None:
                try:
                    qanswer = float(f"{float(q['answer']):.2f}")
                except ValueError:
                    qanswer = float(f"{float(q['answer'].split(',')[0].replace('(', '')):.2f}")
                is_correct = float(model_answer) == qanswer
                try:
                    relative_acc = (qanswer - abs(float(model_answer) - qanswer)) / qanswer
                except ZeroDivisionError:
                    relative_acc = 1.0 if is_correct else 0.0
                
                results_llama[render_id][render_type].update({
                    'is_correct': is_correct,
                    'relative_accuracy': relative_acc
                })
            else:
                results_llama[render_id][render_type].update({
                    'is_correct': False,
                    'relative_accuracy': 0
                })
        else:
            model_answer = extract_qualitative_answer(response, options=q['options'])
            results_llama[render_id][render_type]['extracted_answer'] = model_answer
            
            if model_answer is not None:
                is_correct = model_answer == q['answer']
                results_llama[render_id][render_type].update({
                    'is_correct': is_correct,
                    'relative_accuracy': 1.0 if is_correct else 0.0
                })
            else:
                results_llama[render_id][render_type].update({
                    'is_correct': False, 
                    'relative_accuracy': 0.0
                })

analyses_llama = analyze_results(results_llama, model_name=model, dataset_name="all")


# Create plots comparing GPT and LLaMA analyses
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(llama_data, gpt_data, category, title):
    plt.figure(figsize=(10, 6))
    
    # Extract accuracies for each model
    llama_acc = [data.get('accuracy', 0) for _, data in llama_data.items()]
    gpt_acc = [data.get('accuracy', 0) for _, data in gpt_data.items()]
    labels = list(llama_data.keys())
    
    x = range(len(labels))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], llama_acc, width, label='LLaMA', color='skyblue')
    plt.bar([i + width/2 for i in x], gpt_acc, width, label='GPT', color='lightgreen')
    
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison by {title}')
    plt.xticks(x, labels, rotation=0, fontsize=18)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'analyses/comparison_{category}.png')
    plt.close()

# Generate comparison plots
plot_comparison(
    analyses_llama['by_render_type'], 
    analyses_gpt['by_render_type'],
    'render_type',
    'Render Type'
)

plot_comparison(
    analyses_llama['by_prompt_tier'],
    analyses_gpt['by_prompt_tier'], 
    'prompt_tier',
    'Prompt Tier'
)

plot_comparison(
    analyses_llama['by_prompt_type'],
    analyses_gpt['by_prompt_type'],
    'prompt_type', 
    'Prompt Type'
)

def plot_response_lengths(data, model_name, category, title):
    plt.figure(figsize=(10, 6))
    
    # Extract response lengths for each category
    category_lengths = {}
    for category_name, category_data in data.items():
        category_lengths[category_name] = category_data['response_lengths']
    
    # Plot histogram for each category
    colors = ['salmon', 'orange', 'purple', 'skyblue']
    for i, (category_name, lengths) in enumerate(category_lengths.items()):
        plt.hist(lengths, bins=30, alpha=0.5, label=category_name, 
                color=colors[i % len(colors)])
    
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} Response Lengths by {title}')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'analyses/response_lengths_{model_name.lower()}_{category}.png')
    plt.close()

# Generate response length plots
for model_name, data in [('LLaMA', analyses_llama), ('GPT', analyses_gpt)]:
    plot_response_lengths(
        data['by_render_type'],
        model_name,
        'render_type',
        'Render Type'
    )
    
    plot_response_lengths(
        data['by_prompt_tier'],
        model_name, 
        'prompt_tier',
        'Prompt Tier'
    )
    
    plot_response_lengths(
        data['by_prompt_type'],
        model_name,
        'prompt_type',
        'Prompt Type'
    )

"""