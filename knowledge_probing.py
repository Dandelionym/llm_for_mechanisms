import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import datetime
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 18


# Replace the model file (downloaded form huggingface) here.
# save_name: identification.
# model_name: must be the same as model name file.
config = {
    "model_path": os.path.join("<path>/huggingface/Mistral-7B-Instruct-v0.2", ""),
    "model_name": "Mistral-7B-Instruct-v0.2",
    "save_name": "unsure_test"
}

temperature_range = [0.1, 0.3, 0.5, 0.7, 0.9]

os.makedirs(f"./probing_results_{config['save_name']}", exist_ok=True)

def initialize_model(model_name):
    """
    Initializes the tokenizer and model for use with DataParallel to leverage multiple GPUs if available.

    Args:
    - model_name: The name or path of the model to be loaded.

    Returns:
    - tokenizer: Initialized tokenizer.
    - model: Initialized model, wrapped with DataParallel and moved to available GPUs.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    return tokenizer, model

# Assume 'config' is defined elsewhere in your code
# tokenizer, model = initialize_model(config['model_path'])
tokenizer, model = initialize_model(config['model_path'])

def get_top_k_multiple_choice_answers(question, choices, top_k=4, temperature_=0.7):
    """
    Evaluates a multiple-choice question using a specified model and returns the top-k choices based on their probabilities.
    This function is designed to work with models that may be running on a GPU.

    Args:
    - question: String, the question to be answered.
    - choices: A list of choices for the question.
    - top_k: The number of top choices to return.

    Returns:
    - A list of tuples containing the top-k choices and their probabilities.
    """


    # Format the prompt with the question and choices
    prompt = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[4]}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate the answer from the model
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.logits / temperature_
        # print("Logits: ", logits, "Source logits: ", outputs.logits)

    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(logits[0, -1, :]) # Focus on the last token for the prediction

    # Encode choice tokens
    choice_tokens = {choice: tokenizer.encode(choice + ")", add_special_tokens=False)[0] for choice in "ABCDE"}

    # Extract probabilities for each choice
    choice_probabilities = {choice: probabilities[token_id].item() for choice, token_id in choice_tokens.items()}

    # Sort the choices based on their probabilities, descending
    sorted_choices = sorted(choice_probabilities.items(), key=lambda x: x[1], reverse=True)

    # Select top-k choices
    top_k_choices = sorted_choices[:top_k]

    return top_k_choices


# Example | case study usage | probing
# You can follow this template to prob the model by your own questions.ß
question_sets = {
    "id_1": {
        "question": "During the colloidal synthesis of chiral gold nanoparticles, which involves using amino acids or peptides as chiral inducers, hydrochloric acid as a gold precursor, ascorbic acid as a reductant, CTAB as a stabilizer, and gold nanoparticles as seeds for inducing heterogeneous nucleation. Which one of the following phenomena is commonly observed during the synthesis?",
        "choices": [
            "Increasing concentrations of chiral molecules always form nanostructures with a stronger chiral response, regardless of other experimental parameters.",
            "The increasing concentration of chiral biomolecules usually results in less complicated morphology, such as well-faceted surface, which gives a stronger chiral response.",
            "Decreasing concentration of chiral molecules always forms nanostructures with a weaker chiral response, regardless of other experimental parameters.",
            "The increasing concentration of chiral biomolecules usually results in more complicated morphology, such as chiral twisting, surface wrinkles or even shattered surfaces without a recognizable chiral pattern.",
        ]
    },
    "id_2": {
            "question": "During the seed-mediated colloidal synthesis of gold nanoparticles, with hydrochloric acid as the gold precursor, ascorbic acid as the reductant, and CTAB as the stabilizer, the seed nanoparticles are anisotropic in shape (having varied crystal facets and sites with different curvatures). Which one of the following phenomena is commonly observed during the overgrowth process?",
            "choices": [
                "When the growth rate or supersaturation of the growth solution is relatively high, the deposition of gold atoms would preferentially occur at sites with high surface energies, such as areas with pronounced curvature, defects, significant strain, or minimal passivation by capping agents. The resulting nanostructures usually deviate from equilibrium crystal shape.",
                "When the growth rate or supersaturation of the growth solution is relatively low, the deposition of gold atoms would preferentially occur at sites with high surface energies, such as areas with pronounced curvature, defects, significant strain, or minimal passivation by capping agents. The resulting nanostructures usually deviate from equilibrium crystal shape.",
                "When the growth rate or supersaturation of the growth solution is relatively high, the deposition of gold atoms would preferentially occur at sites with high surface energies, such as areas with pronounced curvature, defects, significant strain, or minimal passivation by capping agents. The resulting nanostructures are usually in equilibrium crystal shape with well-faceted surface.",
                "When the growth rate or supersaturation of the growth solution is relatively low, the deposition of gold atoms would preferentially occur at sites with low surface energies, such as areas with pronounced curvature, defects, significant strain, or minimal passivation by capping agents. The resulting nanostructures are usually in equilibrium crystal shape with well-faceted surface.",
            ]
        },
    "id_3": {
            "question": "Over the last decade, high-index-facet and high-surface-energy nanocrystals have drawn increasing interest in electrocatalysis, photocatalysis, and heterogeneous catalysis. During the colloidal synthesis of gold nanoparticles with high-index facets with gold precursor, reductant, stabilizer (or capping agent), which one of the following statements is true?",
            "choices": [
                "The evolution of high-index facets usually occurs when the growth rate or supersaturation of the growth solution is relatively high, which can be achieved by increasing pH and decreasing the ratio of reductant to gold precursor.",
                "The evolution of high-index facets usually occurs when the growth rate or supersaturation of the growth solution is relatively low, which can be achieved by decreasing pH and increasing the ratio of reductant to gold precursor.",
                "The evolution of high-index facets usually occurs when the growth rate or supersaturation of the growth solution is relatively high, which can be achieved by increasing pH and increasing the ratio of reductant to gold precursor.",
                "The evolution of high-index facets usually occurs when the growth rate or supersaturation of the growth solution is relatively low, which can be achieved by decreasing pH and decreasing the ratio of reductant to gold precursor.",
            ]
        }
    }

all_results_to_save = []

for temperature in temperature_range:
    for idx, question_opts in question_sets.items():
        question = question_opts["question"]
        choices = question_opts["choices"]

        top_k_choices = get_top_k_multiple_choice_answers(question, choices, temperature_=temperature, top_k=5)

        print("Temperature: ", temperature)
        print("Question ID: ", idx)

        # Print results.
        print(f"\nTop-k Choices with Probabilities for test in temperature [{temperature}]:")
        for choice, probability in top_k_choices:
            print(f"Choice: {choice}, Probability: {probability}")

        # Save results.
        os.makedirs("./probing_results", exist_ok=True)
        with open(f"./probing_results_{config['save_name']}/{config['save_name']}__{config['model_name']}___{temperature}___Q{idx}___probing_output.txt", "w") as f:
            results = {
                "model_name": config['model_name'],
                "temperature": temperature,
                "question": question,
                "choices": choices,
                "probability": {k: v for k, v in top_k_choices}
            }
            json.dump(results, fp=f, indent=4)
            all_results_to_save.append(results)
            print(f"Success. Info: {config['save_name']}__{config['model_name']}__{temperature}__Q{idx}")

with open(f"./{config['save_name']}__probing_results.json", mode='w') as g:
    json.dump(all_results_to_save, fp=g, indent=4)

print("Grouped results successfully.")
print("Saved file: ", f"./probing_results_{config['save_name']}/{config['save_name']}__{config['model_name']}___<temperature>___<QID>___probing_output.txt")
print("Time: ", datetime.datetime.now())
