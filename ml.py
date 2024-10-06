import csv
import time
import random
from langchain.llms import Ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM



# Function to generate text using Ollama models and measure time taken
def generate_text_with_ollama(model_name, model, prompt1):
    template = """Complete the following text: {text}"""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            # ("system","Complete the following sentence: "),
            ("system","You are an assistant for sentence completion. Please complete the sentence based on your general knowledge, and do not request current or real-time data."),
            # ("system", "You are a helpful assistant that completes the following text."),
            # ("user", "{input}")
            ("human", 'Complete the following sentence: "{input}"')

        ]
    )    # print("model_name",model_name,model)
    model = OllamaLLM(model=model_name, num_predict=len(prompt1)+20)

    chain = prompt_template | model

    start_time = time.time()  # Start time
    # response = model(prompt)
    # print(prompt1,"prompt",chain)
    response = chain.invoke({"input": prompt1})
    # print(response)
    end_time = time.time()  # End time
    time_taken = end_time - start_time  # Calculate the time taken
    return prompt1, response, model_name, time_taken

# Load all required Ollama models
models = {
    "LLaMA 3.2 (1B)": ("llama3.2:1b"),
    "Gemma 2b": ("gemma2:2b"),
    "Qwen 2.5 (3B)": ("qwen2.5:3b"),
    "Phi-3.5 (3B)": ("phi3.5"),
    "TinyLlama (1.1B)": ("tinyllama"),
    "nemotron-mini (1B)": ("nemotron-mini"),
}

# Path to the input CSV with truncated texts
# input_csv_file = "/Users/apple/Downloads/truncated_wikipedia_texts.csv"

# Path to the output CSV where triplets will be stored
output_csv_file = "llm_outputs_ollama_with_time.csv"

import json

train_file_path = '/Users/apple/Downloads/hellaswag_train.jsonl'
# val_file_path = 'hellaswag_val.jsonl'

truncated_texts=[]

# Function to load HellaSwag dataset from a .jsonl file
def load_hellaswag_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
            truncated_texts.append(data[-1]['ctx'])
    return data

# Load the training and validation datasets
train_data = load_hellaswag_dataset(train_file_path)


# Accessing specific fields in the data
context = train_data[0]['ctx']  # Full context
endings = train_data[0]['endings']  # List of possible endings
correct_label = train_data[0]['label']  # Index of the correct ending

data = []

def process_text(prompt):
    results = []
    for  model,model_name in models.items():
        # try:
        # print(f"Generating text for {model_name}")  # Log model name
        prompt, generated_text, model_name, time_taken = generate_text_with_ollama(model_name, model, prompt)
        # print(prompt," : ", generated_text)
        results.append([prompt, generated_text, model_name, time_taken])
        # print(model_name)
        # print("===================")
        # print(prompt)
        # print("===================")
        # print(results[-1][1])
        # print("===================")

    # except Exception as e:
        #     print(f"Error with {model_name}: {e}")  # Catch errors and continue
    return results


with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers for desired parallelism
    futures = {executor.submit(process_text, prompt): prompt for prompt in truncated_texts[:10000]}
    
    # Collect results as they complete
    for idx, future in enumerate(as_completed(futures), 1):
        print(f"Processed {idx} out of 10000")
        data.extend(future.result())

# Save the triplets to a CSV file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Input Prompt", "LLM Output", "Used LLM", "Time Taken (seconds)"])
    # Write the data
    writer.writerows(data)

print(f"CSV file '{output_csv_file}' created with the generated LLM outputs and time taken.")
