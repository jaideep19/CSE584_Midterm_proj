import csv
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.llms import Ollama

# this is just another dataset that we have created not using prompt
# Function to load and use an Ollama model (or any other LLM)
def load_model(model_name):
    return Ollama(model=model_name,)

def generate_text_with_model(model_name, model, prompt):
    universal_prompt = f'Please complete the following text: "{prompt}"\n\nMake sure the completion is coherent and follows the context.'
    start_time = time.time()  
    response = model.generate([universal_prompt])  
    end_time = time.time() 
    time_taken = end_time - start_time 

    generated_text = response.generations[0][0].text  

    return prompt, generated_text, model_name, time_taken  

models = {
    "LLaMA 3.2 (1B)": load_model("llama3.2:1b"),
    "Gemma 2b": load_model("gemma2:2b"),
    "Qwen 2.5 (0.5B)": load_model("qwen2.5:0.5b"),
    "Phi-3.5 (3B)": load_model("phi3.5"),
    "TinyLlama (1.1B)": load_model("tinydolphin"),
    "DeepSeek-Coder (1B)": load_model("deepseek-coder"),
}

# Path to the input CSV with truncated texts
input_csv_file = "/Users/apple/Downloads/truncated_wikipedia_texts.csv"

# Path to the output CSV where triplets will be stored
output_csv_file = "llm_outputs_with_time.csv"

# Read the truncated texts from the input CSV
with open(input_csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    truncated_texts = [row[0] for row in reader]  # Extract the truncated texts
    truncated_texts = ["Corrie is a unisex surname in the English language"]  # Using a fixed example prompt

# List to store the results as triplets (input, LLM output, used LLM, time taken)
data = []

# Use a thread pool for concurrency
def process_text(text):
    results = []
    for model_name, model in models.items():
        try:
            # Generate text using the model with the universal prompt
            text, generated_text, model_name, time_taken = generate_text_with_model(
                model_name, model, text
            )
            results.append([text, generated_text, model_name, time_taken])
            print(results[-1])
        except Exception as e:
            print(f"Error with {model_name}: {e}")  # Catch errors and continue
    return results

# Use ThreadPoolExecutor to handle text processing in parallel
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers for desired parallelism
    futures = {executor.submit(process_text, text): text for text in truncated_texts[:10]}
    
    # Collect results as they complete
    for idx, future in enumerate(as_completed(futures), 1):
        print(f"Processed {idx} out of 10000")
        data.extend(future.result())

# Save the triplets to a CSV file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Input Text", "LLM Output", "Used LLM", "Time Taken (seconds)"])
    # Write the data
    writer.writerows(data)

print(f"CSV file '{output_csv_file}' created with the generated LLM outputs and time taken.")
