from chunk_generator import chunk_generator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json, os

from pathlib import Path

input_path="C:\\sj\\data\\input"
model_path = "C:\\sj\\models\\Mistral-7B-Instruct-v0.3"
output_path= "C:\\sj\\data\\output"
file_name = output_path + '\\data.json'

#template credit: https://github.com/ALucek
label_template = """
You are an AI assistant tasked with generating a single, realistic question-answer pair based on a given document. The question should be something a user might naturally ask when seeking information contained in the document.

Given: {chunk}

Instructions:
1. Analyze the key topics, facts, and concepts in the given document, choose one to focus on.
2. Generate twenty similar questions that a user might ask to find the information in this document that does NOT contain any company name.
3. Use natural language and occasionally include typos or colloquialisms to mimic real user behavior in the question.
4. Ensure the question is semantically related to the document content WITHOUT directly copying phrases.
5. Make sure that all of the questions are similar to eachother. I.E. All asking about a similar topic/requesting the same information.

Output Format:
Return a object with the following structure:

{{
  Generated question text,
  Generated question text,
}}

Be creative, think like a curious user, and generate your 20 similar questions that would naturally lead to the given document in a semantic search. Ensure your response is a valid JSON object containing only the questions.

"""
anwer_len = len(label_template) + 256
# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the pad_token to eos_token
#tokenizer.pad_token = tokenizer.eos_token


tokenizer.add_special_tokens({'pad_token': '[PAD]'})

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                
bnb_4bit_compute_dtype=torch.float16, # Match input dtype
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.float16,
                                             device_map="cuda",
                                             quantization_config=bnb_config)

model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings sinde new pad_token is added

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
print(model.device)

#TODO: instead of creating chunks in bulk create per folder
chunks = chunk_generator().create_chunks(input_path)

# Generate responses for each chunk
for i, chunk in enumerate(chunks, 1):
    prompt = label_template.format(chunk=chunk.page_content)
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens = anwer_len,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    

    # Extract answers
    generated_questions = response[len(prompt):].strip().strip('{').strip('}')
    
    print(generated_questions)

    # Prepare new entries
    new_entries = [
        {"question": question, "answer": chunk.page_content}
        for question in generated_questions.split('\n') if question.strip()
    ]

    # Load existing data if file exists
    if os.path.exists(file_name):
        with open(file_name, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    # Append new entries
    existing_data.extend(new_entries)

    # Save updated data
    with open(file_name, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)




