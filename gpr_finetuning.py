import json
import random
import torch
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Charger le dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Mélanger et diviser le dataset en 75% pour l'entraînement et 25% pour le test
random.shuffle(data)
split_idx = int(len(data) * 0.75)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Préparer les données pour la génération de titres, aperçus et recommandations
def preprocess_for_output(data):
    return [{"text": f"Title:\n{item['output']['title']}\nOverview:\n{item['output']['overview']}\nRecommendations:\n{item['output']['recommendations']}"} for item in data]

train_dataset = Dataset.from_list(preprocess_for_output(train_data))
test_dataset = Dataset.from_list(preprocess_for_output(test_data))

# Charger le tokenizer GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
tokenizer.pad_token = tokenizer.eos_token  # Définir le token de padding

# Tokeniser les datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True, padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Configurer les arguments de formation
training_args = TrainingArguments(
    output_dir='./results_output',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Charger le modèle GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Data collator pour le language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Fine-tuning du modèle
trainer.train()

# Sauvegarder le modèle et le tokenizer fine-tunés
model.save_pretrained('./fine_tuned_gpt2_output')
tokenizer.save_pretrained('./fine_tuned_gpt2_output')

# Fonction pour générer un titre, un aperçu et des recommandations
def generate_output(input_text, actual_title, actual_overview, actual_recommendations, model, tokenizer):
    # Tokenize the actual outputs to determine the number of tokens
    actual_title_tokens = tokenizer.encode(actual_title, add_special_tokens=False)
    actual_overview_tokens = tokenizer.encode(actual_overview, add_special_tokens=False)
    actual_recommendations_tokens = tokenizer.encode(actual_recommendations, add_special_tokens=False)

    max_new_tokens_title = len(actual_title_tokens)
    max_new_tokens_overview = len(actual_overview_tokens)
    max_new_tokens_recommendations = len(actual_recommendations_tokens)

    prompt_title = "\nTitle:\n"
    prompt_overview = "\nOverview:\n"
    prompt_recommendations = "\nRecommendations:\n"

    # Generate Title
    encoded_input = tokenizer(input_text + prompt_title, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = encoded_input['input_ids'].to(model.device)
    attention_mask = encoded_input['attention_mask'].to(model.device)

    generated_title = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens_title,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.95,
    )
    title_output = tokenizer.decode(generated_title[0], skip_special_tokens=True)
    title_output = title_output.split(prompt_title, 1)[-1].strip()

    # Generate Overview
    encoded_input = tokenizer(input_text + prompt_overview, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = encoded_input['input_ids'].to(model.device)
    attention_mask = encoded_input['attention_mask'].to(model.device)

    generated_overview = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens_overview,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.95,
    )
    overview_output = tokenizer.decode(generated_overview[0], skip_special_tokens=True)
    overview_output = overview_output.split(prompt_overview, 1)[-1].strip()

    # Generate Recommendations
    encoded_input = tokenizer(input_text + prompt_recommendations, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = encoded_input['input_ids'].to(model.device)
    attention_mask = encoded_input['attention_mask'].to(model.device)

    generated_recommendations = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens_recommendations,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.95,
    )
    recommendations_output = tokenizer.decode(generated_recommendations[0], skip_special_tokens=True)
    recommendations_output = recommendations_output.split(prompt_recommendations, 1)[-1].strip()

    return title_output, overview_output, recommendations_output


# Charger le modèle et le tokenizer pour la génération
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2_output')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2_output')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions = []

for example in test_data:
    input_text = example['input']
    actual_title = example['output']['title']
    actual_overview = example['output']['overview']
    actual_recommendations = example['output']['recommendations']

    # Generate the title, overview, and recommendations with dynamic max_new_tokens
    generated_title, generated_overview, generated_recommendations = generate_output(
        input_text, actual_title, actual_overview, actual_recommendations, model, tokenizer
    )

    # Save the results
    predictions.append({
        "input": input_text,
        "actual_title": actual_title,
        "generated_title": generated_title,
        "actual_overview": actual_overview,
        "generated_overview": generated_overview,
        "actual_recommendations": actual_recommendations,
        "generated_recommendations": generated_recommendations
    })


# Enregistrer les prédictions dans un fichier JSON
with open('gpt2_predictions_full.json', 'w') as f:
    json.dump(predictions, f, indent=4)

print("Prédictions générées et enregistrées dans gpt2_predictions_full.json")
