import json
import random
import torch
import re
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset


with open("dataset_with_original_data.json", "r", encoding='utf-8') as file:
    data = json.load(file)

# Préparer les inputs et les outputs pour l'entraînement
inputs = [
    f"Keywords: {item.get('keywords', 'N/A')}. Content: {item.get('content', 'N/A')}. Published Date: {item.get('publishDate', 'N/A')}."
    f"Data Source: {item.get('dataSource', 'N/A')}. Link to Data Source: {item.get('linkToDataSource', 'N/A')}. Platform: {item.get('platform', 'N/A')}."
    f"Site: {item.get('site', 'N/A')}. Stealer: {item.get('stealer', 'N/A')}. Generate Title, Summary, and Recommendations:"
    for item in data
]
titles = [f"Title: {item['title']}" for item in data]
summaries = [f"Summary: {item['summary']}" for item in data]
recommendations = [f"Recommendations: {item['recommendations']}" for item in data]
outputs = [f"{title}\n{summary}\n{recommendation}" for title, summary, recommendation in zip(titles, summaries, recommendations)]

# Créer une liste de dictionnaires avec 'input' et 'output'
data_with_inputs_outputs = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]


data_preprocessed = []
for item in data_with_inputs_outputs:
    data_preprocessed.append({
        "input": item["input"],
        "output": item["output"]
    })


random.shuffle(data_preprocessed)  
split_index = int(len(data_preprocessed) * 0.75)
train_data = data_preprocessed[:split_index]
test_data = data_preprocessed[split_index:]


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        outputs = self.tokenizer(
            item["output"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = outputs["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def fine_tune_model(train_data, test_data, tokenizer, model_save_path, device, epochs=1, batch_size=1, max_length=1024):
    train_dataset = CustomDataset(train_data, tokenizer, max_length)
    eval_dataset = CustomDataset(test_data, tokenizer, max_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=3,
        evaluation_strategy="epoch",
        logging_dir=f'{model_save_path}/logs',
        logging_steps=200,
        learning_rate=3e-5,  
        fp16=True  
    )

    model = BartForConditionalGeneration.from_pretrained("./fine_tuned_bartDataset_model_all_outputs") #modele entraine sur la cyber dataset
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


device = torch.device("cpu")
tokenizer = BartTokenizer.from_pretrained("./fine_tuned_bartDataset_model_all_outputs")
new_model_save_path = "./retrained_bart_modelF"

print("Réentraînement du modèle pour tous les outputs...")
fine_tune_model(train_data, test_data, tokenizer, new_model_save_path, device)


def generate_outputs(odata, tokenizer, model_path, device, max_length=1024, batch_size=4):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    generated_data = []

    with torch.no_grad():
        data = [
            {
                "input": f"Keywords: {item.get('keywords', 'N/A')}. Content: {item.get('content', 'N/A')}. Published Date: {item.get('publishDate', 'N/A')}."
                         f"Data Source: {item.get('dataSource', 'N/A')}. Link to Data Source: {item.get('linkToDataSource', 'N/A')}. Platform: {item.get('platform', 'N/A')}."
                         f"Site: {item.get('site', 'N/A')}. Stealer: {item.get('stealer', 'N/A')}. Generate Title, Summary, and Recommendations:",
                "original_data": item
            } 
            for item in odata
        ]
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            inputs = tokenizer(
                [item["input"] for item in batch_data],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            for item, output_id in zip(batch_data, output_ids):
                generated_output = tokenizer.decode(output_id, skip_special_tokens=True)

                # Split the generated output into title, summary, and recommendations
                title, summary_and_recommendations = generated_output.split("Summary:", 1)
                summary, recommendations = summary_and_recommendations.split("Recommendations:", 1)
                
                # Save the original data and split output
                generated_data.append({
                    "original_data": item["original_data"],
                    "gen_output": {
                        "title": title.strip(),
                        "summary": summary.strip(),
                        "recommendations": recommendations.strip()
                    }
                })

            torch.cuda.empty_cache()

    return generated_data

with open("Infer.json", "r") as file:
    infer_data= json.load(file)
print("Génération des sorties pour tous les aspects..")
generated_data_test = generate_outputs(infer_data, tokenizer, new_model_save_path, device)


with open("test_processed_data.json", "w", encoding='utf-8') as file:
    json.dump(generated_data_test, file, indent=2)

print("les données de test sont sauvegardés.")
