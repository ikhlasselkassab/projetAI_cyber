import json
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Load and prepare the data
with open("dataset_with_original_data.json", "r", encoding='utf-8') as file:
    
    data = json.load(file)

# Prepare the inputs and outputs for training
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

# Create a list of dictionaries with 'input' and 'output'
data_with_inputs_outputs = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]

# Preprocess data
random.shuffle(data_with_inputs_outputs)
split_index = int(len(data_with_inputs_outputs) * 0.75)
train_data = data_with_inputs_outputs[:split_index]
test_data = data_with_inputs_outputs[split_index:]

# Custom Dataset class for GPT-2
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["input"],
            text_pair=item["output"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Fine-tune GPT-2 model
def fine_tune_model(train_data, test_data, tokenizer, model_save_path, device, epochs=1, batch_size=1, max_length=1024):
    train_dataset = CustomDataset(train_data, tokenizer, max_length)
    eval_dataset = CustomDataset(test_data, tokenizer, max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=1,
        evaluation_strategy="epoch",
        logging_dir=f'{model_save_path}/logs',
        logging_steps=200,
        learning_rate=3e-5,
        fp16=True
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Set pad token ID to eos token ID
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

# Initialize device and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Fine-tune the GPT-2 model
new_model_save_path = "./retrained_gpt2_model"
print("Fine-tuning GPT-2 model...")
fine_tune_model(train_data, test_data, tokenizer, new_model_save_path, device)

# Generate outputs using the fine-tuned GPT-2 model
def generate_outputs(odata, tokenizer, model_path, device, max_length=1024, max_new_tokens=200, batch_size=4):
    model = GPT2LMHeadModel.from_pretrained(model_path)
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
                max_length=max_length + max_new_tokens,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

            for item, output_id in zip(batch_data, output_ids):
                generated_output = tokenizer.decode(output_id, skip_special_tokens=True)

                # Extract only the title, summary, and recommendations from the generated output
                title_section = generated_output.split("Generate Title, Summary, and Recommendations:")[1].strip()
                title = title_section.split("Summary:", 1)[0].strip()
                summary_and_recommendations = title_section.split("Summary:", 1)[1]

                if "Recommendations:" in summary_and_recommendations:
                  summary, recommendations = summary_and_recommendations.split("Recommendations:", 1)
                  recommendations = recommendations.split("Title:")[0].strip()  # Remove any text after "Title:"
                  recommendations = recommendations.split("Summary:")[0].strip()  # Remove any text after "Summary:"
                else:
                  summary = summary_and_recommendations.strip()
                  recommendations = "No recommendations provided"

                # Save the original data and split output
                generated_data.append({
                    "original_data": item["original_data"],
                    "gen_output": {
                        "title": title.strip(),  # Ensure only the title is captured
                        "summary": summary.strip(),
                        "recommendations": recommendations.strip()
                    }
                })

            torch.cuda.empty_cache()

    return generated_data


# Load the input data for inference
with open("Inference.json", "r", encoding='utf-8') as file:
    infer_data = json.load(file)

print("Generating outputs...")
generated_data_test = generate_outputs(infer_data, tokenizer, new_model_save_path, device)

# Save the generated data to a JSON file
output_file = "results_gpt2.json"
with open(output_file, "w", encoding='utf-8') as file:
    json.dump(generated_data_test, file, ensure_ascii=False, indent=4)

print(f"Results saved in {output_file}")

