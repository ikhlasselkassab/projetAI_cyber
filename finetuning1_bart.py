import json
import random
import torch
import shutil
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from google.colab import drive



# Load the dataset
with open("dataset.json", "r", encoding='utf-8') as file:
    data = json.load(file)

# Split the dataset into 75% training and 25% testing
random.shuffle(data)  # Shuffle the data to ensure randomness
split_index = int(len(data) * 0.75)
train_data = data[:split_index]
test_data = data[split_index:]

# Create a custom dataset class
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

        # Combine the labels for title, overview, and recommendations into one sequence with clear delimiters
        combined_labels = f"Title: {item['output']['title']} Overview: {item['output']['overview']} Recommendations: {item['output']['recommendations']}"
        outputs = self.tokenizer(
            combined_labels,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = outputs["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in the loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Function to fine-tune the model
def fine_tune_model(train_data, test_data, tokenizer, model_save_path, device, epochs=4, batch_size=1, max_length=1024):
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
        learning_rate=3e-5,  # Adjust the learning rate as needed
    )

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
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

# Function to save the model to Google Drive


# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model_save_path = "./fine_tuned_bart_model_all_outputs"
drive_path = "/content/drive/My Drive/"

print("Fine-tuning model for all outputs...")
fine_tune_model(train_data, test_data, tokenizer, model_save_path, device)



# Function to generate outputs using the fine-tuned model
def generate_outputs(data, tokenizer, model_path, device, max_length=1024, batch_size=8):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    generated_data = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
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
                item["output"]["generated"] = generated_output
                generated_data.append(item)

            # Clear CUDA cache
            torch.cuda.empty_cache()

    return generated_data

# Generate outputs for test data and save results
print("Generating outputs for all aspects...")
generated_data_test = generate_outputs(test_data, tokenizer, model_save_path, device)

# Save the final test dataset with all outputs
with open("test_datasetBart.json", "w", encoding='utf-8') as file:
    json.dump(generated_data_test, file, indent=2)

print("Updated JSON file with generated outputs for test data has been saved.")
