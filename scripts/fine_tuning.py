import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    train_data = df[:300]
    eval_data = df[300:330]
    
    train_texts = (train_data['Report Name'] + " " + train_data['History'] + " " + train_data['Observation']).tolist()
    train_labels = train_data['Impression'].tolist()
    eval_texts = (eval_data['Report Name'] + " " + eval_data['History'] + " " + eval_data['Observation']).tolist()
    eval_labels = eval_data['Impression'].tolist()

    return train_texts, train_labels, eval_texts, eval_labels

def fine_tune_model(train_texts, train_labels, eval_texts, eval_labels, model_name="gemma-7b-it"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize the data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, return_tensors="pt")

    class ReportDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ReportDataset(train_encodings, train_labels)
    eval_dataset = ReportDataset(eval_encodings, eval_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained('./models/fine_tuned_model')
    tokenizer.save_pretrained('./models/fine_tuned_model')

# Example usage in main.py
if __name__ == "__main__":
    train_texts, train_labels, eval_texts, eval_labels = preprocess_data('./data/dataset.csv')
    fine_tune_model(train_texts, train_labels, eval_texts, eval_labels)
