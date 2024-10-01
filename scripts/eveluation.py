import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer
from datasets import load_metric

def evaluate_model(eval_texts, eval_labels, model_name="gemma-7b-it"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize eval set
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, return_tensors="pt")

    # Compute perplexity
    trainer = Trainer(model=model)
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss)
    print(f"Perplexity: {perplexity}")
    
    # Compute ROUGE score
    rouge = load_metric('rouge')
    predictions = trainer.predict(eval_encodings)["predictions"]
    pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    results = rouge.compute(predictions=pred_texts, references=eval_labels)
    print("ROUGE scores:", results)

# Example usage in main.py
if __name__ == "__main__":
    # Load the evaluation data (assumes it's already split)
    eval_texts = [...]  # Load the evaluation texts here
    eval_labels = [...]  # Load the corresponding labels here
    evaluate_model(eval_texts, eval_labels)
