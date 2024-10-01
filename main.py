from scripts.fine_tuning import fine_tune_model
from scripts.evaluation import evaluate_model
from scripts.text_analysis import preprocess_text, compute_embeddings, find_top_word_pairs
from scripts.visualization import visualize_top_word_pairs

# Load dataset
data_path = './data/dataset.csv'

# Fine-tune model
train_texts, train_labels, eval_texts, eval_labels = preprocess_data(data_path)
fine_tune_model(train_texts, train_labels, eval_texts, eval_labels)

# Evaluate model
evaluate_model(eval_texts, eval_labels)

# Text analysis and embedding generation
cleaned_texts = preprocess_text(train_texts + eval_texts)
embeddings, feature_names = compute_embeddings(cleaned_texts)
top_word_pairs = find_top_word_pairs(embeddings, feature_names)

# Visualize word pairs
visualize_top_word_pairs(top_word_pairs)
