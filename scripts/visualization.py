import matplotlib.pyplot as plt
import seaborn as sns

def visualize_top_word_pairs(word_pairs):
    words, pairs, similarities = zip(*word_pairs)
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.array(similarities).reshape(10, 10), annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()

# Example usage in main.py
if __name__ == "__main__":
    word_pairs = [...]  # Load the word pairs generated in the text_analysis.py
    visualize_top_word_pairs(word_pairs)
