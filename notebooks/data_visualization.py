from datasets import load_dataset
import matplotlib.pyplot as plt

def main():
    # Step 1: Load a subset of the Wikipedia dataset (adjust size if needed)
    print("Loading dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]")

    # Step 2: Define a function to compute the word count for each article
    def compute_length(example):
        text = example.get("text", "")
        word_count = len(text.split())
        example["length"] = word_count
        return example

    # Step 3: Map the length function to the dataset
    print("Computing article lengths...")
    dataset = dataset.map(compute_length)

    # Step 4: Extract the lengths
    lengths = dataset["length"]

    # Step 5: Plot a histogram of article lengths
    print("Plotting histogram...")
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Wikipedia Article Lengths (Word Count)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Articles")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
