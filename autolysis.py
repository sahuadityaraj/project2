import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Step 1: Initialize OpenAI API
openai.api_key = os.getenv("AIPROXY_TOKEN")

def generic_analysis(df):
    """Perform a generic analysis on the dataset."""
    print("Basic Information")
    print(df.info())
    print("\nSummary Statistics")
    print(df.describe())

    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

def visualize_data(df, output_dir):
    """Create visualizations and save them as PNG files."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation.png")
    plt.close()

def analyze_with_llm(filename, output_dir):
    """Send summaries and prompts to LLM."""
    df = pd.read_csv(filename)
    generic_analysis(df)
    visualize_data(df, output_dir)

    # Send a summarized prompt to LLM
    summary = f"Perform analysis on this dataset: {filename}"
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=summary,
        max_tokens=200
    )

    # Write results
    with open(f"{output_dir}/README.md", "w") as file:
        file.write("# Analysis Results\n")
        file.write(response["choices"][0]["text"])

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    output_dir = filename.split(".")[0]  # Extract output directory name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analyze_with_llm(filename, output_dir)

if __name__ == "__main__":
    main()

