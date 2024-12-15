import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Initialize OpenAI API key
openai.api_key = os.getenv("AIPROXY_TOKEN")

def generic_analysis(df):
    """Perform a basic analysis of the dataset."""
    print("Basic Information:")
    print(df.info())

    print("\nSummary Statistics (Numeric Columns):")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

def visualize_data(df, output_dir):
    """Generate and save visualizations."""
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        print("No numeric data available for visualization. Skipping heatmap generation.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    plt.savefig(f"{output_dir}/correlation.png")
    print(f"Saved correlation heatmap to: {output_dir}/correlation.png")
    plt.close()

def analyze_with_llm(filename, output_dir):
    """Perform analysis using LLM and save a story."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filename}' is empty or invalid.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file: {e}")
        sys.exit(1)

    generic_analysis(df)
    visualize_data(df, output_dir)

    if openai.api_key is None:
        print("Error: OpenAI API key not set. Please set the AIPROXY_TOKEN environment variable.")
        sys.exit(1)

    summary = f"Here is a dataset summary:\n{df.describe().to_string(max_rows=10, max_cols=5)}"
    prompt = (
        f"Analyze the following dataset and provide a summary:\n\n"
        f"{summary}\n\n"
        f"Include key insights, trends, and anomalies, and describe potential implications."
    )

    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=200
    )

    if "choices" not in response or not response["choices"]:
        print("Error: LLM did not return a valid response.")
        return

    llm_output = response["choices"][0].get("text", "").strip()
    if not llm_output:
        print("Error: LLM response is empty.")
        return

    with open(f"{output_dir}/README.md", "w") as file:
        file.write("# Analysis Story\n\n")
        file.write(llm_output)
    print(f"Saved LLM output to: {output_dir}/README.md")

def main():
    """Main function to handle input arguments."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    output_dir = filename.split(".")[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analyze_with_llm(filename, output_dir)

def analyze_with_llm(filename, output_dir):
    """Perform analysis using LLM and save a story."""
    # Detect and handle encoding issues
    try:
        # Try with default utf-8 first
        df = pd.read_csv(filename, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"Warning: UTF-8 encoding failed for {filename}. Trying alternative encodings...")
        try:
            # Attempt reading with latin1 as a fallback
            df = pd.read_csv(filename, encoding='latin1')
        except Exception as e:
            print(f"Error: An unexpected error occurred while reading the file: {e}")
            sys.exit(1)

    generic_analysis(df)
    visualize_data(df, output_dir)

    if openai.api_key is None:
        print("Error: OpenAI API key not set. Please set the AIPROXY_TOKEN environment variable.")
        sys.exit(1)

    summary = f"Here is a dataset summary:\n{df.describe().to_string(max_rows=10, max_cols=5)}"
    prompt = (
        f"Analyze the following dataset and provide a summary:\n\n"
        f"{summary}\n\n"
        f"Include key insights, trends, and anomalies, and describe potential implications."
    )

    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=200
    )

    if "choices" not in response or not response["choices"]:
        print("Error: LLM did not return a valid response.")
        return

    llm_output = response["choices"][0].get("text", "").strip()
    if not llm_output:
        print("Error: LLM response is empty.")
        return

    with open(f"{output_dir}/README.md", "w") as file:
        file.write("# Analysis Story\n\n")
        file.write(llm_output)
    print(f"Saved LLM output to: {output_dir}/README.md")


if __name__ == "__main__":
    main()
