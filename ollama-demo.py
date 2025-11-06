"""
Simple DSPy Demo with Ollama using qwen3:30b

This demo shows basic DSPy functionality:
1. Setting up an Ollama model
2. Creating a simple signature
3. Using a predictor to generate responses

Run ollama run qwen3:30b to start the model server before executing this script.
"""

import dspy

# Configure DSPy to use Ollama with qwen3:30b
ollama_model = dspy.LM(
    model='ollama/qwen3:30b',
    api_base='http://localhost:11434',
    api_key=''  # Ollama doesn't require an API key
)

dspy.configure(lm=ollama_model)


# Define a simple signature for question answering
class BasicQA(dspy.Signature):
    """Answer questions with short, factual responses."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="A concise answer to the question")


# Create a predictor using the signature
def main():
    # Initialize the predictor
    predictor = dspy.Predict(BasicQA)

    # Example questions
    questions = [
        "What is the capital of France? A",
        "Who wrote Romeo and Juliet? B",
        "What is 2 + 2? C"
    ]

    print("DSPy Demo with Ollama (qwen3:30b)")
    print("=" * 50)
    print()

    # Run predictions
    for question in questions:
        print(f"Question: {question}")
        response = predictor(question=question)
        print(f"Answer: {response.answer}")
        print("-" * 50)
        print()


if __name__ == "__main__":
    main()
