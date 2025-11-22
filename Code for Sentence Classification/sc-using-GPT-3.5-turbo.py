
from openai import OpenAI
import re


client = OpenAI(api_key="Mention_YOUR_API_KEY") ## One need to create personal API key and pass it here

# TC sentence labels
labels = ["BT", "TR", "AT", "QS"]


# Zero-shot sentence classification function using gpt-3.5-turbo

def classify_sentence(sentence):
    prompt = (
        f"Classify the following sentence into one of these classes: {labels}. "
        f"Sentence: \"{sentence}\". Return only the label."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# The next function splits the sentence from the problem text

def split_sentences(text):
    # Simple sentence splitter using regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Remove empty strings
    return [s for s in sentences if s]


# The complete pipeline:
# Input (problem text) ->  Sentence Split -> Sentence Classification

def classify_word_problem(word_problem_text):
    sentences = split_sentences(word_problem_text)
    results = []

    for s in sentences:
        label = classify_sentence(s)
        results.append((s, label))

    return results

# Similarly, the problem texts from the datasets provided, can be passed here.

# Example Scenario

if __name__ == "__main__":
    text = (
        "Stephen has 5 apples. Daniel has 12 apples. Daniel gave 4 apples to Stephen. How many apples does Daniel have now"
    )

    output = classify_word_problem(text)

    for sent, label in output:
        print(f"Sentence: {sent}")
        print(f"Predicted Label: {label}")
