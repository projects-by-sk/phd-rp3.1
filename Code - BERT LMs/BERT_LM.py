import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch

df = pd.read_csv("BT.csv")  #need to appropriately provide path to the dataset

data = []

for _, row in df.iterrows():
    data.append((row['Agent'], "Agent"))
    data.append((row['Q-Value'], "Q-Value"))
    data.append((row['Q-Type'], "Q-Type"))

df_parts = pd.DataFrame(data, columns=["text", "label"])
print(df_parts.head())
print("Total samples:", len(df_parts))

le = LabelEncoder()
df_parts['label_id'] = le.fit_transform(df_parts['label'])

train_df, test_df = train_test_split(df_parts, test_size=0.2, random_state=42, stratify=df_parts['label_id'])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=64)


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(le.classes_)
)

training_args = TrainingArguments(
    output_dir='./results',
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.evaluate()



device = torch.device("cpu")  #this part of code is not required if running the code on a GPU
model.to(device)              #this part of code is not required if running the code on a GPU


def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs.to("cpu"))
    pred = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([pred])[0]


#code to check predictions for example cases
examples = ["Agent2", "number3", "type1", "Agent1", "number1", "type2"]
for ex in examples:
    print(f"{ex} → {predict_label(ex)}")


