
# =================skip doing pip install in terminal======================
import subprocess
import sys

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "datasets"])

# Call the function to install packages
install_packages()
# =========================================================================





### load dataset ###
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a dataset (using an example dataset for demonstration)
dataset = load_dataset("imdb")  # Replace "imdb" with your own dataset if needed

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)





### data loaders, split training ###
import torch
from torch.utils.data import DataLoader

# Convert the tokenized datasets to PyTorch format
tokenized_datasets = tokenized_datasets.remove_columns(["text"])  # Remove unnecessary columns
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)





### define model ### 
from transformers import AutoModelForSequenceClassification

# Load a pre-trained model with a classification head
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)





### optimizer/loss funciton ### 
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)





### training loop ###
from tqdm.auto import tqdm

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        # Move inputs to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress bar
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())





### evaluation ###
model.eval()
accuracy = 0
num_samples = 0

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    accuracy += (predictions == batch["labels"]).sum().item()
    num_samples += predictions.size(0)

print(f"Validation Accuracy: {accuracy / num_samples:.2f}")






### predict next ###
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()
    return prediction

# Example
print(predict("This movie was fantastic!"))
