import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset, Subset
import re
from datasets import load_dataset
from flask import Flask, request, render_template_string, redirect, url_for, send_file
import threading
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

# Model components with necessary fixes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :].to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # For visualization
        self.attn_weights = None

    def scaled_dot_product(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach().cpu()  # Save attention weights for visualization
        output = torch.matmul(attn, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Perform linear operation and split into n_heads
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k)

        # Transpose to get dimensions batch_size * n_heads * seq_len * d_k
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply scaled dot product attention
        scores = self.scaled_dot_product(q, k, v, mask)

        # Concatenate heads and put through final linear layer
        concat = (
            scores.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len, dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.fc_out(x)
        return logits

# Load the full dataset
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Simple tokenizer to convert text into token indices
def simple_tokenizer(text, word_to_index):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()
    words = text.split()
    tokenized = [word_to_index.get(word, word_to_index["<unk>"]) for word in words]
    return tokenized

# Build vocabulary from the dataset
vocab = set()
for sample in raw_dataset:
    words = re.sub(r"[^a-zA-Z0-9\s]", "", sample["text"]).lower().split()
    vocab.update(words)

vocab = list(vocab)
vocab.append("<unk>")  # Add unknown token

word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Creating dataset and dataloader
tokenized_text = []
for sample in raw_dataset:
    tokenized_text.extend(simple_tokenizer(sample["text"], word_to_index))

# Hyperparameters
vocab_size = len(word_to_index)
d_model = 128  # Reduced for faster training
n_heads = 4
d_ff = 512
num_layers = 2
max_len = 50
dropout = 0.1

# Create the model
model = Transformer(vocab_size, d_model, n_heads, d_ff, num_layers, max_len, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.tokens[idx : idx + self.seq_len]
        target_seq = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            idx  # Return index for dataset inspection
        )

seq_len = max_len
text_dataset = TextDataset(tokenized_text, seq_len)
full_dataloader = DataLoader(text_dataset, batch_size=16, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flask app for web interface
app = Flask(__name__)

training_progress = []
model_trained = False
training_started = False
training_progress_lock = threading.Lock()

# For attention visualization
attention_maps = []

def train_model(training_size, num_epochs):
    global training_progress, model_trained, training_started, attention_maps

    # Ensure training_size is within the dataset size
    effective_size = min(training_size, len(text_dataset))
    training_subset_indices = list(range(effective_size))
    subset_dataset = Subset(text_dataset, training_subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=16, shuffle=True)

    training_started = True
    model_trained = False
    attention_maps = []

    for epoch in range(num_epochs):
        for batch_idx, (sample_input, target, _) in enumerate(dataloader):
            sample_input = sample_input.to(device)
            target = target.to(device)

            # Set model to training mode and zero out gradients
            model.train()
            optimizer.zero_grad()

            # Forward pass
            output = model(sample_input)

            # Reshape output to match the target shape
            output = output.view(-1, vocab_size)  # Flatten output
            target = target.view(-1)

            # Compute loss
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log batch information to UI
            with training_progress_lock:
                training_progress.append(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        # Generate attention map for a sample input
        sample_attention_map = generate_attention_map()
        attention_maps.append(sample_attention_map)

        with training_progress_lock:
            training_progress.append(
                f"Epoch [{epoch + 1}/{num_epochs}] completed"
            )
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed")

    model_trained = True
    training_started = False

def generate_attention_map():
    # Use a fixed sample input for visualization
    sample_text = "the meaning of life is"
    tokenized_input = simple_tokenizer(sample_text, word_to_index)
    input_tensor = torch.tensor(tokenized_input, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Forward pass to get attention weights
        _ = model(input_tensor)

        # Extract attention weights from the first layer
        attention = model.layers[0].attention.attn_weights  # Shape: [batch_size, n_heads, seq_len, seq_len]
        if attention is not None:
            # Take the first batch element
            attn = attention[0]  # Shape: [n_heads, seq_len, seq_len]
            return attn.cpu().numpy(), tokenized_input
        else:
            return None, None

def reset_model():
    global model, optimizer, training_progress, attention_maps
    global model_trained, training_started

    # Re-initialize the model and optimizer
    model = Transformer(vocab_size, d_model, n_heads, d_ff, num_layers, max_len, dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Reset training variables
    training_progress = []
    attention_maps = []
    model_trained = False
    training_started = False

@app.route("/", methods=["GET", "POST"])
def index():
    global training_progress, model_trained, training_started
    input_text = ''
    predicted_text = ''
    error = ''
    if request.method == "POST":
        if "train" in request.form:
            if not training_started:
                try:
                    training_size = int(request.form.get('training_size', 50))
                    num_epochs = int(request.form.get('num_epochs', 3))
                    training_size = max(1, training_size)
                    num_epochs = max(1, num_epochs)
                except ValueError:
                    training_size = 50
                    num_epochs = 3
                training_thread = threading.Thread(target=train_model, args=(training_size, num_epochs))
                training_thread.start()
                return redirect(url_for("index"))
        elif "input_text" in request.form:
            if model_trained:
                input_text = request.form["input_text"]
                tokenized_input = simple_tokenizer(input_text, word_to_index)
                input_tensor = (
                    torch.tensor(tokenized_input, dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )

                # Evaluation
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_tokens = torch.argmax(output, dim=-1).view(-1).tolist()
                    predicted_words = [
                        index_to_word.get(token, "<unk>") for token in predicted_tokens
                    ]
                    predicted_text = " ".join(predicted_words)
            else:
                error = "Model is not trained yet."
        elif "reset" in request.form:
            reset_model()
            return redirect(url_for("index"))
    return render_template_string(
        html_template,
        input_text=input_text,
        predicted_text=predicted_text,
        error=error,
        training_progress=training_progress,
        dataset_info=f"Dataset Size: {len(text_dataset)} examples, Vocabulary Size: {len(word_to_index)}",
        model_trained=model_trained,
        training_started=training_started,
        max_training_size=len(text_dataset),
        attention_maps=attention_maps,
    )

@app.route("/attention_map.png")
def attention_map():
    if not attention_maps:
        return ""
    # Use the last attention map
    attn, tokenized_input = attention_maps[-1]
    if attn is None:
        return ""
    num_heads = attn.shape[0]
    seq_len = attn.shape[-1]
    tokens = [index_to_word.get(idx, "<unk>") for idx in tokenized_input]
    fig, axs = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    for i in range(num_heads):
        ax = axs[i] if num_heads > 1 else axs
        im = ax.imshow(attn[i], cmap='viridis')
        ax.set_title(f'Head {i+1}')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

html_template = """
<!doctype html>
<html lang="en">
  <head>
    <title>Transformer Model Interface</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    {% if training_started and not model_trained %}
    <meta http-equiv="refresh" content="5">
    {% endif %}
    <style>
      body {
        margin-top: 50px;
      }
      .progress {
        height: 30px;
      }
      .training-log {
        max-height: 300px;
        overflow-y: auto;
      }
      .img-container {
        text-align: center;
      }
      .img-container img {
        width: 100%;
        height: auto;
      }
      .form-section {
        margin-bottom: 30px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="text-center mb-5">
        <h1>Transformer Model Interface</h1>
        <h5>By Sasank Aduri</h5>
        <p>{{ dataset_info }}</p>
      </div>
      <div class="row">
        <div class="col-md-6">
          <h2>Training Panel</h2>
          <form method="post" class="form-section">
            {% if not training_started %}
              <div class="form-group">
                <label for="training_size">Training Size (number of samples):</label>
                <input type="number" class="form-control" name="training_size" id="training_size" value="50" min="1" max="{{ max_training_size }}">
              </div>
              <div class="form-group">
                <label for="num_epochs">Number of Epochs:</label>
                <input type="number" class="form-control" name="num_epochs" id="num_epochs" value="3" min="1">
              </div>
              <button type="submit" name="train" class="btn btn-success btn-block">Start Training</button>
              {% if model_trained %}
              <button type="submit" name="reset" class="btn btn-danger btn-block mt-2">Reset Model</button>
              {% endif %}
            {% elif training_started and not model_trained %}
              <button class="btn btn-secondary btn-block" disabled>Training in Progress...</button>
            {% endif %}
          </form>
          {% if training_started and not model_trained %}
            <div class="progress mt-3">
              <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
            </div>
          {% endif %}
          <hr>
          <h3>Training Progress:</h3>
          <div class="training-log border p-2">
            <ul class="list-unstyled mb-0">
              {% for log in training_progress %}
                <li>{{ log }}</li>
              {% endfor %}
            </ul>
          </div>
          {% if training_started and not model_trained %}
            <p class="text-muted">Page auto-refreshes every 5 seconds during training.</p>
          {% endif %}
          {% if attention_maps %}
          <hr>
          <h3>Attention Map:</h3>
          <p>This diagram shows how the model attends to different parts of the input sequence when making predictions. Each square represents the attention weight between two words in the input sentence. The darker the square, the higher the attention weight.</p>
          <div class="img-container">
            <img src="{{ url_for('attention_map') }}" alt="Attention Map">
          </div>
          {% endif %}
        </div>
        <div class="col-md-6">
          <h2>Testing Panel</h2>
          <form method="post" class="form-section">
            <div class="form-group">
              <textarea name="input_text" class="form-control" rows="5" placeholder="Enter input text here..." {% if not model_trained %}disabled{% endif %}></textarea>
            </div>
            <button type="submit" class="btn btn-primary btn-block" {% if not model_trained %}disabled{% endif %}>Get Prediction</button>
          </form>
          {% if error %}
            <div class="alert alert-danger mt-3">{{ error }}</div>
          {% endif %}
          {% if input_text %}
            <hr>
            <h3>Input Text:</h3>
            <p>{{ input_text }}</p>
            <h3>Predicted Text:</h3>
            <p>{{ predicted_text }}</p>
          {% endif %}
        </div>
      </div>
    </div>
  </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)