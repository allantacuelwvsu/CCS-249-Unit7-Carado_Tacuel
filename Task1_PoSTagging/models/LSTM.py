import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Dataset Preparation for PoS Tagging
# -------------------------------
class POSTagDataset(Dataset):
    def __init__(self, data, word2idx, tag2idx):
        """
        Dataset accepts tokenized, tagged sentences.
        Converts each (word, tag) into corresponding indices for training.
        """
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = zip(*self.data[idx])
        x = torch.tensor([self.word2idx.get(w.lower(), self.word2idx['UNK']) for w in words])
        y = torch.tensor([self.tag2idx.get(t, self.tag2idx['PAD']) for t in tags])
        return x, y

def collate_fn(batch):
    """
    Custom batch function to pad variable-length sequences in batch.
    Pads word sequences and tag sequences independently.
    """
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_pad, ys_pad

# -------------------------------
# PyTorch LSTM Model for PoS Tagging
# -------------------------------
class LSTMPOSTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=64, pad_idx=0):
        """
        This follows the RNN structure described in the slides:
        xt -> Embedding -> ht = LSTM(ht-1, xt) -> yt = Linear(ht) -> softmax
        We use nn.CrossEntropyLoss which applies the softmax + log internally.
        """
        super(LSTMPOSTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)  # V * ht → tag scores

    def forward(self, x):
        x = self.embedding(x)  # xt → et
        x, _ = self.lstm(x)    # et + ht-1 → ht
        x = self.fc(x)         # ht → yt
        return x  # unnormalized scores; loss will apply softmax

# -------------------------------
# Model Training Loop
# -------------------------------
def train_model(model, dataset, tag2idx, epochs=5, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['PAD'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            outputs = outputs.view(-1, outputs.shape[-1])  # reshape for CE loss
            batch_y = batch_y.view(-1)

            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# -------------------------------
# Prediction Function
# -------------------------------
def predict(model, sentence, word2idx, idx2tag, max_len=50):
    """
    Applies the trained model to an unseen sentence.
    Returns a list of (word, predicted_tag).
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([word2idx.get(w.lower(), word2idx['UNK']) for w in sentence])
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        predictions = torch.argmax(outputs, dim=-1).squeeze(0).tolist()

    return list(zip(sentence, [idx2tag[i] for i in predictions[:len(sentence)]]))