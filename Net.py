import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, vocab_size=27, embedding_dim=64, hidden_size=128, output_size=19, num_layers=2):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        x = torch.cat((hn[-1], hn[-1]), dim=1)  # bei 2 Layern bidirectional

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def predict(self,x:torch.Tensor):
        with torch.no_grad():
            x.unsqueeze_(0)
            out = self(x)
            out = F.softmax(out,dim=1)
        return out.squeeze(0)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)