import torch, math, torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        """
        Initialize the LSTM Language Model.

        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - emb_dim (int): Dimension of the word embeddings.
        - hid_dim (int): Dimension of the hidden state of the LSTM.
        - num_layers (int): Number of LSTM layers.
        - dropout_rate (float): Dropout rate applied to embeddings and LSTM layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, 
                            dropout=dropout_rate, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(hid_dim, vocab_size)

        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize weights for embeddings, LSTM, and fully connected layers.
        """
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)

        # Initialize embedding layer weights
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        
        # Initialize fully connected layer weights
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        
        # Initialize LSTM layer weights
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                    self.hid_dim).uniform_(-init_range_other, init_range_other) 
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim, 
                    self.hid_dim).uniform_(-init_range_other, init_range_other) 

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states for the LSTM.

        Parameters:
        - batch_size (int): Size of the input batch.
        - device (torch.device): Device to which tensors are moved.

        Returns:
        - hidden (tuple): Tuple containing hidden and cell states.
        """
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        """
        Detach hidden states from the computation graph.

        Parameters:
        - hidden (tuple): Tuple containing hidden and cell states.

        Returns:
        - hidden (tuple): Detached tuple containing hidden and cell states.
        """
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, src, hidden):
        """
        Forward pass through the model.

        Parameters:
        - src (torch.Tensor): Input tensor of shape [batch size, seq len].
        - hidden (tuple): Tuple containing hidden and cell states.

        Returns:
        - prediction (torch.Tensor): Output tensor of shape [batch size, seq_len, vocab size].
        - hidden (tuple): Updated hidden states.
        """
        # Embedding layer
        embedding = self.dropout(self.embedding(src))
        # embedding: [batch size, seq len, emb_dim]

        # LSTM layer
        output, hidden = self.lstm(embedding, hidden)      
        # output: [batch size, seq len, hid_dim]
        # hidden = h, c = [num_layers * direction, seq len, hid_dim)

        # Apply dropout to the output
        output = self.dropout(output) 

        # Fully connected layer
        prediction = self.fc(output)
        # prediction: [batch size, seq_len, vocab size]

        return prediction, hidden
