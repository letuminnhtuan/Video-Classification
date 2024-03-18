import torch
import math
import torch.nn.functional as F

class Positional_Encoding(torch.nn.Module):
    def __init__(self, seq_length, n_dim):
        super(Positional_Encoding, self).__init__()
        self.seq_length = seq_length
        self.n_dim = n_dim

    def forward(self):
        # positional vector
        position_encode = torch.zeros((self.seq_length, self.n_dim))
        for pos in range(self.seq_length):
            for i in range(0, self.n_dim, 2):
                position_encode[pos, i] = math.sin(pos / (10000 ** (2 * i / self.n_dim)))
                position_encode[pos, i+1] = math.cos(pos / (10000 ** (2 * i / self.n_dim)))
        return position_encode

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_dim):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_dim_each_head = int(self.n_dim / self.n_head)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # init query, key, value
        self.query_matrix = torch.nn.Linear(self.n_dim_each_head, self.n_dim_each_head, bias=False)
        self.key_matrix = torch.nn.Linear(self.n_dim_each_head, self.n_dim_each_head, bias=False)
        self.value_matrix = torch.nn.Linear(self.n_dim_each_head, self.n_dim_each_head, bias=False)
        self.output_matrix = torch.nn.Linear(self.n_dim_each_head * self.n_head, self.n_dim_each_head * self.n_head, bias=False)

    def forward(self, query, key, value, mask=None):  # (batch_size, seq_length, n_dim)
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)
        # divide head => (batch_size, seq_length, n_head, n_dim_each_head)
        query = query.view(batch_size, seq_length_query, self.n_head, self.n_dim_each_head)
        key = key.view(batch_size, seq_length, self.n_head, self.n_dim_each_head)
        value = value.view(batch_size, seq_length, self.n_head, self.n_dim_each_head)
        q = self.query_matrix(query)
        k = self.key_matrix(key)
        v = self.value_matrix(value)
        # transpose => (batch_size, n_head, seq_length, n_dim_each_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # -------------------------- Compute MultiHead-Attention --------------------------
        """
        - Step 1: compute matmul(q, k^T)
        - Step 2: scale with sqrt(n_dim)
        - Step 3: compute softmax => matrix A
        - Step 4: compute matmul of matrix A and value matrix
        - Step 5: concatenate matrix => matrix Z
        - Step 4: compute matmul of matrix Z and matrix W0
        """
        k_T = k.transpose(-1, -2)  # => (batch_size, n_head, n_dim_each_head, seq_length)
        product = torch.matmul(q, k_T)  # => (batch_size, n_head, seq_length_query, seq_length)
        product = product / math.sqrt(self.n_dim_each_head)
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))
        product = product.to(self.device)
        scores = F.softmax(product, dim=-1)  # => (batch_size, n_head, seq_length_query, seq_length)
        scores = torch.matmul(scores, v)  # => (batch_size, n_head, seq_length_query, n_dim_each_head)
        scores = scores.transpose(1, 2)  # => (batch_size, seq_length_query, n_head, n_dim_each_head)
        scores = scores.contiguous().view(batch_size, seq_length_query, self.n_dim_each_head * self.n_head)
        output = self.output_matrix(scores)
        return output

class TransformerBlock(torch.nn.Module):
    def __init__(self, n_head, n_dim, n_expansion):
        super(TransformerBlock, self).__init__()
        # parameters
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_expansion = n_expansion
        # instances
        self.multihead = MultiHeadAttention(n_head=self.n_head, n_dim=self.n_dim)
        self.norm_attention = torch.nn.LayerNorm(self.n_dim)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_expansion * self.n_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_expansion * self.n_dim, self.n_dim),
            torch.nn.LeakyReLU(),
        )
        self.norm_feedforward = torch.nn.LayerNorm(self.n_dim)

    def forward(self, query, key, value):
        multihead_vector = self.multihead(query, key, value)
        add_norm_vector = self.norm_attention(multihead_vector + query)
        feed_forward_vector = self.feedforward(add_norm_vector)
        output = self.norm_feedforward(feed_forward_vector + add_norm_vector)
        return output

class ViT(torch.nn.Module):
    def __init__(self, input_chanel, output_chanel, n_head, n_expansion, n_layer, num_classes):
        super(ViT, self).__init__()
        # Parameters
        self.input_chanel = input_chanel
        self.output_chanel = output_chanel
        self.n_head = n_head
        self.n_expansion = n_expansion
        self.n_layer = n_layer
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instance
        self.patch_embedding = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=self.input_chanel, out_channels=16, kernel_size=(8, 32, 32), stride=(8, 32, 32), padding=(0, 0, 0)),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0)),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=-3)
        )
        self.transformer_block = TransformerBlock(self.n_head, self.output_chanel, self.n_expansion)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.output_chanel, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_classes),
        )

    def add_cls_token(self, x):
        batch_size = x.shape[0]
        cls_token = torch.nn.Parameter(data=torch.zeros(batch_size, 1, self.output_chanel), requires_grad=True).to(self.device)
        return torch.concat([cls_token, x], dim=1)

    def forward(self, x):
        """ Input shape: (batch_size, chanel, height, width) """
        x = self.patch_embedding(x)     # => (batch_size, seq_len, output_chanel)
        x = x.transpose(-1, -2)
        x = self.add_cls_token(x)       # => (batch_size, seq_len+1, output_chanel)
        position = Positional_Encoding(seq_length=x.shape[1], n_dim=self.output_chanel)
        x = x + position().to(self.device)
        for _ in range(self.n_layer):
            x = self.transformer_block(x, x, x)
        x = x[:, 0, :]
        output = self.fc(x)
        return output
