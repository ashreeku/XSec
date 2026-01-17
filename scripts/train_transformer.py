import json
import os
import sys
import torch
import copy
import math

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5, norm_first=False, batch_first=True, bias=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CrossAttentionSavingDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_cross_attn_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        # **kwargs absorbs extra args like tgt_is_causal, is_causal, etc.
        tgt2, self_attn_weights  = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        self.last_self_attn_weights = self_attn_weights.detach().cpu()
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross (encoder-decoder) attention with weights
        tgt2, cross_attn_weights = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_cross_attn_weights = cross_attn_weights.detach().cpu()  # Save for visualization

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(self, d_model, num_features, num_classes, num_heads=2, num_encoder_layers=2, num_decoder_layers=2, device=torch.device("cpu")):
        super(Transformer, self).__init__()
        self.output_proj_dim = d_model
        self.position_encoding(num_features)

        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model, dropout=0, batch_first=True
        )
        decoder_layer = CrossAttentionSavingDecoderLayer(
            d_model, num_heads, dim_feedforward=d_model, dropout=0, batch_first=True
        )
        self.x_proj = nn.Linear(1, d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.feedforward = nn.Linear(self.output_proj_dim, num_classes)
        self.device = device
        self._init_projection_weights()
        self.apply(self.initialize_tsformer_weights)
    
    def position_encoding(self, num_features):
        src_pe = torch.zeros(num_features, self.output_proj_dim)
        position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.output_proj_dim, 2).float() * (-math.log(10000.0) / self.output_proj_dim))
        src_pe[:, 0::2] = torch.sin(position * div_term)
        src_pe[:, 1::2] = torch.cos(position * div_term)
        src_pe = src_pe.unsqueeze(0)
        self.register_buffer('src_pe', src_pe)
        tgt_pe = torch.zeros(1, self.output_proj_dim)
        position = torch.arange(0, 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.output_proj_dim, 2).float() * (-math.log(10000.0) / self.output_proj_dim))
        tgt_pe[:, 0::2] = torch.sin(position * div_term)
        tgt_pe[:, 1::2] = torch.cos(position * div_term)
        tgt_pe = tgt_pe.unsqueeze(0)
        self.register_buffer('tgt_pe', tgt_pe)
    
    def _init_projection_weights(self):
        """Special initialization for projection layers"""
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)
    
    @staticmethod
    def initialize_tsformer_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.x_proj(x)
        y = torch.full(size=(batch_size, 1, 1), fill_value=-1, dtype=torch.float32).to(self.device)
        y = self.x_proj(y)
        src_input = x * math.sqrt(self.output_proj_dim) + self.src_pe
        tgt_input = y * math.sqrt(self.output_proj_dim) + self.tgt_pe

        memory = self.encoder(
            src_input,
            src_mask=None,
            src_key_padding_mask=None
        )

        output = self.decoder(
            tgt_input, memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )

        output = self.feedforward(output)
        output = output.squeeze()

        return output, None, None, None

    def explain(self):
        cross_attn_weights_list = []
        for decoder_layer in self.decoder.layers:
            # Extract cross-attention weights from each decoder layer
            attn_weights = decoder_layer.last_cross_attn_weights  # Shape: (batch, heads, tgt_len, memory_len)
            if attn_weights is not None:
                cross_attn_weights_list.append(attn_weights)
        
        if len(cross_attn_weights_list) == 0:
            return None

        # Stack and average attentions over layers and heads
        stacked = torch.stack(cross_attn_weights_list, dim=0)      # (num_layers, batch, heads, tgt_len, mem_len)
        mean_attn = stacked.mean(dim=0)                            # Average over layers
        mean_attn = mean_attn.mean(dim=1)                          # Average over heads
        return mean_attn  # Shape: (batch, tgt_len, memory_len)


class Skeleton(nn.Module):
    def __init__(self, feature_dim, num_classes, device=None):
        super(Skeleton, self).__init__()
        self.device = device if device else torch.device("cpu")
        
        # fill here

    def forward(self, x):
        # fill here
        return x, None, None, None
    
    def explain(self, x):
        # fill here assuming x is a single sample not batch
        pass


def train_one_epoch(epoch, optimizer):
    model.train(True)
    for i, (samples, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        samples = samples.to(device)
        labels = labels.to(device)
        predictions, _, _, _ = model(samples)
        predictions = predictions.squeeze()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/{max_epochs}], latest loss: {loss.item():.4f}")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 1
    seed = random.sample(range(1, 10000), num_seeds)[0]
    print(f"Running seed {seed}")
    set_seed(seed)

    x_train, y_train, x_test, y_test = load_data(config["data_path"])

    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = Transformer(config["embedding_dim"], config["feature_dim"], config["num_classes"], device=device).to(device)

    opt_specs = [
        {"params": model.parameters(), "lr": 1e-3}
    ]
    opt = torch.optim.Adam(opt_specs)

    epoch = 1
    max_epochs = 50

    while epoch <= max_epochs:
        train_one_epoch(epoch, opt)
        epoch += 1

    model.train(False)
    gt, pred = evaluate(train_loader, model, device)
    performance = EvaluationScores(gt, pred, config["num_classes"])
    acc = performance.accuracy()
    pre = performance.precision()
    rec = performance.recall()
    fpr = performance.fpr()
    print(f"Train accuracy: {acc}")
    print(f"Train Precision: {pre}")
    print(f"Train Recall: {rec}")
    print(f"Train FPR: {fpr}")

    # testing
    gt, pred = evaluate(test_loader, model, device)
    performance = EvaluationScores(gt, pred, config["num_classes"])
    acc = performance.accuracy()
    pre = performance.precision()
    rec = performance.recall()
    fpr = performance.fpr()
    print(f"Test accuracy: {acc}")
    print(f"Test Precision: {pre}")
    print(f"Test Recall: {rec}")
    print(f"Test FPR: {fpr}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(config["save_path"], f"model_transformer.pt"))