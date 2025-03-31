import torch
import torch.nn as nn
import numpy as np

import tiktoken

import sys
sys.path.append("/Users/ganeshnagaraja/Desktop/DeepLearning/LLM/LLM_from_scratch")
from transformer import TransformerBlock


# Configuration of the gpt model
chatgpt_cfg = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "drop_rate": 0.1,
    "n_layers": 12,
    "qkv_bias": False
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer block
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(chatgpt_cfg) for _ in range(chatgpt_cfg["n_layers"])]
        )

        # LayerNorm
        self.final_norm = nn.LayerNorm(chatgpt_cfg["emb_dim"])
        self.out_head = nn.Linear(chatgpt_cfg["emb_dim"], cfg["vocab_size"], bias=False)


    def forward(self, x):
        # x: [batch, num_tokens]
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used for context
        idx_cond = idx[:, -context_size:]

        # get predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # focus only on the last time step
        # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1) # (batch, vocab_size)

        # get idx of the word with maximum probability
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (batch, 1)

        # append sampled index to the sequence
        idx = torch.cat([idx, idx_next], dim=1) # (batch, seq_len + 1)

    return idx


def test_gpt():
    # initialize the BPE from tiktoken. gpt2 uses bpe encoding
    tokenizer = tiktoken.get_encoding("gpt2")

    # sample sentence to encode
    sentence_1 = "What is going in this world?"
    print("considered sentence: ", sentence_1)

    tokenized_sentence_1 = tokenizer.encode(sentence_1)
    print("tokenized sentence: ", tokenized_sentence_1)

    # convert tokenized sentence into tensor
    tensors_1 = torch.tensor(tokenized_sentence_1)
    print("tokens converted into tensors: ", tensors_1)
    
    # sample sentence 2 to encode
    sentence_2 = "I am not sure about it."
    print("considered sentence: ", sentence_2)

    tokenized_sentence_2 = tokenizer.encode(sentence_2)
    print("tokenized sentence: ", tokenized_sentence_2)

    # convert tokenized sentence into tensor
    tensors_2 = torch.tensor(tokenized_sentence_2)
    print("tokens converted into tensors: ", tensors_2)

    # usually the input is given in batches to the model. (Batch x No_of_words x Dimension_per_word)
    batch = torch.stack((tensors_1,tensors_2), dim=0)
    print(batch.shape) # batch x num_tokens x dimension_size_of_each_token

    # initialize the model   
    gpt_model = GPTModel(chatgpt_cfg)
    gpt_model.eval()
    out = generate_text_simple(gpt_model, batch, 6, chatgpt_cfg["context_length"])
    print("output: ", out)
    print("Output length:", len(out[0]))

    # decode the output
    decoded_text = tokenizer.decode(out[0].squeeze(0).tolist())
    print("decoded text: ", decoded_text)

    decoded_text = tokenizer.decode(out[1].squeeze(0).tolist())
    print("decoded text: ", decoded_text)
    

    # model parameter size
    total_params = sum(p.numel() for p in gpt_model.parameters())
    print("Total number of parameters: ", total_params)

    

if __name__ == "__main__":
    test_gpt()