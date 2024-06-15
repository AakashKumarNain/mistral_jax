"""This script is intended for stacking weights of different transformer
blocks (layers) into a single gigantic transformer layer. For example,
the current `state_dict` consists of 32 layers of `TransformerBlock` where
each layer consists of attention layers, and feed forward layers. This
script will take the weights of all these 32 layers, and stack them into bigger
stacked matrices.

Why do we want to do this?
Once we have stacked the layers, we can run a `scan` operation for the forward
pass instead of doing a `for` loop. `scan` is more efficient, much faster, and
a friendly operation for the compiler (XLA in this case).

How and when to run this script?
Please go through the detailed instructions provided in the README for the same.
"""

import re
import torch


# Set the device to CPU.
device = torch.device("cpu")

# Load the original weight files. Assuming they are in
# the `model_files` directory
state_dict = torch.load("../model_files/mistral-7B-v0.1/consolidated.00.pth")


# We are going to use regex for finding the layers, and stacking them. We
# will also use a new prefix for the stacked layers. This is just a crude
# way to do it, but given that it is going to be used exactly once, naive
# code is okay for this task

prefix_patt_pairs = [
    ("layers.attention.wq.weight", r"layers.\d+.attention.wq"),
    ("layers.attention.wk.weight", r"layers.\d+.attention.wk"),
    ("layers.attention.wv.weight", r"layers.\d+.attention.wv"),
    ("layers.attention.wo.weight", r"layers.\d+.attention.wo"),
    ("layers.attention.wq.weight", r"layers.\d+.attention.wq"),
    ("layers.attention_norm.weight", r"layers.\d+.attention_norm"),
    ("layers.feed_forward.w1.weight", r"layers.\d+.feed_forward.w1"),
    ("layers.feed_forward.w2.weight", r"layers.\d+.feed_forward.w2"),
    ("layers.feed_forward.w3.weight", r"layers.\d+.feed_forward.w3"),
    ("layers.ffn_norm.weight", r"layers.\d+.ffn_norm"),
]
other_keys = ['tok_embeddings.weight', 'norm.weight', 'output.weight']

# We will save the weights in a new dictionary
new_state_dict = {}

for name, pattern in prefix_patt_pairs:
    new_weights = []
    for key, value in state_dict.items():
        match = re.search(pattern, key)
        if match:
            new_weights.append(value)
        elif key in other_keys:
            new_state_dict[key] = value
    if len(new_weights) != 0:
        new_state_dict[name] = torch.stack(new_weights)


# For all the attention layers within the transformer block, we
# can do one more optimization. Instead of having three different
# weights `wq`, `wk`, `wv`, we can have one linear layer with the
# merged dimensions, and then split the output of this linear layer
# in the forward pass. Let us do that.
wq = new_state_dict.pop("layers.attention.wq.weight")
wk = new_state_dict.pop("layers.attention.wk.weight")
wv = new_state_dict.pop("layers.attention.wv.weight")
new_state_dict["layers.attention.wqkv.weight"] = torch.cat([wq, wk, wv], dim=1)


# Save the new weights dict
torch.save(new_state_dict, "../model_files/merged_state_dict_mistral7B.pth")


# You should get these numbers after running this script:
#
# Weights: tok_embeddings.weight            shape:torch.Size([32000, 4096])
# Weights: norm.weight                      shape:torch.Size([4096])
# Weights: output.weight                    shape:torch.Size([32000, 4096])
# Weights: layers.attention.wo.weight       shape:torch.Size([32, 4096, 4096])
# Weights: layers.attention_norm.weight     shape:torch.Size([32, 4096])
# Weights: layers.feed_forward.w1.weight    shape:torch.Size([32, 14336, 4096])
# Weights: layers.feed_forward.w2.weight    shape:torch.Size([32, 4096, 14336])
# Weights: layers.feed_forward.w3.weight    shape:torch.Size([32, 14336, 4096])
# Weights: layers.ffn_norm.weight           shape:torch.Size([32, 4096])
# Weights: layers.attention.wqkv.weight     shape:torch.Size([32, 6144, 4096])

