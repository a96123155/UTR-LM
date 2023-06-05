# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer
# ```该代码定义了一个名为 ESM2 的 PyTorch 模型，继承自 nn.Module。在 __init__ 方法中，定义了一些超参数，例如 num_layers、embed_dim、attention_heads 等等。同时，它还初始化了一些子模块，例如 Embedding 层 embed_tokens、一系列 Transformer 层 layers、预测接触的 ContactPredictionHead 层 contact_head，以及一些线性层 lm_head、supervised_linear、structure_linear 等。该模型的前向传播在 forward 方法中定义，接收一个表示序列的 token 序列 tokens，返回预测的标签和其他附加信息。```

class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
        self.supervised_linear = nn.Linear(self.embed_dim, 1)
        self.structure_linear = nn.Linear(self.embed_dim, 3)
    def forward(self, tokens, repr_layers=[], need_head_weights=True, return_contacts=True, return_representation=True, return_attentions_symm = False, return_attentions = False):
        if return_contacts:
            need_head_weights = True
        
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            #print(f'tokens = {tokens}')
            #print(f'self.mask_idx = {self.mask_idx}')
            #print('x.shape = ', x.shape)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            #print(f'mask_ratio_train = {mask_ratio_train}')
            #print(f'padding_mask = {padding_mask}')
            #print(f'src_lengths = {src_lengths}')
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            #print('mask_ratio_observed = ',mask_ratio_observed)
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
            #print(f'x.shape = {x.shape}:\n', x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            #print(f'x.shape = {x.shape}:\n', x)
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))
#         print(x.shape) # 73, 2, 1280
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x_supervised = self.supervised_linear(x[:,0,:])
        x_structure = self.structure_linear(x)
        x = self.lm_head(x)

        if return_representation:
            result = {"logits": x, "logits_supervised": x_supervised, "logits_structure": x_structure,  "representations": hidden_representations}
        else:
            result = {"logits": x, "logits_supervised": x_supervised, "logits_structure": x_structure}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            if return_attentions: result["attentions"] = attentions
            if return_contacts:
                attentions_symm, contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts
                if return_attentions_symm: result["attentions_symm"] = attentions_symm

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def predict_symmetric_attentions(self, tokens):
        return self(tokens, return_contacts=True)["attentions_symm"]
    
    def predict_attentions(self, tokens):
        return self(tokens, need_head_weights=True)["attentions"]

    def predict_representations(self, tokens):
        return self(tokens, return_representation=True)['representations']

    def predict_logits(self, tokens):
        return self(tokens)['logits']

    def predict_logits_supervised(self, tokens):
        return self(tokens)['logits_supervised']

    def predict_logits_structure(self, tokens):
        return self(tokens)['logits_structure']
