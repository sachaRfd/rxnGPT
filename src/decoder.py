"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""  # noqa

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0  # noqa
        self.flash = hasattr(
            torch.nn.functional,
            "scaled_dot_product_attention",
        )
        # self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"  # noqa
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence    # noqa
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.block_size, config.block_size),
                ).view(1, 1, config.block_size, config.block_size),
            )

    def adapted_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        pad_mask=None,
        scale=None,
    ) -> torch.Tensor:
        """
        Extra method to try and not include padding in the attention calculation.
        """
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # Remove padding from being used in the attention calculation:
        if pad_mask is not None:
            # Create mask for attention where pad_mask is 3: which has bs:
            pad_mask_att = pad_mask == 2
            pad_mask_att = pad_mask_att.unsqueeze(1).unsqueeze(1)
            pad_mask_att = pad_mask_att.repeat(1, attn_weight.shape[1], 1, 1)

            attn_bias = attn_bias.repeat(pad_mask_att.shape[0], 1, 1, 1)
            attn_bias = attn_bias.masked_fill(pad_mask_att, float("-inf"))
            inf_mask = torch.isinf(attn_bias).all(dim=-1)
            attn_bias[inf_mask] = 0

        attn_weight += attn_bias

        attn_weight = torch.softmax(attn_weight, dim=-1)

        return attn_weight @ value

    def forward(self, x, attention_mask=None, src_tgt_language_mask=None):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim # noqa
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)  # noqa
        # efficient attention using Flash Attention CUDA kernels
        # print(self.flash)
        # exit()
        if self.flash:
            if attention_mask is not None:  # or src_tgt_language_mask is not None:
                # Create causal mask - (bs, n_head, T, T)
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(
                    diagonal=0
                )

                causal_mask = causal_mask.unsqueeze(0).repeat(B, 1, 1)

                if src_tgt_language_mask is not None:
                    # for sample in range(B):
                    #     # Add true to all 1s in the src_tgt_mask:
                    #     causal_mask[sample] = causal_mask[sample].masked_fill(
                    #         src_tgt_language_mask[sample] == 1, True
                    #     )
                    # Expand src_tgt_language_mask to match the shape of causal_mask
                    expanded_src_tgt_language_mask = src_tgt_language_mask.unsqueeze(
                        1
                    ).expand(-1, T, -1)
                    # print("Here in  causal mask section: ")
                    # print(causal_mask.shape)
                    # print(causal_mask)

                    # Use the expanded src_tgt_language_mask to modify causal_mask
                    causal_mask = causal_mask.masked_fill(
                        expanded_src_tgt_language_mask == 1, True
                    )

                    # print("After adding bidireactional modification: ")
                    # print(causal_mask[0])

                    # # save the causal_mask to a file:
                    # causal_mask_list = causal_mask[0].tolist()
                    # with open("causal_mask.txt", "w") as f:
                    #     for item in causal_mask_list:
                    #         f.write("%s\n" % item)

                    # exit()

                # View attention mask:
                attention_mask = attention_mask.unsqueeze(1).bool()

                # Fill causal_mask based on conditions:
                # 1. If causal mask is true and attention mask is true, output mask should be true:
                output_causal_mask = torch.where(
                    causal_mask.bool() & attention_mask, True, False
                )
                # 2. If causal mask is true and attention mask is false, output mask should be false:
                output_causal_mask = torch.where(
                    causal_mask.bool() & ~attention_mask, False, output_causal_mask
                )

                causal_mask = output_causal_mask

                # # Add the head to mask:
                causal_mask = causal_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=causal_mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                )

            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0,  # self.dropout if self.training else 0,
                    is_causal=True,
                )

        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            # use_topk_in_attention = False
            # if use_topk_in_attention:
            #     top_k = 10
            #     # only keep the top k values in the attention
            #     v_, _ = att.topk(top_k, dim=-1)
            #     att[att < v_[:, :, :, [-1]]] = -float("Inf")

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask, src_tgt_language_mask=None):
        x = x + self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            src_tgt_language_mask=src_tgt_language_mask,
        )  # noqa
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency  # noqa
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster    # noqa


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),  # Token Pos emb
                wpe_2=nn.Embedding(config.block_size, config.n_embd),  # Token Pos emb
                wste=nn.Embedding(3, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """  # noqa
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        src_tgt_language_mask=None,
        Products=None,
        attention_mask=None,
        use_seperate_pos_enc=False,
        use_bidirectional_src_mask=False,
    ):
        device = idx.device
        b, t = idx.size()  # batch, tokens
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"  # noqa

        # Create 2 position sequences, one starting from when the src_tgt_mask is 1 and one starting from when it is 2:
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Seperate the src and tgt position embeddings for easier task:
        if use_seperate_pos_enc:
            pos_2 = torch.arange(0, t, dtype=torch.long, device=device)

            pos = pos.unsqueeze(0).repeat(b, 1)
            pos_2 = pos_2.unsqueeze(0).repeat(b, 1)

            # Get masks:
            mask_1 = src_tgt_language_mask != 1

            for i in range(b):
                # Get where the products are:
                mask_sum = mask_1[i].sum()

                # Cut the pos sequence to the length of the products:
                to_add = pos_2[i][:mask_sum]

                # Add those positions to the pos sequence:
                pos[i][-mask_sum:] = to_add

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            idx
        )  # token embeddings of shape (b, t, n_embd) # noqa

        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (t, n_embd) # noqa

        # Add info about which tokens are reactants and which are products.
        if src_tgt_language_mask is not None:
            src_tgt_emb = self.transformer.wste(
                src_tgt_language_mask,
            )

            x = self.transformer.drop(
                tok_emb + pos_emb + src_tgt_emb
            )  # Add embeddings with dropout.    ' noqa
        else:
            x = self.transformer.drop(
                tok_emb + pos_emb
            )  # Add embeddings with dropout.    ' noqa

        if use_bidirectional_src_mask:
            src_tgt_language_mask_to_use = src_tgt_language_mask

        else:
            src_tgt_language_mask_to_use = None
        for block in self.transformer.h:
            x = block(
                x,
                attention_mask=attention_mask,
                src_tgt_language_mask=src_tgt_language_mask_to_use,
            )  # noqa
        x = self.transformer.ln_f(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if Products is not None:
                # Only Take the Products :)
                logits = logits[:, -Products.shape[1] :, :]
                targets = targets[:, -Products.shape[1] :]
            loss = F.cross_entropy(
                logits.permute(0, 2, 1),
                targets,
                ignore_index=0,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position  # noqa
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx,
        src_tgt_mask,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        greedy_decode: str = None,
        label2token: dict = None,
        use_sep_pos_enc: bool = False,
        use_attention_mask: bool = False,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """  # noqa

        assert greedy_decode in [
            "greedy",
            "beam",
            "sampling",
        ], "please choose a valid decoding algo."

        assert temperature == 1.0 if greedy_decode == "greedy" else True

        assert top_k is not None if greedy_decode in ["beam", "sampling"] else True

        predicted_product = list()

        # setup attention_mask:
        if use_attention_mask:
            attention_mask = torch.ones(1, idx.size(1), device=idx.device)
        else:
            attention_mask = None
        # Code to check if having padding changes anything:
        # Setup last one to be 0:
        # attention_mask[:, -2] = 0

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size

            if idx.size(1) >= self.config.block_size:
                break


            # forward the model to get the logits for the index in the sequence
            logits, _ = self(
                idx=idx,
                targets=None,
                Products=None,
                src_tgt_language_mask=src_tgt_mask,
                use_seperate_pos_enc=use_sep_pos_enc,
                attention_mask=attention_mask,
                use_bidirectional_src_mask=True,
            )

            # print("finish generate function")
            # exit()

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            if greedy_decode == "greedy":
                # choose the index with the highest probability:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            elif greedy_decode == "sampling":
                # Sample from distribution of prediction:
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)

            # Check that we have not generated the end label:
            value = label2token[idx_next.item()]
            if value == "[ENDP]":
                could_finish = 1
                return predicted_product, could_finish

            # append the chosen index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # generated_value = label
            predicted_product.append(value)
            # append new token to src_tgt_mask:
            src_tgt_mask = torch.cat(
                (src_tgt_mask, torch.tensor([[2]], device=idx.device)), dim=-1
            )

            if use_attention_mask:
                # Add the new token to the attention mask:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones(1, 1, device=idx.device)), dim=-1
                )

        print("Ran out of tokens in generation.")
        print("Generated sample: ", "".join(predicted_product))
        could_finish = 0
        return predicted_product, could_finish

    # @torch.no_grad()
    # def generate_batch(
    #     self,
    #     idx,
    #     src_tgt_mask,
    #     attention_mask,
    #     max_new_tokens: int,
    #     temperature: float = 1.0,
    #     top_k: int = None,
    #     greedy_decode: str = None,
    #     label2token: dict = None,
    # ):
    #     """
    #     Greedy decoding for a batch of samples.
    #     """  # noqa
    #     assert greedy_decode == "greedy", "please choose a valid decoding algo."
    #     assert temperature == 1.0 if greedy_decode == "greedy" else True
    #     assert top_k is not None if greedy_decode in ["beam", "sampling"] else True

    #     # Setup tensors:
    #     to_add_to_src_tgt_mask = torch.tensor([[2]], device=idx.device).repeat(
    #         src_tgt_mask.size(0), 1
    #     )
    #     to_add_to_att_mask = torch.ones(1, 1, device=idx.device).repeat(
    #         src_tgt_mask.size(0), 1
    #     )
    #     predicted_product = torch.zeros(idx.size(0), max_new_tokens, device=idx.device)

    #     # Empty list to store the outputs but make it the size of bs:
    #     # output_product = [None] * idx.size(0)

    #     is_finished = [False] * idx.size(0)

    #     for i in range(max_new_tokens):
    #         logits, _ = self(
    #             idx=idx,
    #             targets=None,
    #             Products=None,
    #             src_tgt_language_mask=src_tgt_mask,
    #             use_seperate_pos_enc=True,
    #             attention_mask=attention_mask,
    #         )
    #         # print(logits.shape)
    #         # exit()

    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature

    #         # choose the index with the highest probability:
    #         idx_next = torch.argmax(logits, dim=-1, keepdim=True)

    #         # append the chosen index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)
    #         predicted_product[:, i] = idx_next.squeeze()

    #         # append new token to src_tgt_mask:
    #         src_tgt_mask = torch.cat(
    #             (
    #                 src_tgt_mask,
    #                 torch.tensor([[2]], device=idx.device).repeat(
    #                     src_tgt_mask.size(0), 1
    #                 ),
    #             ),
    #             dim=-1,
    #         )

    #         # Add the new token to the attention mask:
    #         attention_mask = torch.cat(
    #             (
    #                 attention_mask,
    #                 torch.tensor([[1]], device=idx.device).repeat(
    #                     attention_mask.size(0), 1
    #                 ),
    #             ),
    #             dim=-1,
    #         )

    #     # Now iterate over the saved samples:
    #     predicted_products = []

    #     # print(output_product)
    #     for input_ids in predicted_product:
    #         # print(input_ids)
    #         product = []
    #         for idx in input_ids:
    #             # print(idx)
    #             label = label2token[idx.item()]
    #             # print(label)
    #             if label != "[ENDP]":  # and label != "[PAD]":
    #                 product.append(label)
    #             if label == "[ENDP]":  # and label=="[PAD]":
    #                 break
    #         print("".join(product))
    #         predicted_products.append(product)
    #     # exit()
    #     return predicted_products

    # @torch.no_grad()
    # def beam_search_decode(
    #     self,
    #     idx,
    #     src_tgt_mask,
    #     label2token,
    #     max_new_tokens,
    #     top_k,
    #     use_sep_pos_enc,
    #     use_attention_mask,
    # ):
    #     """
    #     Conduct Beam search using the decoder model and the input reactants
    #     - For now it is still unbatched - And very slow
    #     """
    #     # start_time1 = time.time()
    #     # assert use_attention_mask == True, "Please use attention mask for beam search"

    #     # setup attention_mask:
    #     # attention_mask = torch.ones(1, idx.size(1), device=idx.device)

    #     src_tgt_to_add = torch.tensor([[2]], device=idx.device)
    #     # attention_mask_to_add = torch.ones(1, 1, device=idx.device)

    #     candidates = [
    #         {
    #             "input_ids": idx,
    #             "attention_mask": None,
    #             "src_tgt_mask": src_tgt_mask,
    #             "score": 0.0,
    #         }
    #     ]
    #     completed_candidates = []

    #     for i in range(max_new_tokens):
    #         new_candidates = []

    #         # start_time1 = time.time()
    #         for candidate in candidates:
    #             input_ids, attention_mask, src_tgt_mask, score = (
    #                 candidate["input_ids"],
    #                 candidate["attention_mask"],
    #                 candidate["src_tgt_mask"],
    #                 candidate["score"],
    #             )

    #             logits, _ = self(
    #                 idx=input_ids,
    #                 src_tgt_language_mask=src_tgt_mask,
    #                 attention_mask=None,
    #                 use_seperate_pos_enc=use_sep_pos_enc,
    #             )
    #             logits = torch.log_softmax(logits, dim=-1)
    #             top_beams = torch.topk(logits, top_k, dim=-1)

    #             # start_time1 = time.time()
    #             for j in range(top_k):
    #                 next_token = top_beams.indices[0, 0, j].item()
    #                 next_score = top_beams.values[0, 0, j].item()

    #                 if next_token == 289:
    #                     completed_candidates.append(
    #                         {
    #                             "input_ids": input_ids,
    #                             "attention_mask": attention_mask,
    #                             "src_tgt_mask": src_tgt_mask,
    #                             "score": score + next_score,
    #                         }
    #                     )
    #                 else:
    #                     new_input_ids = torch.cat(
    #                         (
    #                             input_ids,
    #                             torch.tensor([[next_token]], device=input_ids.device),
    #                         ),
    #                         dim=-1,
    #                     )
    #                     # new_attention_mask = torch.cat((attention_mask, attention_mask_to_add), dim=-1)
    #                     new_src_tgt_mask = torch.cat(
    #                         (src_tgt_mask, src_tgt_to_add), dim=-1
    #                     )
    #                     new_candidates.append(
    #                         {
    #                             "input_ids": new_input_ids,
    #                             "attention_mask": None,
    #                             "src_tgt_mask": new_src_tgt_mask,
    #                             "score": score + next_score,
    #                         }
    #                     )

    #             # Sort candidates by score and only keep top_k:
    #             new_candidates = sorted(
    #                 new_candidates, key=lambda x: x["score"], reverse=True
    #             )[:top_k]
    #         # end_time1 = time.time()
    #         # print(f"Beam search took: {end_time1 - start_time1} seconds")
    #         # Sort candidates by score:
    #         candidates = sorted(new_candidates, key=lambda x: x["score"], reverse=True)[
    #             :top_k
    #         ]

    #     # end_time1 = time.time()
    #     # exit()
    #     # Add remaining Candidates:
    #     completed_candidates.extend(candidates)

    #     # print(len(completed_candidates))
    #     # exit()

    #     # Sort Final Candidates:
    #     completed_candidates = sorted(
    #         completed_candidates, key=lambda x: x["score"], reverse=True
    #     )[:top_k]

    #     # print(f"Beam search took: {end_time1 - start_time1} seconds")

    #     # start_time_2 = time.time()
    #     # Now iterate over the saved samples:
    #     predicted_product = []
    #     predicted_scores = []
    #     for candidate in completed_candidates:
    #         input_ids, score = candidate["input_ids"], candidate["score"]
    #         product = []
    #         is_part_of_product = False
    #         for idx in input_ids[0]:
    #             label = label2token[idx.item()]
    #             if label == "[BEGP]":
    #                 is_part_of_product = True
    #             if label != "[ENDP]" and label != "[BEGP]" and is_part_of_product:
    #                 product.append(label)
    #             if label == "[ENDP]":
    #                 break
    #         # print("".join(product))
    #         predicted_product.append(product)
    #         predicted_scores.append(score)

    #     return predicted_product, predicted_scores
