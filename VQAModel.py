pip install evaluate
import torch
import skimage.io as io
import skimage.transform as transform
import torchvision
import clip
import pandas as pd
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import string
import random
import numpy as np
from transformers import set_seed, GPT2Config, AutoTokenizer

def isEglish(s):
    return s.isascii()

def punc(s):
    for c in string.punctuation:
        s=s.replace(c,"")
    return s.lower()

def update_classes(pkl_train, pkl_val, pkl_test):
    # standardize answer ids across datasets and compute the maximum number of generated output tokens based on the train set
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    with open(pkl_train, 'rb') as f:
            data_train = pickle.load(f)
    with open(pkl_val, 'rb') as f:
            data_val = pickle.load(f)
    with open(pkl_test, 'rb') as f:
            data_test = pickle.load(f)

    cur_id = 0
    class_names_list = []
    class_ids_list = [[],[],[]]

    for i, data in enumerate([data_train,data_val,data_test]):

        for answer in data['answers']:
            if answer not in class_names_list:
                class_names_list.append(answer)
                class_ids_list[i].append(cur_id)
                cur_id+=1
            else:
                class_ids_list[i].append(class_names_list.index(answer))
    q_lens = []
    a_lens = []
    for question in data_train['questions']:
        q_lens.append(len(tokenizer.encode(question)))
    for answer in data_train['answers']:
        a_lens.append(len(tokenizer.encode(str(answer))))

    data_train['class_ids'] = class_ids_list[0]
    data_val['class_ids'] = class_ids_list[1]
    data_test['class_ids'] = class_ids_list[2]

    data_train['class_names'] = class_names_list
    data_val['class_names'] = class_names_list
    data_test['class_names'] = class_names_list

    data_train['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data_val['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data_test['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))

    with open(pkl_train, 'wb') as f:
        pickle.dump(data_train,f)
    with open(pkl_val, 'wb') as f:
        pickle.dump(data_val,f)
    with open(pkl_test, 'wb') as f:
        pickle.dump(data_test,f)


def preprocess_slake(split, out_path):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open('../project/Slake1.0/{}.json'.format(split)) as f:
        data =  json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_img_prefixes = []
    img_ids = []
    img_paths = []
    all_questions = []
    all_answers = []
    img_dict = {}

    # preloading CLIP embeddings for images. Since multiple questions can be associated with one image we construct a dictionary with img ids
    # as keys for computational efficiency
    for i in tqdm(range(len(data))):
        d = data[i]
        if isEglish(d['answer']) and isEglish(d['question']):
            img_id = d["img_id"]
            filename = "../project/Slake1.0/imgs/"+d['img_name']
            with torch.no_grad():
                prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()
            if img_id not in img_dict.keys():
                img_dict[img_id] = [[d['question']],[d['answer']],prefix_i,filename]
            else:
                img_dict[img_id][0].append(d['question'])
                img_dict[img_id][1].append(d['answer'])
    # this dictionary is converted into a format that is sutiable for the data loader. Each data point contains a 'img_id', that corresponds is the index of the corresponding
    # CLIP embedding of the image in 'img_prefix'.
    for img_id, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][2])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_ids.append(img_id)
            img_paths.append(img_dict[imgs][2])

    all_data = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions,'answers': all_answers,'img_path': img_paths}

    with open(out_path, 'wb') as f:
        pickle.dump(all_data,f)
    print('Done')
    print("%0d embeddings saved " % len(all_questions))

if __name__=='__main__':
    for split in ['train','test','validate']:
        out_path = "../project/slake_pkl/{}.pkl".format(split)
        preprocess_slake(split,out_path)

    pkl_train = "../project/slake_pkl/train.pkl"
    pkl_val = "../project/slake_pkl/validate.pkl"  # Replace with your val data path
    pkl_test = "../project/slake_pkl/test.pkl"


    update_classes(pkl_train, pkl_val, pkl_test)

"""###Train"""

from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tqdm import tqdm
import copy
import os
import numpy as np
import time
import random
import torch.nn as nn
import torch.nn.functional as nnf
import os
import numpy as np
import random
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as nnf
from accelerate import Accelerator
import pdb

def pytorch_model_run(train_loader, valid_loader, model_obj, args):
    accelerator = Accelerator()
    device = accelerator.device

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = model_obj.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    valid_loader = accelerator.prepare(valid_loader)

    best_valid_loss = float("inf")
    counter = 0
    n_epochs = args.epochs
    accelerator.wait_for_everyone()

    def calculate_metrics(logits, targets, tokenizer):
        """Calculate accuracy and return token comparisons"""
        preds = torch.argmax(logits, dim=-1)

        # Calculate accuracy
        correct = (preds == targets).sum().item()
        total = targets.numel() - (targets == 0).sum().item()  # exclude padding
        accuracy = correct / max(total, 1)

        # Get token-level comparisons
        pred_tokens = [tokenizer.decode([t]) for t in preds.flatten().cpu().numpy()]
        target_tokens = [tokenizer.decode([t]) for t in targets.flatten().cpu().numpy()]

        return accuracy, pred_tokens, target_tokens

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        token_comparisons = []

        with tqdm(train_loader, desc=f"Epoch {epoch}") as epoch_pbar:
            for i, (prefix, labels, tokens, mask, q_len) in enumerate(epoch_pbar):
                with accelerator.accumulate(model):
                    prefix = prefix.type(torch.float32)
                    tokens = tokens.type(torch.long)
                    mask = mask.type(torch.long)
                    q_len = q_len.type(torch.long)

                    outputs = model(prefix, labels, tokens, mask, q_len, batch_size=args.batch_size)
                    logits = outputs.logits
                    loss = 0.
                    batch_acc = 0.
                    batch_token_comps = []

                    shift = 10 if args.setting in ["p_tuning", "prompttuning"] else 0

                    for b in range(logits.size(0)):
                        # Get relevant portions (excluding question and visual prefix)
                        start_idx = q_len[b] + model.prefix_length + 1
                        condensed_tokens = tokens[b, start_idx:]
                        condensed_logits = logits[b, shift+start_idx-1:-1]  # -1 for proper alignment

                        # Calculate metrics
                        acc, pred_tokens, target_tokens = calculate_metrics(
                            condensed_logits.reshape(-1, logits.shape[-1]),
                            condensed_tokens.flatten(),
                            model.tokenizer
                        )
                        batch_acc += acc
                        batch_token_comps.append((pred_tokens, target_tokens))

                        # Loss calculation
                        loss += nnf.cross_entropy(
                            condensed_logits.reshape(-1, logits.shape[-1]),
                            condensed_tokens.flatten(),
                            ignore_index=0
                        )

                    loss = loss / logits.size(0)
                    batch_acc = batch_acc / logits.size(0)

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    total_acc += batch_acc
                    total_samples += 1
                    token_comparisons.extend(batch_token_comps)

                    epoch_pbar.set_postfix({
                        'loss': f"{total_loss/total_samples:.4f}",
                        'acc': f"{total_acc/total_samples:.4f}"
                    })

        # Print sample token comparisons every few epochs
        if epoch % 2 == 0:
            print("\nSample token comparisons (Predicted -> Actual):")
            for i, (pred, target) in enumerate(token_comparisons[:3]):
                print(f"\nExample {i+1}:")
                print("Predicted:", " ".join(pred))
                print("Actual:   ", " ".join(target))
                print("Match:    ", " ".join("✓" if p == t else "✗" for p, t in zip(pred, target)))

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_samples = 0
        val_token_comparisons = []

        with torch.no_grad():
            for i, (prefix, labels, tokens, mask, q_len) in enumerate(valid_loader):
                prefix = prefix.type(torch.float32)
                tokens = tokens.type(torch.long)
                mask = mask.type(torch.long)
                q_len = q_len.type(torch.long)

                outputs = model(prefix, labels, tokens, mask, q_len, batch_size=args.batch_size)
                logits = outputs.logits
                batch_loss = 0.
                batch_acc = 0.
                batch_token_comps = []

                for b in range(logits.size(0)):
                    start_idx = q_len[b] + model.prefix_length + 1
                    condensed_tokens = tokens[b, start_idx:]
                    condensed_logits = logits[b, start_idx-1:-1]  # -1 for alignment

                    # Calculate metrics
                    acc, pred_tokens, target_tokens = calculate_metrics(
                        condensed_logits.reshape(-1, logits.shape[-1]),
                        condensed_tokens.flatten(),
                        model.tokenizer
                    )
                    batch_acc += acc
                    batch_token_comps.append((pred_tokens, target_tokens))

                    batch_loss += nnf.cross_entropy(
                        condensed_logits.reshape(-1, logits.shape[-1]),
                        condensed_tokens.flatten(),
                        ignore_index=0
                    )

                batch_loss = batch_loss / logits.size(0)
                batch_acc = batch_acc / logits.size(0)

                val_loss += batch_loss.item()
                val_acc += batch_acc
                val_samples += 1
                val_token_comparisons.extend(batch_token_comps)

        avg_val_loss = val_loss / val_samples
        avg_val_acc = val_acc / val_samples

        # Print validation token comparisons
        print("\nValidation token comparisons (Predicted -> Actual):")
        for i, (pred, target) in enumerate(val_token_comparisons[:3]):
            print(f"\nExample {i+1}:")
            print("Predicted:", " ".join(pred))
            print("Actual:   ", " ".join(target))
            print("Match:    ", " ".join("✓" if p == t else "✗" for p, t in zip(pred, target)))

        # Save best model
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"open_ended_latest.pt"))

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Train Loss: {total_loss/total_samples:.4f} | "
              f"Train Acc: {total_acc/total_samples:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {avg_val_acc:.4f}")

        # Early stopping
        if avg_val_loss > (total_loss/total_samples):
            counter += 1
            if counter >= 5:
                print("Early stopping triggered")
                break
        else:
            counter = 0

    return model

"""###Utils.py"""

import torch
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
def treebank_tokenize(s):
    return TreebankWordTokenizer().tokenize(s)
def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    generated=None,
    entry_length=65,
    temperature=1.0,
    stop_token: str = "<|endoftext|>",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            logits = logits.softmax(-1).log()
            # final_logit

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            if model.model_type == "biogpt":
                next_token_embed = model.gpt.biogpt.embed_tokens(
                    next_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            elif model.model_type == "gpt2":
                next_token_embed = model.gpt.transformer.embed_tokens(
                    next_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.gpt.get_input_embeddings()(tokens[:,-1])
                next_token_embed=next_token_embed.squeeze().view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

"""###Prefix_Mappers.py"""

# source: https://github.com/rmokady/CLIP_prefix_caption
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Tuple, Optional, Union
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.5))
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

"""###Predict.py"""

from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score,roc_auc_score
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer
import pdb
from evaluate import load
import collections
from torch.cuda.amp import autocast
import os

def eval_gpt_open_ended(model, dataset, args, print_vis_token_meaning=True):
    model.eval()
    model=model.cuda()
    bert_score = load("bertscore")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    bleu_avg1=0.
    bert_avg1 = 0.
    bert_avg2 = 0.
    bert_avg3 = 0.
    f1_avg = 0.
    acc = 0.
    acc_oe = 0.
    acc_yn = 0.
    c_oe =1e-9
    c_yn =1e-9
    with tqdm(total=len(dataset)) as epoch_pbar:
        epoch_pbar.set_description("Testing")
        for item in range(len(dataset)):
            prefix,  labels, tokens, mask, q_len = dataset[item]
            prefix = prefix.type(torch.float32).cuda()
            tokens = tokens.type(torch.long).cuda()
            mask = mask.cuda()
            with autocast(dtype=torch.float16):
              with torch.no_grad():
                  embed = model.generate(prefix,labels,tokens,mask,q_len).view(1,tokens.size(0),-1)
                  if print_vis_token_meaning:
                    prefix_projections = embed[:,q_len:q_len+model.prefix_length,:]
                    for i in range(prefix_projections.size(1)):
                      print_nearest_text_token(prefix_projections[0,i], model)
                  out_text = generate_beam(model, model.tokenizer,generated=embed,entry_length=dataset.max_seqs_len[1], temperature=1)[0]

            if out_text.lower()==dataset.answers_raw[item].lower():
              acc+=1
            if dataset.answers_raw[item].lower()=='yes' or dataset.answers_raw[item].lower()=='no':
              if out_text.lower()==dataset.answers_raw[item].lower():
                acc_yn+=1
              c_yn+=1
            else:
              if out_text.lower()==dataset.answers_raw[item].lower():
                acc_oe+=1
              c_oe+=1

            reference = [str(dataset.answers_raw[item])]
            candidate = [out_text]

            bleu_1 = sentence_bleu(reference[0], candidate[0], weights=(1, 0, 0, 0))

            a = bert_score.compute(references = reference,predictions = candidate,model_type = 'bert-base-uncased')
            bert_avg1+= a['precision'][0]
            bert_avg2+= a['recall'][0]
            bert_avg3+= a['f1'][0]


            f1_avg += compute_f1(tokenizer.encode(reference[0]),tokenizer.encode(candidate[0]))
            bleu_avg1+=bleu_1


    print('------------')
    print("BLEU {}".format(round(bleu_avg1/len(dataset),3)))
    print("BERTScore {}".format(round(bert_avg3/len(dataset),3)))
    print("F1 {}".format(round(f1_avg/len(dataset),3)))
    print("Accuracy {}".format(round(acc/len(dataset),3)))
    print("Accuracy YN{}".format(round(acc_yn/c_yn,3)))
    print("Accuracy OE{}".format(round(acc_oe/c_oe,3)))

def print_nearest_text_token(vis_token, model):
    """print the nearest token in the vocabulary to the given token through model.gpt.embeddings.weight"""
    embeddings = model.gpt.transformer.embed_tokens.weight
    distances = torch.norm(embeddings - vis_token, dim=1)
    nearest_token_idx = torch.argmin(distances)
    print(model.tokenizer.decode([nearest_token_idx.item()]))

def compute_f1(gold_toks, pred_toks):
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


import numpy as np
from tqdm import tqdm
import sys
import os
import pdb
from typing import Tuple, Optional, Union

from peft import LoraConfig, get_peft_model,get_peft_config,PeftModelForCausalLM,TaskType,PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig

import torch
import torch.nn as nn
from torch.nn import functional as nnf

import transformers
from transformers import set_seed, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from transformers.models.biogpt import BioGptForCausalLM, BioGptTokenizer, BioGptConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig

class VQAmedModel(nn.Module):
    def forward(self, prefix, labels, tokens, mask, q_len, batch_size=None):  # Make batch_size optional
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        if self.gpttype == 'microsoft/biogpt':
            embedding = self.gpt.transformer.embed_tokens(tokens)
        else:
            embedding = self.gpt.transformer.embed_tokens(tokens)

        # Get actual batch size from input tensor
        actual_batch_size = embedding.size(0)

        for b in range(actual_batch_size):  # Use actual batch size
            embedding[b, q_len[b]:q_len[b]+self.prefix_length, :] = prefix_projections[b]

        return self.gpt(inputs_embeds=embedding, attention_mask=mask)

    def generate(self, prefix, labels, tokens, mask, q_len):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
        if self.gpttype=='microsoft/biogpt':
            embedding_txt = self.gpt.transformer.embed_tokens(tokens)
        else:
            embedding_txt = self.gpt.transformer.embed_tokens(tokens)
        embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embedding_txt
    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="lora",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAmedModel, self).__init__()
        self.model_type = args.model_type
        gpttype = args.model_type
        self.gpttype = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=False,)
        # load the relevant fine-tuning strategy
        if setting == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.embed_tokens.weight.shape[1]
        if mapping_type == "MLP":
            self.clip_project = MLP((
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")


# adaptation of VQAmedModel for ablation studies
class VQAmedModel_abl(nn.Module):
    def forward(self, prefix, labels, tokens, mask, q_len, batch_size,abl):
        embeddings = self.gpt.transformer.embed_tokens(tokens)
        if abl=="replace_visual":
            for b in range(batch_size):
                embeddings[b,q_len[b]:q_len[b]+self.prefix_length,:] = self.nv_tokens[b]
        elif abl=="remove_question":
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
        elif abl=="swap":
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
        return self.gpt(inputs_embeds=embeddings, attention_mask=mask)

    def generate(self, prefix, labels, tokens, mask, q_len,abl):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
        embeddings = self.gpt.transformer.embed_tokens(tokens)
        if abl=="replace_visual":
            embeddings[q_len:q_len+self.prefix_length,:] = self.nv_tokens[0]
        elif abl=="remove_question":
            prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
            embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
        elif abl=="swap":
            prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
            embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embeddings

    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="frozen",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAmedModel_abl, self).__init__()
        gpttype = "roberta-base"
        self.model_type = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=False,)
        if setting == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.embed_tokens.weight.shape[1]
        # for the replace_visual ablation study we replace the visual tokens with learnable parameters
        self.nv_tokens = torch.nn.Parameter(torch.randn(args.batch_size,prefix_length,self.gpt_embedding_size),requires_grad=True).cuda()
        if mapping_type == "MLP":
            self.clip_project = MLP((prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")


from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from transformers import set_seed, GPT2Config, AutoTokenizer
from transformers import AutoTokenizer
from transformers.models.biogpt import BioGptTokenizer
import os
import pandas as pd
from torch.utils.data import Dataset
import pickle
from torch.utils.data import DataLoader, random_split
import numpy as np
import pdb

class medvqaDataset(Dataset):
    def __init__(self, path, split='train', like_test=False, prefix_length=2, model_type='gpt2'):
        super().__init__()
        data_path = path + split + '.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.img_ids = data["img_ids"]
        self.img_prefixes = data["img_prefix"]
        self.questions = data['questions']
        self.answers = data['answers']
        self.img_paths = data['img_path']
        self.max_seqs_len = data['max_seqs_len']
        self.labels = data['class_ids']
        self.train_setting = (split != 'test') and (not like_test)
        self.prefix_len = prefix_length

    def __len__(self):
        return len(self.answers)

    def pad_sequences(self, index):
        # Special tokens and their masks
        question_token = torch.tensor(self.tokenizer.encode('question: '))
        context_token = torch.tensor(self.tokenizer.encode(' context:'))
        answer_token = torch.tensor(self.tokenizer.encode('answer '))
        eos_token = torch.tensor(self.tokenizer.encode('<|endoftext|>'))

        question_mask = torch.ones(len(question_token))
        context_mask = torch.ones(len(context_token))
        answer_mask = torch.ones(len(answer_token))
        eos_mask = torch.zeros(len(eos_token))

        if self.train_setting:
            # Tokenize question and answer
            q_tokens = torch.tensor(self.tokenizer.encode(self.questions[index]))
            a_tokens = torch.tensor(self.tokenizer.encode(str(self.answers[index])))

            # Apply padding to question
            q_tokens, q_mask, leftover = self.make_padding(
                self.max_seqs_len[0], q_tokens, question=True
            )

            # Apply padding to answer
            a_tokens, a_mask, _ = self.make_padding(
                self.max_seqs_len[1], a_tokens, leftover_tokens=leftover
            )

            # Calculate question length (before visual prefix)
            q_len = len(question_token) + len(q_tokens) + len(context_token)

            # Handle answer padding and EOS token
            if len((a_tokens == 0).nonzero()) != 0:
                pad_start = (a_tokens == 0).nonzero()[0]
                a_tokens = torch.cat((a_tokens[:pad_start], eos_token, a_tokens[pad_start:]))
                a_mask = torch.cat((a_mask[:pad_start], eos_mask, a_mask[pad_start:]))
            else:
                a_tokens = torch.cat((a_tokens, eos_token))
                a_mask = torch.cat((a_mask, eos_mask))

            # Build full sequence
            visual_prefix = torch.ones(self.prefix_len)
            visual_mask = torch.ones(self.prefix_len)

            tokens = torch.cat([
                question_token,
                q_tokens,
                context_token,
                visual_prefix,
                answer_token,
                a_tokens
            ])

            mask = torch.cat([
                question_mask,
                q_mask,
                context_mask,
                visual_mask,
                answer_mask,
                a_mask
            ])

            # Verify shapes match
            assert tokens.shape == mask.shape, \
                f"Token and mask shape mismatch: {tokens.shape} vs {mask.shape}"

            return tokens, mask, q_len

        else:
            # Test mode processing
            q_tokens = torch.tensor(self.tokenizer.encode(self.questions[index]))
            q_tokens, q_mask, _ = self.make_padding_test_setting(
                self.max_seqs_len[0], q_tokens
            )

            q_len = len(question_token) + len(q_tokens) + len(context_token)
            visual_prefix = torch.ones(self.prefix_len)
            visual_mask = torch.ones(self.prefix_len)

            tokens = torch.cat([
                question_token,
                q_tokens,
                context_token,
                visual_prefix,
                answer_token
            ])

            mask = torch.cat([
                question_mask,
                q_mask,
                context_mask,
                visual_mask,
                answer_mask
            ])

            assert tokens.shape == mask.shape
            return tokens, mask, q_len

    def make_padding(self, max_len, tokens, question=False, leftover_tokens=0):
        current_len = tokens.size(0)
        padding_needed = max_len - current_len

        if padding_needed > 0:
            if question:
                # For questions, we keep the original tokens and track leftover space
                mask = torch.ones(current_len)
                leftover_tokens = padding_needed
            else:
                # For answers, apply padding with zeros
                tokens = torch.cat((tokens, torch.zeros(padding_needed + leftover_tokens)))
                mask = torch.cat((
                    torch.ones(current_len),
                    torch.zeros(padding_needed + leftover_tokens)
                ))
        elif padding_needed == 0:
            if question:
                mask = torch.ones(current_len)
            else:
                tokens = torch.cat((tokens, torch.zeros(leftover_tokens)))
                mask = torch.cat((torch.ones(current_len), torch.zeros(leftover_tokens)))
        else:  # padding_needed < 0
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len)
            else:
                tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
                mask = torch.cat((torch.ones(max_len), torch.zeros(leftover_tokens)))

        return tokens, mask, leftover_tokens

    def make_padding_test_setting(self, max_len, tokens, do_padding=False):
        current_len = tokens.size(0)
        padding_needed = max_len - current_len

        if padding_needed > 0:
            if do_padding:
                tokens = torch.cat((tokens, torch.zeros(padding_needed)))
                mask = torch.cat((torch.ones(current_len), torch.zeros(padding_needed)))
                padding_len = padding_needed
            else:
                mask = torch.ones(current_len)
                padding_len = 0
        elif padding_needed == 0:
            mask = torch.ones(current_len)
            padding_len = 0
        else:  # padding_needed < 0
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
            padding_len = 0

        return tokens, mask, padding_len

    def __getitem__(self, index):
        prefix = self.img_prefixes[self.img_ids[index]]
        tokens, mask, q_len = self.pad_sequences(index)
        return prefix, self.labels[index], tokens, mask, q_len


import torch
from torch.utils.data import DataLoader
import os
from argparse import Namespace

def get_default_args():
    """Returns a Namespace with default arguments (identical to original parser)"""
    return Namespace(
        model_type="roberta-base",                # Choices: ["roberta-base", "microsoft/biogpt", ...]
        setting="lora",                    # Choices: ["lora", "frozen", ...]
        ablation="none",                     # Choices: ["none", "remove_question", ...]
        mapping_type="MLP",                  # Choices: ["MLP", "Transformer"]
        prefix_length=1,                    # Match your .pkl files
        dataset_path="../project",
        batch_size=64,
        epochs=10,
        lr=0.001,
        warmup_steps=600,
        seed=42,
        iters_to_accumulate=4,
        validation_step=1000,
        out_dir="../project/checkpoints",
        checkpoint=None,
        eval=False,
        verbose=True,
        dataset="slake"                      # "pathvqa", "ovqa", or "slake"
    )

def main():
    # ==== Configure args directly here (modify as needed) ====
    args = get_default_args()

    # Example overrides (uncomment what you need):
    # args.model_type = "microsoft/biogpt"
    # args.setting = "lora"
    # args.ablation = "remove_visual"
    # args.eval = True
    # args.checkpoint = "./checkpoints/open_ended_latest.pt"
    # ========================================================

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Prepare datasets
    train_dataset = medvqaDataset(
        os.path.join(args.dataset_path, args.dataset + '/'),
        split="train",
        prefix_length=args.prefix_length,
        model_type=args.model_type
    )
    val_dataset = medvqaDataset(
        os.path.join(args.dataset_path, args.dataset + '/'),
        split="validate",
        prefix_length=args.prefix_length,
        model_type=args.model_type
    )
    test_dataset = medvqaDataset(
        os.path.join(args.dataset_path, args.dataset + '/'),
        split="test",
        prefix_length=args.prefix_length,
        model_type=args.model_type,
        like_test=True
    )

    # Initialize model
    if args.ablation != "none":
        model = VQAmedModel_abl(
            prefix_length=args.prefix_length,
            clip_length=4,
            setting=args.setting,
            mapping_type=args.mapping_type,
            args=args  # Pass Namespace directly
        )
    else:
        model = VQAmedModel(
            prefix_length=args.prefix_length,
            clip_length=4,
            setting=args.setting,
            mapping_type=args.mapping_type,
            args=args  # Pass Namespace directly
        )

    # Train or evaluate
    if not args.eval:
        pytorch_model_run(
            DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=args.batch_size),
            model,
            args
        )
    else:
        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        eval_gpt_open_ended(model, test_dataset, args)

if __name__ == "__main__":
    main()

import json
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import clip
from transformers import AutoTokenizer


# ─── ② adjust these two paths to your environment ─────────────────────────────
xml_folder = Path("../project/Slake1.0/imgs/xmlab1")
ckpt_path  = Path("../project/checkpoints/open_ended_latest.pt")
# ────────────────────────────────────────────────────────────────────────────────

# 1) Setup device, CLIP & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# 2) Load your fine‑tuned VQAmedModel
model = VQAmedModel(
    prefix_length=1,
    clip_length=4,
    setting="frozen",
    mapping_type="MLP",
    args=type("A", (), {"model_type":"roberta-base","batch_size":1})()
)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
model.to(device).eval()

# 3) Load question + ground truth (handle list or dict)
raw = json.loads((xml_folder/"question.json").read_text())
entry = raw[0] if isinstance(raw, list) else raw
question   = entry["question"]
ground_ans = entry.get("answer", "<no‑answer‑found>")

# 4) Display the input image
img = Image.open(xml_folder/"source.jpg").convert("RGB")
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()

print(f"Question:      {question}")
print(f"Ground truth:  {ground_ans}\n")

# 5) Compute CLIP prefix (float16 → float32)
with torch.no_grad():
    clip_prefix = clip_model.encode_image(
        clip_preprocess(img).unsqueeze(0).to(device)
    ).squeeze(0).to(torch.float32)    # → shape (512,)

# 6) Tokenize prompt
prompt = f"question: {question} context:"
q_ids  = tokenizer.encode(prompt)
q_len  = len(q_ids)

# 7) Build text embeddings
text_emb = model.gpt.transformer.embed_tokens(
    torch.tensor(q_ids, device=device)
).unsqueeze(0)      # → (1, q_len, D)

# 8) Project visual prefix into GPT space
with torch.no_grad():
    #  → (1, D * prefix_length)
    prefix_proj_flat = model.clip_project(clip_prefix.unsqueeze(0))
    # reshape to (1, prefix_length, D)
    prefix_proj = prefix_proj_flat.view(
        1,
        model.prefix_length,
        model.gpt_embedding_size
    )
# 9) Concatenate [text prompt] + [visual prefix]
inputs_embeds = torch.cat([text_emb, prefix_proj], dim=1)
# shape: (1, q_len + prefix_length, D)

# 10) Greedy generation from GPT
out_ids = model.gpt.generate(
    inputs_embeds=inputs_embeds,
    max_new_tokens=30,
    eos_token_id=tokenizer.eos_token_id,
)

# 11) Decode only the newly generated tokens
generated_ids = out_ids[0, inputs_embeds.size(1):]
prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print(f"Model’s answer: {prediction}")

# cell: one‐shot inference + display for a single test example

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 1) Paths ─ adjust these to your setup
DATA_ROOT     = "../project"                        # root of your project
PICKLE_DIR    = os.path.join(DATA_ROOT, "slake") + os.sep  # must contain train.pkl, validate.pkl, test.pkl
CHECKPOINT    = os.path.join(DATA_ROOT, "checkpoints",
                             "open_ended_latest.pt")          # your saved model
PREFIX_LENGTH = 1                                              # same as you used in preprocessing
MODEL_TYPE    = "roberta-base"                                      # or "microsoft/biogpt", etc.

test_dataset = medvqaDataset(
    path=PICKLE_DIR,
    split="test",
    prefix_length=PREFIX_LENGTH,
    model_type=MODEL_TYPE
)

class Args: pass
args = Args()
args.model_type   = MODEL_TYPE
args.setting      = "lora"
args.mapping_type = "MLP"
args.prefix_length= PREFIX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQAmedModel(PREFIX_LENGTH, clip_length=4, setting=args.setting,
                    mapping_type=args.mapping_type, args=args)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device).eval()

# ── 3) Pick one example ──────────────────────────────
idx = 0
prefix, label, tokens, mask, q_len = test_dataset[idx]
# group_id = test_dataset.img_ids[idx]   # this is the xmllab folder index
xml_folder = f"xmlab42"
# rebuild the image path manually:
img_folder = os.path.join(
    DATA_ROOT,
    "Slake1.0",
    "imgs",
    xml_folder
)
img_file = os.path.join(img_folder, "source.jpg")
print(img_file)


# ── 4) Move to device & generate ─────────────────────
prefix = prefix.to(device).float().unsqueeze(0)
tokens = tokens.to(device).long().unsqueeze(0)
mask   = mask.to(device).long().unsqueeze(0)
q_len_t= torch.tensor([q_len], device=device)

with torch.no_grad():
    emb   = model.generate(prefix, label, tokens.squeeze(0), mask.squeeze(0), q_len)
    emb   = emb.unsqueeze(0)
    beams = generate_beam(
        model=model,
        tokenizer=model.tokenizer,
        beam_size=5,
        generated=emb,
        entry_length=test_dataset.max_seqs_len[1],
        temperature=1.0,
        stop_token="<|endoftext|>"
    )

# ── 5) Display ───────────────────────────────────────
img      = Image.open(img_file).convert("RGB")
question = test_dataset.questions[idx]
gt       = test_dataset.answers[idx]
pred     = beams[0]

plt.figure(figsize=(6,6))
plt.imshow(img); plt.axis("off")
plt.title(f"Q: {question}\nGT: {gt}\nPred: {pred}", wrap=True)
plt.show()

print("All beam candidates:", beams)

!pip install pylint

!pyreverse -o jpg -p MedicalVQA -a 1 -fmodell.py

!apt-get update -qq && apt-get install -y graphviz

