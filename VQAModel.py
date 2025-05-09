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

def assign_class_ids(slake_train_pkl, slake_val_pkl, slake_test_pkl):
    # standardize answer ids across entriessets and compute the maximum number of generated output input_tokens based on the train set
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    with open(slake_train_pkl, 'rb') as f:
            entries_train = pickle.load(f)
    with open(slake_val_pkl, 'rb') as f:
            entries_val = pickle.load(f)
    with open(slake_test_pkl, 'rb') as f:
            entries_test = pickle.load(f)

    cur_id = 0
    class_names_list = []
    class_ids_list = [[],[],[]]

    for i, entries in enumerate([entries_train,entries_val,entries_test]):

        for answer in entries['answers']:
            if answer not in class_names_list:
                class_names_list.append(answer)
                class_ids_list[i].append(cur_id)
                cur_id+=1
            else:
                class_ids_list[i].append(class_names_list.index(answer))
    question_lengthss = []
    a_lens = []
    for question in entries_train['questions']:
        question_lengthss.append(len(tokenizer.encode(question)))
    for answer in entries_train['answers']:
        a_lens.append(len(tokenizer.encode(str(answer))))

    entries_train['class_ids'] = class_ids_list[0]
    entries_val['class_ids'] = class_ids_list[1]
    entries_test['class_ids'] = class_ids_list[2]

    entries_train['class_names'] = class_names_list
    entries_val['class_names'] = class_names_list
    entries_test['class_names'] = class_names_list

    entries_train['max_seqs_len']=(int(np.mean(question_lengthss)+2*np.std(question_lengthss)),int(np.mean(a_lens)+2*np.std(a_lens)))
    entries_val['max_seqs_len']=(int(np.mean(question_lengthss)+2*np.std(question_lengthss)),int(np.mean(a_lens)+2*np.std(a_lens)))
    entries_test['max_seqs_len']=(int(np.mean(question_lengthss)+2*np.std(question_lengthss)),int(np.mean(a_lens)+2*np.std(a_lens)))

    with open(slake_train_pkl, 'wb') as f:
        pickle.dump(entries_train,f)
    with open(slake_val_pkl, 'wb') as f:
        pickle.dump(entries_val,f)
    with open(slake_test_pkl, 'wb') as f:
        pickle.dump(entries_test,f)


def prepare_slake_entries(split, output_file_path):
    device = torch.device('cuda:0')
    vision_encoder, image_preprocessor = clip.load("ViT-B/32", device=device, jit=False)
    with open('../project/Slake1.0/{}.json'.format(split)) as f:
        entries =  json.load(f)
    print("%0d captions loaded from json " % len(entries))
    all_visual_embeddings = []
    image_indexs = []
    image_locations = []
    question_list = []
    answer_list = []
    image_entry_map = {}

    # preloading CLIP embeddings for images. Since multiple questions can be associated with one image we construct a dictionary with img ids
    # as keys for computational efficiency
    for i in tqdm(range(len(entries))):
        d = entries[i]
        if isEglish(d['answer']) and isEglish(d['question']):
            image_index = d["image_index"]
            filename = "../project/Slake1.0/imgs/"+d['image_file']
            with torch.no_grad():
                visual_features = vision_encoder.encode_image(image_preprocessor(Image.open(filename)).unsqueeze(0).to(device)).cpu()
            if image_index not in image_entry_map.keys():
                image_entry_map[image_index] = [[d['question']],[d['answer']],visual_features,filename]
            else:
                image_entry_map[image_index][0].append(d['question'])
                image_entry_map[image_index][1].append(d['answer'])
    # this dictionary is converted into a format that is sutiable for the entries loader. Each entries point contains a 'image_index', that corresponds is the index of the corresponding
    # CLIP embedding of the image in 'img_visual_prefix'.
    for image_index, imgs in enumerate(image_entry_map.keys()):
        all_visual_embeddings.append(image_entry_map[imgs][2])
        for q in range(len(image_entry_map[imgs][0])):
            question_list.append(image_entry_map[imgs][0][q])
            answer_list.append(image_entry_map[imgs][1][q])
            image_indexs.append(image_index)
            image_locations.append(image_entry_map[imgs][2])

    all_entries = {"img_visual_prefix": torch.cat(all_visual_embeddings, dim=0), "image_indexs": image_indexs, "questions": question_list,'answers': answer_list,'img_path': image_locations}

    with open(output_file_path, 'wb') as f:
        pickle.dump(all_entries,f)
    print('Done')
    print("%0d embeddings saved " % len(question_list))

if __name__=='__run_pipeline__':
    for split in ['train','test','validate']:
        output_file_path = "../project/slake_pkl/{}.pkl".format(split)
        prepare_slake_entries(split,output_file_path)

    slake_train_pkl = "../project/slake_pkl/train.pkl"
    slake_val_pkl = "../project/slake_pkl/validate.pkl"  # Replace with your val entries path
    slake_test_pkl = "../project/slake_pkl/test.pkl"


    assign_class_ids(slake_train_pkl, slake_val_pkl, slake_test_pkl)

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
        pred_input_tokens = [tokenizer.decode([t]) for t in preds.flatten().cpu().numpy()]
        target_input_tokens = [tokenizer.decode([t]) for t in targets.flatten().cpu().numpy()]

        return accuracy, pred_input_tokens, target_input_tokens

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        token_comparisons = []

        with tqdm(train_loader, desc=f"Epoch {epoch}") as epoch_pbar:
            for i, (visual_prefix, target_labels, input_tokens, attention_mask, question_lengths) in enumerate(epoch_pbar):
                with accelerator.accumulate(model):
                    visual_prefix = visual_prefix.type(torch.float32)
                    input_tokens = input_tokens.type(torch.long)
                    attention_mask = attention_mask.type(torch.long)
                    question_lengths = question_lengths.type(torch.long)

                    outputs = model(visual_prefix, target_labels, input_tokens, attention_mask, question_lengths, batch_size=args.batch_size)
                    logits = outputs.logits
                    loss = 0.
                    batch_acc = 0.
                    batch_token_comps = []

                    shift = 10 if args.finetune_mode in ["p_tuning", "prompttuning"] else 0

                    for b in range(logits.size(0)):
                        # Get relevant portions (excluding question and visual visual_prefix)
                        start_idx = question_lengths[b] + model.visual_prefix_length + 1
                        condensed_input_tokens = input_tokens[b, start_idx:]
                        condensed_logits = logits[b, shift+start_idx-1:-1]  # -1 for proper alignment

                        # Calculate metrics
                        acc, pred_input_tokens, target_input_tokens = calculate_metrics(
                            condensed_logits.reshape(-1, logits.shape[-1]),
                            condensed_input_tokens.flatten(),
                            model.tokenizer
                        )
                        batch_acc += acc
                        batch_token_comps.append((pred_input_tokens, target_input_tokens))

                        # Loss calculation
                        loss += nnf.cross_entropy(
                            condensed_logits.reshape(-1, logits.shape[-1]),
                            condensed_input_tokens.flatten(),
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
            for i, (visual_prefix, target_labels, input_tokens, attention_mask, question_lengths) in enumerate(valid_loader):
                visual_prefix = visual_prefix.type(torch.float32)
                input_tokens = input_tokens.type(torch.long)
                attention_mask = attention_mask.type(torch.long)
                question_lengths = question_lengths.type(torch.long)

                outputs = model(visual_prefix, target_labels, input_tokens, attention_mask, question_lengths, batch_size=args.batch_size)
                logits = outputs.logits
                batch_loss = 0.
                batch_acc = 0.
                batch_token_comps = []

                for b in range(logits.size(0)):
                    start_idx = question_lengths[b] + model.visual_prefix_length + 1
                    condensed_input_tokens = input_tokens[b, start_idx:]
                    condensed_logits = logits[b, start_idx-1:-1]  # -1 for alignment

                    # Calculate metrics
                    acc, pred_input_tokens, target_input_tokens = calculate_metrics(
                        condensed_logits.reshape(-1, logits.shape[-1]),
                        condensed_input_tokens.flatten(),
                        model.tokenizer
                    )
                    batch_acc += acc
                    batch_token_comps.append((pred_input_tokens, target_input_tokens))

                    batch_loss += nnf.cross_entropy(
                        condensed_logits.reshape(-1, logits.shape[-1]),
                        condensed_input_tokens.flatten(),
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
    input_tokens = None
    scores = None
    device = next(model.parameters()).device
    sequestion_lengthsgths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            logits = logits.softmax(-1).log()
            # final_logit

            if scores is None:
                scores, next_input_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_input_tokens, scores = next_input_tokens.permute(1, 0), scores.squeeze(0)
                if input_tokens is None:
                    input_tokens = next_input_tokens
                else:
                    input_tokens = input_tokens.expand(beam_size, *input_tokens.shape[1:])
                    input_tokens = torch.cat((input_tokens, next_input_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                sequestion_lengthsgths[~is_stopped] += 1
                scores_sum_average = scores_sum / sequestion_lengthsgths[:, None]
                scores_sum_average, next_input_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_input_tokens_source = next_input_tokens // scores_sum.shape[1]
                sequestion_lengthsgths = sequestion_lengthsgths[next_input_tokens_source]
                next_input_tokens = next_input_tokens % scores_sum.shape[1]
                next_input_tokens = next_input_tokens.unsqueeze(1)
                input_tokens = input_tokens[next_input_tokens_source]
                input_tokens = torch.cat((input_tokens, next_input_tokens), dim=1)
                generated = generated[next_input_tokens_source]
                scores = scores_sum_average * sequestion_lengthsgths
                is_stopped = is_stopped[next_input_tokens_source]
            if model.language_model == "biogpt":
                next_token_embed = model.gpt.biogpt.embed_input_tokens(
                    next_input_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            elif model.language_model == "gpt2":
                next_token_embed = model.gpt.transformer.embed_input_tokens(
                    next_input_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.gpt.get_input_embeddings()(input_tokens[:,-1])
                next_token_embed=next_token_embed.squeeze().view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_input_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / sequestion_lengthsgths
    output_list = input_tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, sequestion_lengthsgths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


# source: https://github.com/rmokady/CLIP_visual_prefix_caption
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

    def forward(self, x, y=None, attention_mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)
            attention = attention.attention_masked_fill(attention_mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, attention_mask=None):
        x_, attention = self.attn(self.norm1(x), y, attention_mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, attention_mask=None):
        x = x + self.attn(self.norm1(x), y, attention_mask)[0]
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

    def forward_with_attention(self, x, y=None, attention_mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, attention_mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, attention_mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, attention_mask)
            else:  # self or cross
                x = layer(x, y, attention_mask)
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
        visual_prefix = self.visual_prefix_const.unsqueeze(0).expand(x.shape[0], *self.visual_prefix_const.shape)
        visual_prefix = torch.cat((x, visual_prefix), dim=1)
        out = self.transformer(visual_prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, visual_prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.visual_prefix_const = nn.Parameter(torch.randn(visual_prefix_length, dim_embedding), requires_grad=True)

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

def eval_gpt_open_ended(model, entriesset, args, print_vis_token_meaning=True):
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
    with tqdm(total=len(entriesset)) as epoch_pbar:
        epoch_pbar.set_description("Testing")
        for item in range(len(entriesset)):
            visual_prefix,  target_labels, input_tokens, attention_mask, question_lengths = entriesset[item]
            visual_prefix = visual_prefix.type(torch.float32).cuda()
            input_tokens = input_tokens.type(torch.long).cuda()
            attention_mask = attention_mask.cuda()
            with autocast(dtype=torch.float16):
              with torch.no_grad():
                  embed = model.generate(visual_prefix,target_labels,input_tokens,attention_mask,question_lengths).view(1,input_tokens.size(0),-1)
                  if print_vis_token_meaning:
                    visual_prefix_projections = embed[:,question_lengths:question_lengths+model.visual_prefix_length,:]
                    for i in range(visual_prefix_projections.size(1)):
                      print_nearest_text_token(visual_prefix_projections[0,i], model)
                  out_text = generate_beam(model, model.tokenizer,generated=embed,entry_length=entriesset.max_seqs_len[1], temperature=1)[0]

            if out_text.lower()==entriesset.answers_raw[item].lower():
              acc+=1
            if entriesset.answers_raw[item].lower()=='yes' or entriesset.answers_raw[item].lower()=='no':
              if out_text.lower()==entriesset.answers_raw[item].lower():
                acc_yn+=1
              c_yn+=1
            else:
              if out_text.lower()==entriesset.answers_raw[item].lower():
                acc_oe+=1
              c_oe+=1

            reference = [str(entriesset.answers_raw[item])]
            candidate = [out_text]

            bleu_1 = sentence_bleu(reference[0], candidate[0], weights=(1, 0, 0, 0))

            a = bert_score.compute(references = reference,predictions = candidate,language_model = 'bert-base-uncased')
            bert_avg1+= a['precision'][0]
            bert_avg2+= a['recall'][0]
            bert_avg3+= a['f1'][0]


            f1_avg += compute_f1(tokenizer.encode(reference[0]),tokenizer.encode(candidate[0]))
            bleu_avg1+=bleu_1


    print('------------')
    print("BLEU {}".format(round(bleu_avg1/len(entriesset),3)))
    print("BERTScore {}".format(round(bert_avg3/len(entriesset),3)))
    print("F1 {}".format(round(f1_avg/len(entriesset),3)))
    print("Accuracy {}".format(round(acc/len(entriesset),3)))
    print("Accuracy YN{}".format(round(acc_yn/c_yn,3)))
    print("Accuracy OE{}".format(round(acc_oe/c_oe,3)))

def print_nearest_text_token(vis_token, model):
    """print the nearest token in the vocabulary to the given token through model.gpt.embeddings.weight"""
    embeddings = model.gpt.transformer.embed_input_tokens.weight
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

class VisualQAModel(nn.Module):
    def forward(self, visual_prefix, target_labels, input_tokens, attention_mask, question_lengths, batch_size=None):  # Make batch_size optional
        visual_prefix_projections = self.clip_project(visual_prefix).view(-1, self.visual_prefix_length, self.gpt_embedding_size)

        if self.gpttype == 'microsoft/biogpt':
            embedding = self.gpt.transformer.embed_input_tokens(input_tokens)
        else:
            embedding = self.gpt.transformer.embed_input_tokens(input_tokens)

        # Get actual batch size from input tensor
        actual_batch_size = embedding.size(0)

        for b in range(actual_batch_size):  # Use actual batch size
            embedding[b, question_lengths[b]:question_lengths[b]+self.visual_prefix_length, :] = visual_prefix_projections[b]

        return self.gpt(inputs_embeds=embedding, attention_attention_mask=attention_mask)

    def generate(self, visual_prefix, target_labels, input_tokens, attention_mask, question_lengths):
        visual_prefix_projections = self.clip_project(visual_prefix.view(1, -1)).view(self.visual_prefix_length, self.gpt_embedding_size)
        if self.gpttype=='microsoft/biogpt':
            embedding_txt = self.gpt.transformer.embed_input_tokens(input_tokens)
        else:
            embedding_txt = self.gpt.transformer.embed_input_tokens(input_tokens)
        embedding_txt[question_lengths:question_lengths+self.visual_prefix_length,:] = visual_prefix_projections
        return embedding_txt
    def __init__(
        self,
        visual_prefix_length=2,
        clip_length=2,
        visual_prefix_size=512,
        num_layers=8,
        finetune_mode="lora",
        mapping_type="MLP",
        args=None,
    ):
        super(VisualQAModel, self).__init__()
        self.language_model = args.language_model
        gpttype = args.language_model
        self.gpttype = gpttype
        self.finetune_mode = finetune_mode
        self.visual_prefix_length = visual_prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=False,)
        # load the relevant fine-tuning strategy
        if finetune_mode == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=="visual_prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_input_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_input_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_input_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.embed_input_tokens.weight.shape[1]
        if mapping_type == "MLP":
            self.clip_project = MLP((
                    visual_prefix_size,
                    (self.gpt_embedding_size * visual_prefix_length) // 2,
                    self.gpt_embedding_size * visual_prefix_length,
                    self.gpt_embedding_size * visual_prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                visual_prefix_size,
                self.gpt_embedding_size,
                visual_prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")


# adaptation of VisualQAModel for ablation studies
class VisualQAModel_abl(nn.Module):
    def forward(self, visual_prefix, target_labels, input_tokens, attention_mask, question_lengths, batch_size,abl):
        embeddings = self.gpt.transformer.embed_input_tokens(input_tokens)
        if abl=="replace_visual":
            for b in range(batch_size):
                embeddings[b,question_lengths[b]:question_lengths[b]+self.visual_prefix_length,:] = self.nv_input_tokens[b]
        elif abl=="remove_question":
            visual_prefix_projections = self.clip_project(visual_prefix).view(-1, self.visual_prefix_length, self.gpt_embedding_size)
            embeddings[:,question_lengths[0]:question_lengths[0]+self.visual_prefix_length,:] = visual_prefix_projections
        elif abl=="swap":
            visual_prefix_projections = self.clip_project(visual_prefix).view(-1, self.visual_prefix_length, self.gpt_embedding_size)
            embeddings[:,question_lengths[0]:question_lengths[0]+self.visual_prefix_length,:] = visual_prefix_projections
        return self.gpt(inputs_embeds=embeddings, attention_attention_mask=attention_mask)

    def generate(self, visual_prefix, target_labels, input_tokens, attention_mask, question_lengths,abl):
        visual_prefix_projections = self.clip_project(visual_prefix.view(1, -1)).view(self.visual_prefix_length, self.gpt_embedding_size)
        embeddings = self.gpt.transformer.embed_input_tokens(input_tokens)
        if abl=="replace_visual":
            embeddings[question_lengths:question_lengths+self.visual_prefix_length,:] = self.nv_input_tokens[0]
        elif abl=="remove_question":
            visual_prefix_projections = self.clip_project(visual_prefix.view(1, -1)).view(self.visual_prefix_length, self.gpt_embedding_size)
            embeddings[question_lengths:question_lengths+self.visual_prefix_length,:] = visual_prefix_projections
        elif abl=="swap":
            visual_prefix_projections = self.clip_project(visual_prefix.view(1, -1)).view(self.visual_prefix_length, self.gpt_embedding_size)
            embeddings[question_lengths:question_lengths+self.visual_prefix_length,:] = visual_prefix_projections
        return embeddings

    def __init__(
        self,
        visual_prefix_length=2,
        clip_length=2,
        visual_prefix_size=512,
        num_layers=8,
        finetune_mode="frozen",
        mapping_type="MLP",
        args=None,
    ):
        super(VisualQAModel_abl, self).__init__()
        gpttype = "roberta-base"
        self.language_model = gpttype
        self.finetune_mode = finetune_mode
        self.visual_prefix_length = visual_prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=False,)
        if finetune_mode == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=="visual_prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_input_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_input_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_input_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif finetune_mode=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.embed_input_tokens.weight.shape[1]
        # for the replace_visual ablation study we replace the visual input_tokens with learnable parameters
        self.nv_input_tokens = torch.nn.Parameter(torch.randn(args.batch_size,visual_prefix_length,self.gpt_embedding_size),requires_grad=True).cuda()
        if mapping_type == "MLP":
            self.clip_project = MLP((visual_prefix_size,
                    (self.gpt_embedding_size * visual_prefix_length) // 2,
                    self.gpt_embedding_size * visual_prefix_length,
                    self.gpt_embedding_size * visual_prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                visual_prefix_size,
                self.gpt_embedding_size,
                visual_prefix_length,
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
from torch.utils.entries import Dataset
import pickle
from torch.utils.entries import DataLoader, random_split
import numpy as np
import pdb

class MedicalVQADataset(Dataset):
    def __init__(self, path, split='train', like_test=False, visual_prefix_length=2, language_model='gpt2'):
        super().__init__()
        entries_path = path + split + '.pkl'
        with open(entries_path, 'rb') as f:
            entries = pickle.load(f)

        self.language_model = language_model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.image_indexs = entries["image_indexs"]
        self.visual_embeddings = entries["img_visual_prefix"]
        self.questions = entries['questions']
        self.answers = entries['answers']
        self.image_locations = entries['img_path']
        self.max_seqs_len = entries['max_seqs_len']
        self.target_labels = entries['class_ids']
        self.train_finetune_mode = (split != 'test') and (not like_test)
        self.visual_prefix_len = visual_prefix_length

    def __len__(self):
        return len(self.answers)

    def pad_sequences(self, index):
        # Special input_tokens and their attention_masks
        question_token = torch.tensor(self.tokenizer.encode('question: '))
        context_token = torch.tensor(self.tokenizer.encode(' context:'))
        answer_token = torch.tensor(self.tokenizer.encode('answer '))
        eos_token = torch.tensor(self.tokenizer.encode('<|endoftext|>'))

        question_attention_mask = torch.ones(len(question_token))
        context_attention_mask = torch.ones(len(context_token))
        answer_attention_mask = torch.ones(len(answer_token))
        eos_attention_mask = torch.zeros(len(eos_token))

        if self.train_finetune_mode:
            # Tokenize question and answer
            q_input_tokens = torch.tensor(self.tokenizer.encode(self.questions[index]))
            a_input_tokens = torch.tensor(self.tokenizer.encode(str(self.answers[index])))

            # Apply padding to question
            q_input_tokens, q_attention_mask, leftover = self.make_padding(
                self.max_seqs_len[0], q_input_tokens, question=True
            )

            # Apply padding to answer
            a_input_tokens, a_attention_mask, _ = self.make_padding(
                self.max_seqs_len[1], a_input_tokens, leftover_input_tokens=leftover
            )

            # Calculate question length (before visual visual_prefix)
            question_lengths = len(question_token) + len(q_input_tokens) + len(context_token)

            # Handle answer padding and EOS token
            if len((a_input_tokens == 0).nonzero()) != 0:
                pad_start = (a_input_tokens == 0).nonzero()[0]
                a_input_tokens = torch.cat((a_input_tokens[:pad_start], eos_token, a_input_tokens[pad_start:]))
                a_attention_mask = torch.cat((a_attention_mask[:pad_start], eos_attention_mask, a_attention_mask[pad_start:]))
            else:
                a_input_tokens = torch.cat((a_input_tokens, eos_token))
                a_attention_mask = torch.cat((a_attention_mask, eos_attention_mask))

            # Build full sequence
            visual_visual_prefix = torch.ones(self.visual_prefix_len)
            visual_attention_mask = torch.ones(self.visual_prefix_len)

            input_tokens = torch.cat([
                question_token,
                q_input_tokens,
                context_token,
                visual_visual_prefix,
                answer_token,
                a_input_tokens
            ])

            attention_mask = torch.cat([
                question_attention_mask,
                q_attention_mask,
                context_attention_mask,
                visual_attention_mask,
                answer_attention_mask,
                a_attention_mask
            ])

            # Verify shapes match
            assert input_tokens.shape == attention_mask.shape, \
                f"Token and attention_mask shape mismatch: {input_tokens.shape} vs {attention_mask.shape}"

            return input_tokens, attention_mask, question_lengths

        else:
            # Test mode processing
            q_input_tokens = torch.tensor(self.tokenizer.encode(self.questions[index]))
            q_input_tokens, q_attention_mask, _ = self.make_padding_test_finetune_mode(
                self.max_seqs_len[0], q_input_tokens
            )

            question_lengths = len(question_token) + len(q_input_tokens) + len(context_token)
            visual_visual_prefix = torch.ones(self.visual_prefix_len)
            visual_attention_mask = torch.ones(self.visual_prefix_len)

            input_tokens = torch.cat([
                question_token,
                q_input_tokens,
                context_token,
                visual_visual_prefix,
                answer_token
            ])

            attention_mask = torch.cat([
                question_attention_mask,
                q_attention_mask,
                context_attention_mask,
                visual_attention_mask,
                answer_attention_mask
            ])

            assert input_tokens.shape == attention_mask.shape
            return input_tokens, attention_mask, question_lengths

    def make_padding(self, max_len, input_tokens, question=False, leftover_input_tokens=0):
        current_len = input_tokens.size(0)
        padding_needed = max_len - current_len

        if padding_needed > 0:
            if question:
                # For questions, we keep the original input_tokens and track leftover space
                attention_mask = torch.ones(current_len)
                leftover_input_tokens = padding_needed
            else:
                # For answers, apply padding with zeros
                input_tokens = torch.cat((input_tokens, torch.zeros(padding_needed + leftover_input_tokens)))
                attention_mask = torch.cat((
                    torch.ones(current_len),
                    torch.zeros(padding_needed + leftover_input_tokens)
                ))
        elif padding_needed == 0:
            if question:
                attention_mask = torch.ones(current_len)
            else:
                input_tokens = torch.cat((input_tokens, torch.zeros(leftover_input_tokens)))
                attention_mask = torch.cat((torch.ones(current_len), torch.zeros(leftover_input_tokens)))
        else:  # padding_needed < 0
            if question:
                input_tokens = input_tokens[:max_len]
                attention_mask = torch.ones(max_len)
            else:
                input_tokens = torch.cat((input_tokens[:max_len], torch.zeros(leftover_input_tokens)))
                attention_mask = torch.cat((torch.ones(max_len), torch.zeros(leftover_input_tokens)))

        return input_tokens, attention_mask, leftover_input_tokens

    def make_padding_test_finetune_mode(self, max_len, input_tokens, do_padding=False):
        current_len = input_tokens.size(0)
        padding_needed = max_len - current_len

        if padding_needed > 0:
            if do_padding:
                input_tokens = torch.cat((input_tokens, torch.zeros(padding_needed)))
                attention_mask = torch.cat((torch.ones(current_len), torch.zeros(padding_needed)))
                padding_len = padding_needed
            else:
                attention_mask = torch.ones(current_len)
                padding_len = 0
        elif padding_needed == 0:
            attention_mask = torch.ones(current_len)
            padding_len = 0
        else:  # padding_needed < 0
            input_tokens = input_tokens[:max_len]
            attention_mask = torch.ones(max_len)
            padding_len = 0

        return input_tokens, attention_mask, padding_len

    def __getitem__(self, index):
        visual_prefix = self.visual_embeddings[self.image_indexs[index]]
        input_tokens, attention_mask, question_lengths = self.pad_sequences(index)
        return visual_prefix, self.target_labels[index], input_tokens, attention_mask, question_lengths

import torch
from torch.utils.entries import DataLoader
import os
from argparse import Namespace

def get_default_args():
    """Returns a Namespace with default arguments (identical to original parser)"""
    return Namespace(
        language_model="roberta-base",                # Choices: ["roberta-base", "microsoft/biogpt", ...]
        finetune_mode="lora",                    # Choices: ["lora", "frozen", ...]
        ablation="none",                     # Choices: ["none", "remove_question", ...]
        mapping_type="MLP",                  # Choices: ["MLP", "Transformer"]
        visual_prefix_length=1,                    # Match your .pkl files
        entriesset_path="../project",
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
        entriesset="slake"                      # "pathvqa", "ovqa", or "slake"
    )

def run_pipeline():
    # Process VQA-RAD entriesset
    for split in ['train','test','validate']:
        output_file_path = f"../entries/vqarad_pkl/{split}.pkl"
        prepare_vqarad_entries(split, output_file_path)

    # ==== Configure args directly here (modify as needed) ====
    args = get_default_args()

    # Example overrides (uncomment what you need):
    # args.language_model = "microsoft/biogpt"
    # args.finetune_mode = "lora"
    # args.ablation = "remove_visual"
    # args.eval = True
    # args.checkpoint = "./checkpoints/open_ended_latest.pt"
    # ========================================================

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Prepare entriessets
    train_entriesset = MedicalVQADataset(
        os.path.join(args.entriesset_path, args.entriesset + '/'),
        split="train",
        visual_prefix_length=args.visual_prefix_length,
        language_model=args.language_model
    )
    val_entriesset = MedicalVQADataset(
        os.path.join(args.entriesset_path, args.entriesset + '/'),
        split="validate",
        visual_prefix_length=args.visual_prefix_length,
        language_model=args.language_model
    )
    test_entriesset = MedicalVQADataset(
        os.path.join(args.entriesset_path, args.entriesset + '/'),
        split="test",
        visual_prefix_length=args.visual_prefix_length,
        language_model=args.language_model,
        like_test=True
    )

    # Initialize model
    if args.ablation != "none":
        model = VisualQAModel_abl(
            visual_prefix_length=args.visual_prefix_length,
            clip_length=4,
            finetune_mode=args.finetune_mode,
            mapping_type=args.mapping_type,
            args=args  # Pass Namespace directly
        )
    else:
        model = VisualQAModel(
            visual_prefix_length=args.visual_prefix_length,
            clip_length=4,
            finetune_mode=args.finetune_mode,
            mapping_type=args.mapping_type,
            args=args  # Pass Namespace directly
        )

    # Train or evaluate
    if not args.eval:
        pytorch_model_run(
            DataLoader(train_entriesset, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_entriesset, batch_size=args.batch_size),
            model,
            args
        )
    else:
        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        eval_gpt_open_ended(model, test_entriesset, args)

if __name__ == "__run_pipeline__":
    run_pipeline()

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
vision_encoder, clip_image_preprocessor = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# 2) Load your fine‑tuned VisualQAModel
model = VisualQAModel(
    visual_prefix_length=1,
    clip_length=4,
    finetune_mode="frozen",
    mapping_type="MLP",
    args=type("A", (), {"language_model":"roberta-base","batch_size":1})()
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

# 5) Compute CLIP visual_prefix (float16 → float32)
with torch.no_grad():
    clip_visual_prefix = vision_encoder.encode_image(
        clip_image_preprocessor(img).unsqueeze(0).to(device)
    ).squeeze(0).to(torch.float32)    # → shape (512,)

# 6) Tokenize prompt
prompt = f"question: {question} context:"
q_ids  = tokenizer.encode(prompt)
question_lengths  = len(q_ids)

# 7) Build text embeddings
text_emb = model.gpt.transformer.embed_input_tokens(
    torch.tensor(q_ids, device=device)
).unsqueeze(0)      # → (1, question_lengths, D)

# 8) Project visual visual_prefix into GPT space
with torch.no_grad():
    #  → (1, D * visual_prefix_length)
    visual_prefix_proj_flat = model.clip_project(clip_visual_prefix.unsqueeze(0))
    # reshape to (1, visual_prefix_length, D)
    visual_prefix_proj = visual_prefix_proj_flat.view(
        1,
        model.visual_prefix_length,
        model.gpt_embedding_size
    )
# 9) Concatenate [text prompt] + [visual visual_prefix]
inputs_embeds = torch.cat([text_emb, visual_prefix_proj], dim=1)
# shape: (1, question_lengths + visual_prefix_length, D)

# 10) Greedy generation from GPT
out_ids = model.gpt.generate(
    inputs_embeds=inputs_embeds,
    max_new_input_tokens=30,
    eos_token_id=tokenizer.eos_token_id,
)

# 11) Decode only the newly generated input_tokens
generated_ids = out_ids[0, inputs_embeds.size(1):]
prediction = tokenizer.decode(generated_ids, skip_special_input_tokens=True).strip()

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
PREFIX_LENGTH = 1                                              # same as you used in image_preprocessoring
MODEL_TYPE    = "roberta-base"                                      # or "microsoft/biogpt", etc.

test_entriesset = MedicalVQADataset(
    path=PICKLE_DIR,
    split="test",
    visual_prefix_length=PREFIX_LENGTH,
    language_model=MODEL_TYPE
)

class Args: pass
args = Args()
args.language_model   = MODEL_TYPE
args.finetune_mode      = "lora"
args.mapping_type = "MLP"
args.visual_prefix_length= PREFIX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisualQAModel(PREFIX_LENGTH, clip_length=4, finetune_mode=args.finetune_mode,
                    mapping_type=args.mapping_type, args=args)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device).eval()

# ── 3) Pick one example ──────────────────────────────
idx = 0
visual_prefix, label, input_tokens, attention_mask, question_lengths = test_entriesset[idx]
# group_id = test_entriesset.image_indexs[idx]   # this is the xmllab folder index
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
visual_prefix = visual_prefix.to(device).float().unsqueeze(0)
input_tokens = input_tokens.to(device).long().unsqueeze(0)
attention_mask   = attention_mask.to(device).long().unsqueeze(0)
question_lengths_t= torch.tensor([question_lengths], device=device)

with torch.no_grad():
    emb   = model.generate(visual_prefix, label, input_tokens.squeeze(0), attention_mask.squeeze(0), question_lengths)
    emb   = emb.unsqueeze(0)
    beams = generate_beam(
        model=model,
        tokenizer=model.tokenizer,
        beam_size=5,
        generated=emb,
        entry_length=test_entriesset.max_seqs_len[1],
        temperature=1.0,
        stop_token="<|endoftext|>"
    )

# ── 5) Display ───────────────────────────────────────
img      = Image.open(img_file).convert("RGB")
question = test_entriesset.questions[idx]
gt       = test_entriesset.answers[idx]
pred     = beams[0]

plt.figure(figsize=(6,6))
plt.imshow(img); plt.axis("off")
plt.title(f"Q: {question}\nGT: {gt}\nPred: {pred}", wrap=True)
plt.show()

print("All beam candidates:", beams)

!pip install pylint

!pyreverse -o jpg -p MedicalVQA -a 1 -fmodell.py

!apt-get update -qq && apt-get install -y graphviz




def prepare_vqarad_entries(split, output_file_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vision_encoder, image_preprocessor = clip.load("ViT-B/32", device=device, jit=False)
    with open(f"../entries/VQA-RAD/{split}.json") as f:
        entries = json.load(f)
    print(f"{len(entries)} entries loaded from VQA-RAD {split}.json")

    all_visual_embeddings, image_indexs, image_locations = [], [], []
    question_list, answer_list = []
    image_entry_map = {}

    for i, d in enumerate(entries):
        if isEglish(d['answer']) and isEglish(d['question']):
            image_index = d["image_index"]
            filename = f"../entries/VQA-RAD/images/{d['image_file']}"
            with torch.no_grad():
                visual_features = vision_encoder.encode_image(image_preprocessor(Image.open(filename)).unsqueeze(0).to(device)).cpu()
            if image_index not in image_entry_map:
                image_entry_map[image_index] = [[d['question']], [d['answer']], visual_features, filename]
            else:
                image_entry_map[image_index][0].append(d['question'])
                image_entry_map[image_index][1].append(d['answer'])

    for image_index, imgs in enumerate(image_entry_map.keys()):
        all_visual_embeddings.append(image_entry_map[imgs][2])
        for q in range(len(image_entry_map[imgs][0])):
            question_list.append(image_entry_map[imgs][0][q])
            answer_list.append(image_entry_map[imgs][1][q])
            image_indexs.append(image_index)
            image_locations.append(image_entry_map[imgs][2])

    all_entries = {
        "img_visual_prefix": torch.cat(all_visual_embeddings, dim=0),
        "image_indexs": image_indexs,
        "questions": question_list,
        "answers": answer_list,
        "img_path": image_locations
    }

    with open(output_file_path, 'wb') as f:
        pickle.dump(all_entries, f)
    print('VQA-RAD image_preprocessoring done.')
    print(f"{len(question_list)} embeddings saved.")
