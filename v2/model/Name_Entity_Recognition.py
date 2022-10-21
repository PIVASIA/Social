# -*- coding: utf-8 -*-

from cProfile import label
from transformers import AutoModelForTokenClassification, AutoTokenizer, TokenClassificationPipeline, AutoConfig
import json
import os
import re
import torch
# import time

def _merge_complete_word(word):
    tokens = word.split(" ")
    words = []
    last_word = ""
    for token in tokens:
        last_word = re.sub('@@', '', token)
        words.append(last_word)
        last_word = ""
    return "".join(words)

def merge_word_and_label(subwords, labels):
    entities = []
    label_groups = []
    for subword_ids, label_ids in zip(subwords, labels):
        entity_ids = []
        label_group_ids = []
        completeWord = ""
        completeLabel = ""
        for word, label in zip(subword_ids, label_ids):
            if label.startswith("B"):
                if completeWord and completeLabel:
                    entity_ids.append(completeWord)
                    label_group_ids.append(completeLabel.split("-")[1])
                completeWord = word
                completeLabel = label
            elif label.startswith("I"):
                completeWord = completeWord + " " + word
        if completeWord and completeLabel:
            entity_ids.append(completeWord)
            label_group_ids.append(completeLabel.split("-")[1])
        for i, word in enumerate(entity_ids):
            word = _merge_complete_word(word)
            entity_ids[i] = word
        entities.append(entity_ids)
        label_groups.append(label_group_ids)
    return entities, label_groups

def NER4CS(data_inp):
    label_list = ['O','B-san_pham', 'I-san_pham', 'B-muc_giam_gia', 'I-muc_giam_gia',
              'B-date', 'I-date', 'B-gia_san_pham', 'I-gia_san_pham','B-vi_tri', 'I-vi_tri', 
              'B-so_nha', 'I-so_nha', 'B-ten_duong', 'I-ten_duong', 'B-ten_thon_lang_todanpho', 'I-ten_thon_lang_todanpho', 
              'B-ten_phuong_xa_thitran', 'I-ten_phuong_xa_thitran', 'B-ten_quan_huyen_thanhpho', 'I-ten_quan_huyen_thanhpho',
              'B-ten_tinh_thanh_pho', 'I-ten_tinh_thanh_pho']

    phobert = AutoModelForTokenClassification.from_pretrained('checkpoints/ner/')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # PATH_TO_INPUT = input_path
    # with open(PATH_TO_INPUT, 'r', encoding='utf-8') as fin:
    #     data = json.load(fin)
    data = data_inp
    text = []
    fids = []

    last_fid = ""
    current_text = ""
    for d in range(len(data)):
        if data[d]['fid'] == last_fid and (len(current_text) + len(data[d]['text'])) < 256:
            current_text += " " + data[d]['text']
        else:
            if current_text:
                text.append(current_text)
                fids.append(last_fid)
            current_text = data[d]['text']
            last_fid = data[d]['fid']

    for line in text:
        list_ids = tokenizer(line)['input_ids']
        # if sentence is longer than 256 tokens, ignore token 257 onward
        if len(list_ids) >= 256:
            list_ids = list_ids[0:255]
        
        # lấy id của các tokens tương ứng 
        input_ids = torch.tensor([list_ids])
        # input_ids = input_ids.resize_(1,256)
        # không dùng tokenize(decode(encode)), text sẽ bị lỗi khi tokenize do conflict với tokenizer mặc định
        # lấy các token để đánh tags 
        tokens = tokenizer.convert_ids_to_tokens(list_ids)
        # print(input_ids.shape)
        outputs = phobert(input_ids).logits
        
        predictions = torch.argmax(outputs, dim=2)

        tmp = []
        prd = []
        labels = []
        lb = ''
        for i in [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())]:
            # print(i)   
            if i[1].startswith("B"):
                prd.append(' '.join(tmp))
                labels.append(lb)
                tmp = [i[0]]
                lb = i[1].split('-')[1]
                # print(lb)
            elif i[1].startswith("I"):
                tmp.append(i[0])
            # elif "@@" in i[0]:
            #     tmp.append(i[0])
        for i in range(len(prd)):
            if "@@" in prd[i]:
                prd[i] = prd[i].replace("@@ ","")
                prd[i] = prd[i].replace("@@","")
        print(prd[1:])
        print(labels[1:])
        yield prd[1:], labels[1:]
    # labels = []
    # subwords = []
    # for pred in preds:
    #     label_ids = []
    #     subwords_ids = []
    #     for token in pred:
    #         class_id = config.label2id[token["entity_group"]]
    #         label_ids.append(label_list[class_id])
    #         subwords_ids.append(token["word"])
    #     labels.append(label_ids)
    #     subwords.append(subwords_ids)

    # for i in range(len(labels[0])):
    #     if labels[0][-i].startswith('B') and labels[0][-i-1].startswith('B'):
    #         labels[0][-i] = labels[0][-i].replace('B', 'I')
    # entities, label_groups = merge_word_and_label(subwords, labels)

    # result = {}
    # for fid, entity, label in zip(fids, entities, label_groups):
    #     if fid not in result:
    #         result[fid] = {}
        
    #     for e, l in zip(entity, label):
    #         if l not in result[fid]:
    #             result[fid][l] = []
            
    #         result[fid][l].append(e)

    # ouput = []
    # for k, v in result.items():
    #     item = {"fid": k}
    #     for u, v in v.items():
    #         item[u] = v
    #     ouput.append(item)

    # basename = os.path.basename(PATH_TO_INPUT).split(".")[0]
    # PATH_TO_OUTPUT = os.path.join("data/result", "%s_ner.json" % basename)

    # with open(PATH_TO_OUTPUT, 'w+', encoding="utf-8") as fou:
    #     json.dump(ouput, fou, ensure_ascii=False)

# NER4CS('data/inference/1625740876000_posts_post_cls.json')