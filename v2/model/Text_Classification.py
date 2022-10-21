# Import all libraries
import numpy as np

# Huggingface transformers
from transformers import AutoModel, AutoTokenizer, AdamW

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
import os
import pytorch_lightning as pl
import json

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_dict = np.array(['brand_acitivity', 'brand_community', 'brand_emotional',
                    'brand_resonance', 'event_brand', 'event_current',
                    'franchise_recruitment', 'misc', 'partner_info',
                    'product_educational', 'product_info', 'promotion_info',
                    'recruitment', 'store_info'])

class SequenceClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self,
                 model_name, 
                 n_classes=10, 
                 steps_per_epoch=None, 
                 n_epochs=30, 
                 lr=5e-5):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, input_ids, attn_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('val_loss', loss , prog_bar=True, logger=True)
        
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('test_loss', loss , prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        
        return [optimizer]

def decode_label(preds):
    preds_label = []
    for pred in preds:
        preds_label.append(label_dict[pred == 1])
    return preds_label

def thresholding(pred_prob, thresh=0.43):
    y_pred = []
    for tag_label_row in pred_prob:
        temp = []
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1)
            else:
                temp.append(0)
        y_pred.append(temp)
    return y_pred

def TS4CS(input_path):
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    model_path = "checkpoints/ts4cs/Sequence-epoch=29-val_loss=0.11.ckpt"

    model = SequenceClassifier.load_from_checkpoint(checkpoint_path=model_path)
    model = model.to(device)
    model.eval()

    #load data
    PATH_TO_INPUT = input_path
    with open(PATH_TO_INPUT, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    # convert to array
    texts = []
    fids = []
    for d in data:
        texts.append(d['text'])
        fids.append(d['fid'])

    # Tokenize all questions in x_test
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_text = tokenizer(
            text,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Add the input_ids from encoded question to the list.    
        input_ids.append(encoded_text['input_ids'])
        # Add its attention mask 
        attention_masks.append(encoded_text['attention_mask'])
    
    # Now convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    INFERENCE_BATCH_SIZE = 32  

    # Create the DataLoader.
    dataset = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(dataset) # remain item order
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=INFERENCE_BATCH_SIZE)

    preds = []
    # Predict 
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
    
        # Unpack the inputs from our dataloader
        b_input_ids, b_attn_mask = batch
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            pred = model(b_input_ids, b_attn_mask)
            pred = torch.sigmoid(pred)
            # Move predicted output to CPU
            pred = pred.detach().cpu().numpy()

        preds.append(pred)

    flat_preds = np.concatenate(preds, axis=0)

    THRESDHOLD = 0.43

    pred_labels = thresholding(flat_preds, thresh=THRESDHOLD) 
    pred_labels = np.array(pred_labels)

    labels = decode_label(pred_labels)

    result = []
    for fid, text, label in zip(fids, texts, labels):
        result.append({
            "fid": fid,
            "text": text,
            "label": list(label) # convert to list for JSON Serialization
        })
    return result
    # basename = os.path.basename(PATH_TO_INPUT).split(".")[0]
    # PATH_TO_OUTPUT = os.path.join("data/result", "%s.json" % basename)

    # with open(PATH_TO_OUTPUT, 'w', encoding="utf-8") as fou:
    #     json.dump(result, fou, ensure_ascii=False)
