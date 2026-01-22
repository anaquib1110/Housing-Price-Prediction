# model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, RobertaModel

class HybridBERTRoBERTaDataset(Dataset):
    def __init__(self, texts, keywords, locations, targets=None, bert_tokenizer=None, roberta_tokenizer=None, max_len=128):
        self.texts = texts
        self.keywords = keywords
        self.locations = locations
        self.targets = targets
        self.bert_tokenizer = bert_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        keyword = self.keywords.iloc[idx]
        location = self.locations.iloc[idx]
        
        combined_text = f"{text} keyword: {keyword} location: {location}"
        
        bert_encoding = self.bert_tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        roberta_encoding = self.roberta_tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'bert_input_ids': bert_encoding['input_ids'].flatten(),
            'bert_attention_mask': bert_encoding['attention_mask'].flatten(),
            'bert_token_type_ids': bert_encoding['token_type_ids'].flatten(),
            'roberta_input_ids': roberta_encoding['input_ids'].flatten(),
            'roberta_attention_mask': roberta_encoding['attention_mask'].flatten(),
        }
        
        if self.targets is not None:
            item['targets'] = torch.tensor(self.targets.iloc[idx], dtype=torch.long)
            
        return item

class HybridBERTRoBERTaModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', roberta_model_name='roberta-base', num_classes=2):
        super(HybridBERTRoBERTaModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dropout = nn.Dropout(0.3)
        
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_dropout = nn.Dropout(0.3)
        
        bert_hidden_size = self.bert.config.hidden_size
        roberta_hidden_size = self.roberta.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size + roberta_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, bert_input_ids, bert_attention_mask, bert_token_type_ids, 
                roberta_input_ids, roberta_attention_mask):
        
        bert_output = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids
        )
        bert_pooled_output = self.bert_dropout(bert_output.pooler_output)
        
        roberta_output = self.roberta(
            input_ids=roberta_input_ids,
            attention_mask=roberta_attention_mask
        )
        roberta_pooled_output = self.roberta_dropout(roberta_output.pooler_output)
        
        concatenated_output = torch.cat((bert_pooled_output, roberta_pooled_output), dim=1)
        logits = self.classifier(concatenated_output)
        
        return logits
