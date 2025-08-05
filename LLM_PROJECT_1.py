#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installing required packages
get_ipython().system('pip install --upgrade transformers')
get_ipython().system('pip install --upgrade accelerate')
get_ipython().system('pip install kaggle')
get_ipython().system('pip install transformers datasets')
get_ipython().system('pip install transformers torch')


# In[2]:


# Importing necessary libraries
import os
import torch
import random
import re
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[3]:


# Downloading the dataset from kaggle
get_ipython().system('kaggle datasets download -d kazanova/sentiment140')


# In[4]:


# Unzipping the data files
get_ipython().system('unzip sentiment140.zip -d sentiment140')


# In[5]:


# Loading the dataset
Data = pd.read_csv('sentiment140/training.1600000.processed.noemoticon.csv',
                 encoding='latin-1',  header = None,
                 names=['target', 'ids', 'date', 'flag', 'user', 'text'])


# In[6]:


# Printing Data information
print(Data.info())


# In[7]:


# Dropping unrequired data columns
Data = Data.drop(['ids', 'date', 'flag', 'user'], axis=1)


# In[8]:


# Displaying the first few rows
print(Data.head())


# In[9]:


# checking null values
print(Data.isnull().sum())


# In[10]:


# Data summary
print(Data.describe())


# In[11]:


# Visualising the data
sns.countplot(x='target', data=Data)
plt.show()


# In[12]:


# Setting the placeholder names
Hashtags = re.compile(r"^#\S+|\s#\S+")
Mentions = re.compile(r"^@\S+|\s@\S+")
URLs = re.compile(r"https?://\S+")


# In[13]:


# Defining a function for processing the text
def TextProcessing(text):
    text = re.sub(r'http\S+', '', text)
    text = Hashtags.sub(' hashtag', text)
    text = Mentions.sub(' entity', text)
    return text.strip().lower()


# In[14]:


# Processing the text using applied function
Data['ProcessedText'] = Data.text.apply(TextProcessing)


# In[15]:


# Checking the data after processing the text
Data.head()


# In[16]:


# Ensuring labels are correct for binary classification (0 and 4 -> 0 and 1)
Data['target'] = Data['target'].apply(lambda x: 0 if x == 0 else 1)


# In[17]:


# Setting the labels and text
Labels = Data.target.values
Text = Data.ProcessedText.values


# In[18]:


# Setting the tokeniser
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)


# In[19]:


# Tokenizing the text
input_ids = []
attention_mask = []

for i in Text:
    encoded_data = tokenizer.encode_plus(
        i,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_data['input_ids'])
    attention_mask.append(encoded_data['attention_mask'])

InputID = torch.cat(input_ids, dim=0)
AttentionMask = torch.cat(attention_mask, dim=0)
Labels = torch.tensor(Labels)


# In[20]:


# Setting the parameters for the model
Dataset = TensorDataset(InputID,AttentionMask,Labels)
TrainingDataSize = int(0.8*len(Dataset))
ValidationDataSize = len(Dataset) - TrainingDataSize


# In[21]:


# Setting the training and testing data size
train_dataset,val_dataset = random_split(Dataset,[TrainingDataSize,ValidationDataSize])


# In[22]:


# Printing the training and testing data size
print('Training Size - ',TrainingDataSize)
print('Validation Size - ',ValidationDataSize)


# In[23]:


# DataLoader for training and validation
TrainingData = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
ValidationData = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)


# In[24]:


# Setting the Model appication
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2, output_attentions = False, output_hidden_states = False)


# In[25]:


# Setting the seeding parameters for the model
SeedingValidity = 17
random.seed(SeedingValidity)
np.random.seed(SeedingValidity)
torch.manual_seed(SeedingValidity)
torch.cuda.manual_seed_all(SeedingValidity)


# In[26]:


# Setting the device for model computation
ModelDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(ModelDevice)


# In[27]:


# Setting the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)


# In[28]:


# Setting up the scheduler
Epoches = 1
TotalTrainingSteps = len(TrainingData) * Epoches
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=TotalTrainingSteps)


# In[29]:


# Defining a function for accuracy calculation
def PrintAccuracy(preds, labels):
    PredictionFlats = np.argmax(preds, axis=1).flatten()
    LabellingFlats = labels.flatten()
    return np.sum(PredictionFlats == LabellingFlats) / len(LabellingFlats)


# In[30]:


# Defining a function for model evaluation
def ModelEvaluation(dataloader_test):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_test:
        batch = tuple(b.to(ModelDevice) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_test)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


# In[31]:


# Training the model
torch.cuda.empty_cache()
for epoch in tqdm(range(1, Epoches + 1)):
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(TrainingData, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(ModelDevice) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total / len(TrainingData)
    tqdm.write(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = ModelEvaluation(ValidationData)
    val_acc = PrintAccuracy(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Validation accuracy: {val_acc}')

