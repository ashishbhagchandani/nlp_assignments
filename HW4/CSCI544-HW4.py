#!/usr/bin/env python
# coding: utf-8

# ### Import all the required libraries 

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence


# ### Read train data, where each line contains three space-separated values index, word, tag. Where sentence are append to sentences list and tag are append to tags list  

# In[ ]:


sentences, tags = [], []
sentence, tag = [], []
with open('data/train') as f:
    for line in f:
        ls = (line.rstrip('\n')).split(" ")
        if len(ls) == 3:
            sentence.append(ls[1])
            tag.append(ls[2])
        else:
            sentences.append(sentence)
            tags.append(tag)
            sentence, tag = [], []


# ### Read dev data, where each line contains three space-separated values index, word, tag. Where sentence are append to sentencesdev list and tagdev are append to tags list

# In[ ]:


sentencesdev, tagsdev = [], []
sentencedev, tagdev = [], []
with open('data/dev') as f:
    for line in f:
        ls = (line.rstrip('\n')).split(" ")
        if len(ls) == 3:
            sentencedev.append(ls[1])
            tagdev.append(ls[2])
        else:
            sentencesdev.append(sentencedev)
            tagsdev.append(tagdev)
            sentencedev, tagdev = [], []

#This functions helps to build word2idx and tag2idx dictionary. Here <pad> is pre append to both word2idx and tag2idx and <unk> is pre append to word2idx
# In[ ]:


def tokenize_and_build_vocab(sentences, tags):
    # initialize word and tag lists
    words = []
    tag_list = []

    # tokenize sentences and tags
    for sentence in sentences:
        for word in sentence:
            words.append(word)

    for tag_seq in tags:
        for tag in tag_seq:
            tag_list.append(tag)

    # build vocabulary
    word2idxt = {'<PAD>': 0, '<unk>': 1}
    tag2idxt = {}
    for word in words:
        if word not in word2idxt:
            word2idxt[word] = len(word2idxt)

    for tag in tag_list:
        if tag not in tag2idxt:
            tag2idxt[tag] = len(tag2idxt)
    tag2idxt['<PAD>']= -1
    return word2idxt, tag2idxt


# ### word2idx and tag2idx for dev data

# In[ ]:


word2idx_dev, tag2idx_dev = tokenize_and_build_vocab(sentencesdev, tagsdev)


# ### word2idx and tag2idx for train data

# In[ ]:


word2idx, tag2idx = tokenize_and_build_vocab(sentences, tags)


# # Task 1: Simple Bidirectional LSTM model

# ### PyTorch implementation of a Bidirectional LSTM (BLSTM) model
# ### The model consists of an embedding layer to map each word index to a fixed-size embedding vector, a BLSTM layer to capture contextual information from the input sequence, a fully connected layer with an ELU activation function and dropout regularization, and a linear layer to produce the final output logits.

# In[ ]:


class BLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 128)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, output_dim)
        
    def forward(self, tl):
        text, lens = tl
        embedded = self.dropout(self.embedding(text))
        

        packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        last_hidden_states, lens_unpacked = pad_packed_sequence(outputs, batch_first=True)
        x = self.dropout(last_hidden_states)
        x = self.fc(x)
        x = self.elu(x)
        logits = self.classifier(x)
        return logits


# ### It takes in a list of sentences and their corresponding tags, and returns them as a pair of tensors where each sentence and tag is represented as a sequence of integers based on the provided word2idx and tag2idx dictionaries.

# In[ ]:


class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        
        sentence_idx = [self.word2idx[word] if word in word2idx else self.word2idx['<unk>'] for word in sentence]
        tags_idx = [self.tag2idx[tag] for tag in tags]
        return torch.LongTensor(sentence_idx), torch.LongTensor(tags_idx)


# ### The collate_fn function takes in a batch of sentences and tags, pads them to the maximum length, and returns them along with the corresponding lengths. The train_dataset instance is created using the NERDataset class and the train_loader instance is created using the DataLoader class to load the data in batches for training.

# In[ ]:


def collate_fn(batch):
    sentences, tags = zip(*batch)
    lens = [len(s) for s in sentences]

    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=word2idx['<PAD>'])
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=tag2idx['<PAD>'])
    return padded_sentences, padded_tags, lens

train_dataset = NERDataset(sentences, tags, word2idx, tag2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# In[ ]:


def collate_fn_dev(batch):
    sentences_dev, tags_dev = zip(*batch)
    lens_dev = [len(s_dev) for s_dev in sentences_dev]

    padded_sentences_dev = pad_sequence(sentences_dev, batch_first=True, padding_value=word2idx_dev['<PAD>'])
    padded_tags_dev = pad_sequence(tags_dev, batch_first=True, padding_value=tag2idx_dev['<PAD>'])
    return padded_sentences_dev, padded_tags_dev, lens_dev

train_datasetdev = NERDataset(sentencesdev, tagsdev, word2idx, tag2idx)
train_loaderdev = DataLoader(train_datasetdev, batch_size=32, shuffle=False, collate_fn=collate_fn_dev)


# ### Creates an instance of the BLSTMModel

# In[ ]:


# create BLSTM model
model = BLSTMModel(vocab_size=len(word2idx), embedding_dim=100, hidden_dim=256, output_dim=len(tag2idx)-1, dropout=0.33)

# move model to device
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# ### Creates an instance of the Stochastic Gradient Descent optimizer for the BLSTMModel with specified learning rate and momentum. Further instance of the Cross Entropy Loss function to calculate the loss during training and a learning rate scheduler that will adjust the learning rate based on the model's F1 score on dev data.

# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['<PAD>']).to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15, verbose=True)


# #### This code trains the model for 120 epochs and save the model when max_f1 > f1 and print f1 score of training and dev data at every epoch

# In[ ]:


num_epochs = 120
max_f1 = 0
for epoch in range(num_epochs):
    model.train()
    true_labels_train = []
    pred_labels_train = []
    for i, (sentences, tags, lens) in enumerate(train_loader):
        optimizer.zero_grad()
        sentences, tags = sentences.to(device), tags.to(device)
        outputs = model((sentences, lens))
        predicted = torch.argmax(outputs, dim=2)
        temptags = tags.view(-1).cpu().numpy()
        temppred = predicted.view(-1).cpu().numpy()
        for ii in range(len(temptags)):
            if temptags[ii] == -1:
                continue
            else:
                true_labels_train.append(temptags[ii])
                pred_labels_train.append(temppred[ii])
    
        loss = criterion(outputs.view(-1, len(tag2idx)-1), tags.view(-1))
        loss.backward()
        optimizer.step()
        
    f1_train = f1_score(true_labels_train, pred_labels_train, average='macro', zero_division=1)
    
    model.eval()
    with torch.no_grad():
        true_labels_dev = []
        pred_labels_dev = []
        for sentences_dev, tags_dev, lens_dev in train_loaderdev:
            sentences_dev, tags_dev = sentences_dev.to(device), tags_dev.to(device)
            outputs_dev = model((sentences_dev, lens_dev))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temptagsdev = tags_dev.view(-1).cpu().numpy()
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for ii in range(len(temptagsdev)):
                if temptagsdev[ii] == -1:
                    continue
                else:
                    true_labels_dev.append(temptagsdev[ii])
                    pred_labels_dev.append(temppreddev[ii])

        f1 = f1_score(true_labels_dev, pred_labels_dev, average='macro', zero_division=1)
        scheduler.step(f1)
        if max_f1 < f1:
            max_f1 = f1
            print('model saved')
            torch.save(model.state_dict(), 'blstm1.pt')
    print(f'Epoch [{epoch+1}/{num_epochs}], f1-train:{f1_train}, Loss: {loss.item():.4f}, F1 Score: {f1:.4f}')


# ### Load the model with best f1 score on dev data

# In[ ]:


model.load_state_dict(torch.load('blstm1.pt'))


# ### make idx as key and respective word as value 
# ### make idx as key and respective tag as value

# In[ ]:


idx2word = {idx: word for word, idx in word2idx.items()}
idx2tag = {idx: word for word, idx in tag2idx.items()}


# ### Apply best model on dev data and generate dev1.out file with index, word, tag, pred_tag

# In[ ]:


model.eval()
with open('dev1.out', 'w') as f:
    with torch.no_grad():
        sentences_dev, tags_dev = [], []
        for i in range(len(sentencesdev)):
            getsentence = sentencesdev[i]
            sentences_dev_idx = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in getsentence]
            gettag = tagsdev[i]
            sentences_dev_idx = torch.LongTensor(sentences_dev_idx).unsqueeze(0).to(device)
            gettlen = len(getsentence)
            outputs_dev = model((sentences_dev_idx, [gettlen]))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for i in range(len(getsentence)):
                word = getsentence[i]
                tag = gettag[i]
                pred = idx2tag[temppreddev[i]]
                f.write(f"{i+1} {word} {pred}\n")
            f.write("\n")


model.eval()
with open('deveval1.out', 'w') as f:
    with torch.no_grad():
        sentences_dev, tags_dev = [], []
        for i in range(len(sentencesdev)):
            getsentence = sentencesdev[i]
            sentences_dev_idx = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in getsentence]
            gettag = tagsdev[i]
            sentences_dev_idx = torch.LongTensor(sentences_dev_idx).unsqueeze(0).to(device)
            gettlen = len(getsentence)
            outputs_dev = model((sentences_dev_idx, [gettlen]))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for i in range(len(getsentence)):
                word = getsentence[i]
                tag = gettag[i]
                pred = idx2tag[temppreddev[i]]
                f.write(f"{i+1} {word} {tag} {pred}\n")
            f.write("\n")


# ### result from running the script conll03eval.txt < dev1.out
# processed 51577 tokens with 5942 phrases; found: 5160 phrases; correct: 4437.
# accuracy:  95.73%; precision:  85.99%; recall:  74.67%; FB1:  79.93
#               LOC: precision:  90.39%; recall:  82.96%; FB1:  86.52  1686
#              MISC: precision:  88.34%; recall:  75.60%; FB1:  81.47  789
#               ORG: precision:  78.94%; recall:  67.93%; FB1:  73.03  1154
#               PER: precision:  85.24%; recall:  70.85%; FB1:  77.38  1531
# ### Read test data, where each line contains three space-separated values index, word, tag. Where sentence are append to sentences list and tag are append to tags list

# In[ ]:


testsentences = []
testsentence = []
with open('data/test') as f:
    for line in f:
        ls = (line.rstrip('\n')).split(" ")
        if len(ls) == 2:
            testsentence.append(ls[1])
        else:
            testsentences.append(testsentence)
            testsentence = []


# ### Apply best model on test data and generate test1.out file with index, word, pred_tag

# In[ ]:


model.eval()
with open('test1.out', 'w') as f:
    with torch.no_grad():
        sentences_test = [] 
        for i in range(len(testsentences)):
            getsentence = testsentences[i]
            sentences_test_idx = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in getsentence]
            sentences_test_idx = torch.LongTensor(sentences_test_idx).unsqueeze(0).to(device)
            gettlen = len(getsentence)
            outputs_dev = model((sentences_test_idx, [gettlen]))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for i in range(len(getsentence)):
                word = getsentence[i]
                pred = idx2tag[temppreddev[i]]
                f.write(f"{i+1} {word} {pred}\n")
            f.write("\n")


# # Task 2: Using GloVe word embeddings

# ### Reads GloVe embedding file and creates dictionary where each word in the file is associated with its corresponding pre-trained embedding vector.

# In[ ]:

sentences, tags = [], []
sentence, tag = [], []
with open('data/train') as f:
    for line in f:
        ls = (line.rstrip('\n')).split(" ")
        if len(ls) == 3:
            sentence.append(ls[1])
            tag.append(ls[2])
        else:
            sentences.append(sentence)
            tags.append(tag)
            sentence, tag = [], []

sentencesdev, tagsdev = [], []
sentencedev, tagdev = [], []
with open('data/dev') as f:
    for line in f:
        ls = (line.rstrip('\n')).split(" ")
        if len(ls) == 3:
            sentencedev.append(ls[1])
            tagdev.append(ls[2])
        else:
            sentencesdev.append(sentencedev)
            tagsdev.append(tagdev)
            sentencedev, tagdev = [], []

def tokenize_and_build_vocab(sentences, tags):
    # initialize word and tag lists
    words = []
    tag_list = []

    # tokenize sentences and tags
    for sentence in sentences:
        for word in sentence:
            words.append(word)

    for tag_seq in tags:
        for tag in tag_seq:
            tag_list.append(tag)

    # build vocabulary
    word2idxt = {'<PAD>': 0, '<unk>': 1}
    tag2idxt = {}
    for word in words:
        if word not in word2idxt:
            word2idxt[word] = len(word2idxt)

    for tag in tag_list:
        if tag not in tag2idxt:
            tag2idxt[tag] = len(tag2idxt)
    tag2idxt['<PAD>']= -1
    return word2idxt, tag2idxt

word2idx_dev, tag2idx_dev = tokenize_and_build_vocab(sentencesdev, tagsdev)
word2idx, tag2idx = tokenize_and_build_vocab(sentences, tags)


glove_embeddings = {}
with open('glove.6B.100d/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.array([float(val) for val in values[1:]])
        glove_embeddings[word] = embedding


# ### This code initializes an embedding weight matrix of size len(word2idx) and 100. Then check each word in the glove embeddings if the word is there lower or original it gets its embedding else if word is not present lower or original then zeros are generated of dimension (len(word2idx, 100))

# In[ ]:


embedding_dim = 100
embedding_weights = np.zeros((len(word2idx), embedding_dim), dtype=np.float64)
for word, idx in word2idx.items():
    if word.lower() in glove_embeddings:
        word_embedding =  glove_embeddings.get(word.lower())
    elif word in glove_embeddings:
        word_embedding =  glove_embeddings.get(word)
    else:
        word_embedding = np.random.uniform(0, 0, embedding_dim)
    embedding_weights[idx] = word_embedding


# ### Creates an instance of the nn.Embedding layer, and then assigns the embedding_weights to the layer's weight parameter.

# In[ ]:


embedding_layer = nn.Embedding(len(word2idx), embedding_dim)
embedding_layer.weight = nn.Parameter(torch.FloatTensor(embedding_weights))


# ### PyTorch implementation of a Bidirectional LSTM (BLSTM) model
# ### The model consists of an glove embedding layer, a BLSTM layer to capture contextual information from the input sequence, a fully connected layer with an ELU activation function and dropout regularization, and a linear layer to produce the final output logits.

# In[ ]:


class BLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tl):
        text, lens = tl
        embedded = self.dropout(self.embedding(text))
        
        packed = pack_padded_sequence(embedded, lens, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        last_hidden_states, lens_unpacked = pad_packed_sequence(outputs, batch_first=True)
        x = self.dropout(last_hidden_states)
        x = self.fc1(last_hidden_states)
        x = self.elu(x)
        logits = self.fc2(x)
        return logits


# ### It takes in a list of sentences and their corresponding tags, and returns them as a pair of tensors where each sentence and tag is represented as a sequence of integers based on the provided word2idx and tag2idx dictionaries.

# In[ ]:


class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        
        sentence_idx = [self.word2idx[word] if word in word2idx else self.word2idx['<unk>'] for word in sentence]
        tags_idx = [self.tag2idx[tag] for tag in tags]
        return torch.LongTensor(sentence_idx), torch.LongTensor(tags_idx)


# ### The collate_fn function takes in a batch of sentences and tags, pads them to the maximum length, and returns them along with the corresponding lengths. The train_dataset instance is created using the NERDataset class and the train_loader instance is created using the DataLoader class to load the data in batches for training.

# In[ ]:


def collate_fn(batch):
    sentences, tags = zip(*batch)
    lens = [len(s) for s in sentences]

    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=word2idx['<PAD>'])
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=tag2idx['<PAD>'])
    return padded_sentences, padded_tags, lens

train_dataset = NERDataset(sentences, tags, word2idx, tag2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# In[ ]:


def collate_fn_dev(batch):
    sentences_dev, tags_dev = zip(*batch)
    lens_dev = [len(s_dev) for s_dev in sentences_dev]

    padded_sentences_dev = pad_sequence(sentences_dev, batch_first=True, padding_value=word2idx_dev['<PAD>'])
    padded_tags_dev = pad_sequence(tags_dev, batch_first=True, padding_value=tag2idx_dev['<PAD>'])
    return padded_sentences_dev, padded_tags_dev, lens_dev

train_datasetdev = NERDataset(sentencesdev, tagsdev, word2idx, tag2idx)
train_loaderdev = DataLoader(train_datasetdev, batch_size=32, shuffle=False, collate_fn=collate_fn_dev)


# ### Creates an instance of the BLSTMModel

# In[ ]:


# create BLSTM model
model = BLSTMModel(vocab_size=len(word2idx), embedding_dim=100, hidden_dim=256, output_dim=len(tag2idx), dropout=0.33, embedding_layer=embedding_layer)

# move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# ### Creates an instance of the Stochastic Gradient Descent optimizer for the BLSTMModel with specified learning rate and momentum. Further instance of the Cross Entropy Loss function to calculate the loss during training and a learning rate scheduler that will adjust the learning rate based on the model's F1 score on dev data.

# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['<PAD>']).to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=15, verbose=True)


# ### This code trains the model for 120 epochs and save the model when max_f1 > f1 and print f1 score of training and dev data at every epoch

# In[ ]:


num_epochs = 120
max_f1 = 0
for epoch in range(num_epochs):
    model.train()
    true_labels_train = []
    pred_labels_train = []
    for i, (sentences, tags, lens) in enumerate(train_loader):
        optimizer.zero_grad()
        sentences, tags = sentences.to(device), tags.to(device)
        outputs = model((sentences, lens))
        predicted = torch.argmax(outputs, dim=2)
        temptags = tags.view(-1).cpu().numpy()
        temppred = predicted.view(-1).cpu().numpy()
        for ii in range(len(temptags)):
            if temptags[ii] == -1:
                continue
            else:
                true_labels_train.append(temptags[ii])
                pred_labels_train.append(temppred[ii])
    
        loss = criterion(outputs.view(-1, len(tag2idx)), tags.view(-1))
        loss.backward()
        optimizer.step()
        
    f1_train = f1_score(true_labels_train, pred_labels_train, average='macro', zero_division=1)
    
    model.eval()
    with torch.no_grad():
        true_labels_dev = []
        pred_labels_dev = []
        for sentences_dev, tags_dev, lens_dev in train_loaderdev:
            sentences_dev, tags_dev = sentences_dev.to(device), tags_dev.to(device)
            outputs_dev = model((sentences_dev, lens_dev))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temptagsdev = tags_dev.view(-1).cpu().numpy()
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for ii in range(len(temptagsdev)):
                if temptagsdev[ii] == -1:
                    continue
                else:
                    true_labels_dev.append(temptagsdev[ii])
                    pred_labels_dev.append(temppreddev[ii])

        f1 = f1_score(true_labels_dev, pred_labels_dev, average='macro', zero_division=1)
        scheduler.step(f1)
        if max_f1 < f1:
            max_f1 = f1
            print('model saved')
            torch.save(model.state_dict(), 'blstm2.pt')
    print(f'Epoch [{epoch+1}/{num_epochs}], f1-train:{f1_train}, Loss: {loss.item():.4f}, F1 Score: {f1:.4f}')


# ### Load the model with best f1 score on dev data

# In[ ]:


model.load_state_dict(torch.load('blstm2.pt'))


# ### Apply best model on dev data and generate dev1.out file with index, word, tag, pred_tag¶

# In[ ]:


model.eval()
with open('deveval2.out', 'w') as f:
    with torch.no_grad():
        sentences_dev, tags_dev = [], []
        for i in range(len(sentencesdev)):
            getsentence = sentencesdev[i]
            sentences_dev_idx = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in getsentence]
            gettag = tagsdev[i]
            sentences_dev_idx = torch.LongTensor(sentences_dev_idx).unsqueeze(0).to(device)
            gettlen = len(getsentence)
            outputs_dev = model((sentences_dev_idx, [gettlen]))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for i in range(len(gettag)):
                word = getsentence[i]
                tag = gettag[i]
                pred = idx2tag[temppreddev[i]]
                f.write(f"{i+1} {word} {tag} {pred}\n")
            f.write("\n")

model.eval()
with open('dev2.out', 'w') as f:
    with torch.no_grad():
        sentences_dev, tags_dev = [], []
        for i in range(len(sentencesdev)):
            getsentence = sentencesdev[i]
            sentences_dev_idx = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in getsentence]
            gettag = tagsdev[i]
            sentences_dev_idx = torch.LongTensor(sentences_dev_idx).unsqueeze(0).to(device)
            gettlen = len(getsentence)
            outputs_dev = model((sentences_dev_idx, [gettlen]))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for i in range(len(gettag)):
                word = getsentence[i]
                tag = gettag[i]
                pred = idx2tag[temppreddev[i]]
                f.write(f"{i+1} {word} {pred}\n")
            f.write("\n")


# ### result from running the script conll03eval.txt < dev2.out
# processed 51577 tokens with 5942 phrases; found: 5726 phrases; correct: 5178.
# accuracy:  97.67%; precision:  90.43%; recall:  87.14%; FB1:  88.76
#               LOC: precision:  95.12%; recall:  91.24%; FB1:  93.14  1762
#              MISC: precision:  86.55%; recall:  80.26%; FB1:  83.29  855
#               ORG: precision:  87.88%; recall:  82.70%; FB1:  85.21  1262
#               PER: precision:  89.50%; recall:  89.74%; FB1:  89.62  1847
# ### Read test data, where each line contains three space-separated values index, word, tag. Where sentence are append to sentences list and tag are append to tags list

# In[ ]:


testsentences = []
testsentence = []
with open('data/test') as f:
    for line in f:
        ls = (line.rstrip('\n')).split(" ")
        if len(ls) == 2:
            testsentence.append(ls[1])
        else:
            testsentences.append(testsentence)
            testsentence = []


# ### Apply best model on test data and generate test2.out file with index, word, pred_tag

# In[ ]:


model.eval()
with open('test2.out', 'w') as f:
    with torch.no_grad():
        sentences_test = [] 
        for i in range(len(testsentences)):
            getsentence = testsentences[i]
            sentences_test_idx = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in getsentence]
            sentences_test_idx = torch.LongTensor(sentences_test_idx).unsqueeze(0).to(device)
            gettlen = len(getsentence)
            outputs_dev = model((sentences_test_idx, [gettlen]))
            predicted_dev = torch.argmax(outputs_dev, dim=2)
            temppreddev = predicted_dev.view(-1).cpu().numpy()
            for i in range(len(getsentence)):
                word = getsentence[i]
                pred = idx2tag[temppreddev[i]]
                f.write(f"{i+1} {word} {pred}\n")
            f.write("\n")


# ### Bonus CNN-BiLSTM

# unfortunately I was not able to train and test the below model but I have commented my implementation

# ### generated char2idx

# In[ ]:


# char2idx = {}
# for sentence in sentences:
#     for word in sentence:
#         for char in word:
#             if char not in char2idx:
#                 char2idx[char] = len(char2idx)
# char2idx['<pad>'] = len(char2idx)
# char2idx['<unk>'] = len(char2idx)


# ### created CCNN BiLSTM model

# In[ ]:


# class CNNBiLSTM(nn.Module):
#     def __init__(self, vocab_size, char2idx, char_embed_dim, word_embed_dim, hidden_dim, num_layers, kernel_sizes, output_dim, dropout):
#         super(CNNBiLSTM, self).__init__()
#         self.word_embedding = nn.Embedding(vocab_size, word_embed_dim)
#         self.char_embedding = nn.Embedding(len(char2idx), char_embed_dim)
        
#         # Convolutional layer
#         self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=char_embed_dim, out_channels=hidden_dim, kernel_size=k) for k in kernel_sizes])
        
#         # BiLSTM layer
#         self.lstm = nn.LSTM(word_embed_dim+hidden_dim*len(kernel_sizes), hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)
        
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_dim*2, output_dim)
        
#         # Softmax activation
#         self.softmax = nn.LogSoftmax(dim=1)
        
#     def forward(self, text, text_lengths, chars, char_lengths):
#         word_embeds = self.dropout(self.word_embedding(text))
#         char_embeds = self.dropout(self.char_embedding(chars))
        
#         # Convolutional layer
#         char_embeds = char_embeds.permute(0, 2, 1)
#         conv_outputs = []
#         for conv in self.conv_layers:
#             conv_output = conv(char_embeds)
#             relu_output = nn.functional.relu(conv_output)
#             max_output, _ = relu_output.max(dim=2)
#             conv_outputs.append(max_output)
#         char_cnn = torch.cat(conv_outputs, dim=1)
        
#         # Concatenate word and character embeddings
#         combined_embeds = torch.cat((word_embeds, char_cnn), dim=2)
        
#         # Pack padded sequence
#         packed_embeds = nn.utils.rnn.pack_padded_sequence(combined_embeds, text_lengths, batch_first=True, enforce_sorted=False)
        
#         # LSTM layer
#         packed_outputs, _ = self.lstm(packed_embeds)
        
#         # Unpack padded sequence
#         outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
#         # Dropout layer
#         outputs = self.dropout(outputs)
        
#         # Fully connected layer
#         outputs = self.fc(outputs)
        
#         # Softmax activation
#         outputs = self.softmax(outputs)
        
#         return outputs


# ### data loader

# In[ ]:


# class NERDataset(Dataset):
#     def __init__(self, sentences, tags, word2idx, tag2idx, char2idx):
#         self.sentences = sentences
#         self.tags = tags
#         self.word2idx = word2idx
#         self.tag2idx = tag2idx
#         self.char2idx = char2idx
#         self.max_char_len = 15
        
#     def __len__(self):
#         return len(self.sentences)
    
#     def __getitem__(self, index):
#         sentence = self.sentences[index]
#         tags = self.tags[index]
        
#         # Convert words and tags to indices
#         word_indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]
#         tag_indices = [self.tag2idx[tag] for tag in tags]
        
#         # Convert characters to indices
#         char_indices = []
#         for word in sentence:
#             chars = []
#             for char in word:
#                 chars.append(self.char2idx.get(char, self.char2idx["<unk>"]))
#             chars = chars[:self.max_char_len] + [84] * (self.max_char_len - len(chars))
#             char_indices.append(chars)

#         return (torch.LongTensor(word_indices), torch.LongTensor(tag_indices), torch.LongTensor(char_indices))
    
# train_dataset = NERDataset(sentences, tags, word2idx, tag2idx, char2idx)

# def collate_fn(batch):
#     # Sort the batch by descending sentence length
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
    
#     # Separate the inputs and targets
#     words, tags, chars = zip(*batch)
    
#     # Compute the lengths of each sentence and character sequence
#     sentence_lengths = [len(sentence) for sentence in words]
#     char_lengths = [[len(word) for word in sentence] for sentence in chars]
    
#     # Pad the word and character sequences
#     padded_words = pad_sequence(words, batch_first=True, padding_value=word2idx['<PAD>'])
# #     padded_chars = [pad_sequence(chars[i], batch_first=True, padding_value=char2idx['<pad>']) for i in range(len(chars))]
#     padded_chars = []
# #     for i in range(len(chars)):
# #         sentence_chars = chars[i]
# #         print(sentence_chars)
# #         # Pad the character sequence for this sentence
# #         padded_sentence_chars = pad_sequence(sentence_chars, batch_first=True, padding_value=char2idx['<pad>'])
# #         # Add the padded character sequence to the list of padded character sequences
# #         padded_chars.append(padded_sentence_chars)
# #     # Pad the tag sequences
#     padded_tags = pad_sequence(tags, batch_first=True, padding_value=tag2idx['<PAD>'])
    
#     return (padded_words, sentence_lengths, chars, char_lengths, padded_tags)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# ### model initialization

# In[ ]:


# model = CNNBiLSTM(vocab_size=len(word2idx), 
#                   char2idx=char2idx, 
#                   char_embed_dim=30, 
#                   word_embed_dim=100, 
#                   hidden_dim=256, 
#                   num_layers=2, 
#                   kernel_sizes=[3, 5], 
#                   output_dim=len(tag2idx), 
#                   dropout=0.33)
# # vocab_size , char2idx, char_embed_dim, word_embed_dim, hidden_dim, num_layers, kernel_sizes, output_dim, dropout
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)


# ### train function

# In[ ]:


# def train(model, iterator, optimizer, criterion, device):
#     epoch_loss = 0
#     epoch_acc = 0
#     epoch_f1 = 0

#     model.train()

#     for i, batch in enumerate(iterator):
#         text, text_lengths, chars, char_lengths, tags = batch

#         text = text.to(device)
# #         chars = chars.to(device)
#         chars = tuple(char_tensor.to(device) for char_tensor in chars)

#         tags = tags.to(device)
# #         text_lengths = text_lengths.to(device)
# #         char_lengths = char_lengths.to(device)

#         optimizer.zero_grad()

#         predictions = model(text, text_lengths, chars, char_lengths)
#         predictions = predictions.view(-1, predictions.shape[-1])
#         tags = tags.view(-1)

#         loss = criterion(predictions, tags)
#         loss.backward()

#         optimizer.step()

#         # calculate metrics
#         acc, f1 = calculate_metrics(predictions, tags, tag2idx['O'], device)
#         epoch_loss += loss.item()
#         epoch_acc += acc
#         epoch_f1 += f1

#     return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)


# ### train for 10 epochs

# In[ ]:


# N_EPOCHS = 10
# best_valid_f1 = 0.0
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['<PAD>']).to(device)
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=15, verbose=True)
# for epoch in range(N_EPOCHS):

#     train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, device)

#     if valid_f1 > best_valid_f1:
#         best_valid_f1 = valid_f1
#         torch.save(model.state_dict(), 'cnn_bilstm_model.pt')

#     print(f'Epoch: {epoch+1:02}')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Val. F1: {valid_f1:.4f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




