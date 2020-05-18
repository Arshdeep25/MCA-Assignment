'''
Reference - https://github.com/jojonki/word2vec-pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import pickle
import string
import nltk
nltk.download('abc')
from nltk.corpus import abc
import random


class WordDataset(Dataset):
    def __init__(self, skipgram_train, w2i):
        self.skipgram_train = skipgram_train
        self.w2i = w2i

    def __len__(self):
        return len(self.skipgram_train)

    def __getitem__(self, idx):
        return self.w2i[self.skipgram_train[idx][0]], self.w2i[self.skipgram_train[idx][1]], self.skipgram_train[idx][2]


class Model(nn.Module):
    def __init__(self, vocab_size, embedding):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding)

    def forward(self, input, context):
        batch_size = input.shape[0]
        input_embedding = self.embeddings(input).view((batch_size, 1, -1))
        context_embedding = self.embeddings(context).view((batch_size, 1, -1))
        score = torch.bmm(input_embedding.view(batch_size, 1, 100), context_embedding.view(batch_size, 100, 1))
        score = F.logsigmoid(score).view(batch_size, -1)

        return score


def train(vocab_size, training_data, w2i):
    embedding = 100
    model = Model(vocab_size, embedding).to(device)
    dataset = WordDataset(training_data, w2i)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=2048)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(5):
        total_loss = 0
        start_epoch = time.time()
        model.train()
        for input, context, output in dataloader:
            input = input.to(device)
            context = context.to(device)
            output = output.to(device).float()

            optimizer.zero_grad()
            score = model(input, context)

            loss = criterion(score.view(-1), output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch", epoch, " Loss :", total_loss / len(dataloader), " Time :", time.time() - start_epoch)
        pickle.dump(model, open('model_'+str(epoch)+'.pkl', 'wb'))
    return model


if __name__ == '__main__':

    device = 'cuda'
    context_size = 2
    text = abc.raw().split()
    text = [''.join(punct for punct in word if punct not in string.punctuation + '\n') for word in text]
    text = [word for word in text if word]

    vocab = set(text)
    vocab_size = len(vocab)

    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}

    pickle.dump(w2i, open('w2i.pkl', 'wb'))
    pickle.dump(i2w, open('i2w.pkl', 'wb'))

    training_data = []

    for i in range(context_size, len(text)-context_size):
        training_data.append((text[i], text[i - context_size], 1))
        training_data.append((text[i], text[i - context_size + 1], 1))
        training_data.append((text[i], text[i + context_size - 1], 1))
        training_data.append((text[i], text[i+context_size], 1))

        random_indices = []
        if i-context_size-1 >= 0:
            random_indices.append(random.randint(0, i-context_size-1))
            random_indices.append(random.randint(0, i - context_size - 1))
        if i+context_size+1 < len(text):
            random_indices.append(random.randint(i + context_size + 1, len(text)-1))
            random_indices.append(random.randint(i + context_size + 1, len(text)-1))

        for indices in random_indices:
            training_data.append((text[i], text[indices], 0))

    print(len(training_data))
    model = train(vocab_size, training_data, w2i)
