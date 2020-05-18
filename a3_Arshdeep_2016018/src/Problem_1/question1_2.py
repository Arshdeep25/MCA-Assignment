'''
Reference - https://github.com/n0obcoder/skip-gram-model
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import nltk
nltk.download('abc')
from nltk.corpus import abc


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


if __name__ == '__main__':

    model = torch.load(open('model_3.pkl', 'rb'), map_location=torch.device('cpu'))
    w2i = pickle.load(open('w2i.pkl', 'rb'))
    i2w = pickle.load(open('i2w.pkl', 'rb'))

    embeddings = model.embeddings.weight.data.cpu()
    print(embeddings.shape)

    tsne = TSNE(n_components=2, verbose=2, init='pca', random_state=32).fit_transform(embeddings)
    pickle.dump(tsne, open('tsne_3.pkl', 'wb'))
    # tsne = pickle.load(open('tsne.pkl', 'rb'))

    x, y = [], []
    annotations = []
    for idx, coord in enumerate(tsne):
        annotations.append(i2w[idx])
        x.append(coord[0])
        y.append(coord[1])

    plt.figure()
    plt.scatter(x, y)
    for i, txt in enumerate(annotations):
        plt.annotate(txt, (x[i], y[i]))

    plt.savefig('tsne_3.png')
