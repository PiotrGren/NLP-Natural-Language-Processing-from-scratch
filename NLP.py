import wget

wget.download("http://bigbang.prz.edu.pl/nmum/word2vec/names.txt")


names = open('names.txt', 'r').read().splitlines()
print(names[:10])
print(len(names))
print(min(len(name) for name in names)) #shortest name
print(max(len(name) for name in names)) #longesst name



#Creation of a dictionary to store BIGRAMS
BGs = {}
for name in names:
  #To every name in the dictionary we add a beginning/end sign 2hich we select as a dot '.'
  string = ['.'] + list(name) + ['.']

  #Creating bigrams
  for ch1, ch2 in zip(string, string[1:]):
    bigram = (ch1, ch2)
    BGs[bigram] = BGs.get(bigram, 0) + 1



#Dictionary b sorted by frequency of occurrence of bigrams
sorted(BGs.items(), key = lambda kv: -kv[1])



#What characters (letters) appear in the name file, i.e. tokens#
chars = sorted(list(set(''.join(names))))
print(chars)
print(len(chars))


#stoi - string_to_index
stoi = {s:i+1 for i,s in enumerate(chars)}

#At position 0 we add a token representing the beginning and end of the word (name)
stoi['.'] = 0

#itos - index_to_string
itos = {i:s for s,i in stoi.items()}


#We will use the pytorch library to learn ANN
import torch

N = torch.zeros((27, 27), dtype=torch.int32)

for w in names:
  #chs - to skrót od char (znak)
  chs = ['.'] + list(w) + ['.']
  #sprawdzamy kazda pare znakow
  for ch1, ch2 in zip(chs, chs[1:]):
    #i w tablicy N zapamietujemy, ze dane znaki tworzyly pare
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1


import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')


#Change frequency of bigrams into probability
P = N[0,:].float()
P = P / P.sum()

seed = 1234

#Using the pytorch library, we will generate the index of the next token using the information about the probability of occurrence, which is stored in the array P
g = torch.Generator().manual_seed(seed)
ix = torch.multinomial(P, num_samples=1, replacement=True, generator=g).item()
#Next token
itos[ix]


#Example for 3 tokens
seed = 123456789
g = torch.Generator().manual_seed(seed)

#We generate a random array made up of 3 cells
P3 = torch.rand(3, generator=g)
print("Frequencies of token occurrence")
print(P3)

P3 = P3 / P3.sum()
print("\nProbabilities of occurrence of 3 tokens")
print(P3)

torch.multinomial(P3, num_samples=100, replacement=True, generator=g)



#N+1 - some pairs of tokens do not appear in the body, e.g. (ó,ó),
#For such pairs, the cost function (expressing how well the generated name resembles a name from the dictionary) will have the value infinity, log(probabilities_of_appearance_pairs) = infinity
#To prevent this, we add 1 to all token pairs.

P = (N+1).float()

#/= works faster than P = P / sth , because it does not create a new array in memory
P /= P.sum(1, keepdims=True)



import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(P, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, str(P[i, j].item())[:5] , ha="center", va="top", color='gray')
plt.axis('off')

#Set the grain to any number you like
seed = 169783

g = torch.Generator().manual_seed(seed)

for i in range(25):

  out = []
  ix = 0 #zaczynamy generacje od kropki, indeks kropki to 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0: #jesli wylosujemy kropke to konczymy generacje
      break
  print(''.join(out))

  M = torch.ones((27, 27), dtype=torch.int32)
PP = M.float()/27

g = torch.Generator().manual_seed(seed)

random_names = []

for i in range(25):

  out = []
  ix = 0
  while True:
    pp = PP[ix]
    ix = torch.multinomial(pp, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
  random_names.append(''.join(out))




  #Entire Collection
log_likelihood = 0.0
n = 0

for w in names[:25]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]

    logprob = -1*torch.log(prob)
    log_likelihood += logprob
    n += 1
  
  print(f'{w}  : {log_likelihood/n:.4f}')




#Single word

#Insert english name (in lowercase)
#e.g. first_name='andrei'

first_name = 'piotr'

log_likelihood = 0.0
n = 0

for w in [first_name]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]

    logprob = -1*torch.log(prob)
    log_likelihood += logprob
    n += 1
  
  print(f'{w}  : {log_likelihood/n:.4f}')


  #Randomly generated words

log_likelihood = 0.0
n = 0

for w in random_names:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]

    logprob = -1*torch.log(prob)
    log_likelihood += logprob
    n += 1
  
  print(f'{w}  : {log_likelihood/n:.4f}')



#Create the training set of bigrams (x,y)
#x - features (arguments)
#y - answers (function value for arguments)

xs, ys = [], []

for w in names[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)


import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()

plt.imshow(xenc)


import torch.nn.functional as F
yenc = F.one_hot(ys, num_classes=27).float()
plt.imshow(yenc)


W = torch.randn((27, 27))
xenc @ W


logits = xenc @ W # log-counts
counts = logits.exp() # equivalent N

probs = counts / counts.sum(1, keepdims=True)
probs


seed = 169783
g = torch.Generator().manual_seed(seed)
W = torch.randn((27, 27), generator=g)

print(W)


xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
# btw: the last 2 lines here are together called a 'softmax'






nlls = torch.zeros(5)
for i in range(5):
  x = xs[i].item()
  y = ys[i].item()

  print('--------')
  print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
  print('input to the neural net:', x)
  print('output probabilities from the neural net:', probs[i])
  print('label (actual next character):', y)

  p = probs[i, y]
  print('probability assigned by the net to the the correct character:', p.item())

  logp = torch.log(p)
  print('log likelihood:', logp.item())

  nll = -logp
  print('negative log likelihood:', nll.item())
  
  nlls[i] = nll

print('=========')
print('=========')
print('=========')

print('average negative log likelihood, i.e. loss =', nlls.mean().item())



#Randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(169783)
W = torch.randn((27, 27), generator=g, requires_grad=True)



# forward pass
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean()


# backward pass
W.grad = None # set to zero the gradient
loss.backward()


# Update weights to better values as we have done so far 
#0.1 is learning rate
#W.grad is the derivative (by conjecture, the derivative of the cost function after the weights)

W.data += -0.1 * W.grad


seed= 169783

#Create the dataset
xs, ys = [], []
for w in names:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

#Initialize the 'network'
g = torch.Generator().manual_seed(seed)
W = torch.randn((27, 27), generator=g, requires_grad=True)


learning_rate = 1.5
epochs = 1500

#Gradient descent
for k in range(epochs):

  #Forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())

  #Backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  #Update weights
  W.data += - learning_rate * W.grad


  g = torch.Generator().manual_seed(2147483647)

for i in range(25):

  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    p = counts / counts.sum(1, keepdims=True)
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import urllib.request
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


torch.manual_seed(169783)


#In context_size we specify the number of words of the sequence based on which we predict the next word
CONTEXT_SIZE = 3

#Size of the vector representation of the token (word)
EMBEDDING_DIM = 10

#The text is divided into single words so as to create tokens
test_sentence = """Empathy for the poor may not come easily to people who never experienced it. They may blame the victims and insist their predicament can be overcome through determination and hard work.
But they may not realize that extreme poverty can be psychologically and physically incapacitating — a perpetual cycle of bad diets, health care and education exacerbated by the shaming and self-fulfilling prophecies that define it in the public imagination.
Gordon Parks — perhaps more than any artist — saw poverty as “the most savage of all human afflictions” and realized the power of empathy to help us understand it. It was neither an abstract problem nor political symbol, but something he endured growing up destitute in rural Kansas and having spent years documenting poverty throughout the world, including the United States.
That sensitivity informed “Freedom’s Fearful Foe: Poverty,” his celebrated photo essay published in Life magazine in June 1961. He took readers into the lives of a Brazilian boy, Flavio da Silva, and his family, who lived in the ramshackle Catacumba favela in the hills outside Rio de Janeiro. These stark photographs are the subject of a new book, “Gordon Parks: The Flavio Story” (Steidl/The Gordon Parks Foundation), which accompanies a traveling exhibition co-organized by the Ryerson Image Centre in Toronto, where it opens this week, and the J. Paul Getty Museum. Edited with texts by the exhibition’s co-curators, Paul Roth and Amanda Maddox, the book also includes a recent interview with Mr. da Silva and essays by Beatriz Jaguaribe, Maria Alice Rezende de Carvalho and Sérgio Burgi.
""".split()


char_num = 0
for item in test_sentence:
  for letter in item:
    char_num += 1
print(char_num)

import nltk
nltk.download('punkt')
nltk.download('stopwords')

#Function that takes the text from the specified URL and cleans it up
def read_data(file_path):
    data = urllib.request.urlopen(file_path)
    data = data.read().decode('utf8')
    tokenized_data = word_tokenize(data)

    stop_words = set(stopwords.words('english'))

    stop_words.update(['.',',',':',';','(',')','#','--','...','"'])
    cleaned_words = [ i for i in tokenized_data if i not in stop_words ]
    
    return(cleaned_words)


ngrams = []
for i in range(len(test_sentence) - CONTEXT_SIZE):
    tup = [test_sentence[j] for j in np.arange(i , i + CONTEXT_SIZE) ]
    ngrams.append((tup,test_sentence[i + CONTEXT_SIZE]))

#How many ngrams are there
print(len(ngrams))

vocab = set(test_sentence)
print(vocab)

vocab = set(test_sentence)
print("Length of vocabulary",len(vocab))

word_to_ix = {word: i for i, word in enumerate(vocab)}

for key in word_to_ix.keys():
  print(key, word_to_ix[key])


def get_key(word_id):
    for key, val in word_to_ix.items():
        if(val == word_id):
            print(key)

#wczytuje sie reprezentcje wektorowa tokenow jako tablice np.array
def cluster_embeddings(filename,nclusters):
    X = np.load(filename)
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X)
    center = kmeans.cluster_centers_
    distances = euclidean_distances(X,center)

    for i in np.arange(0,distances.shape[1]):
        word_id = np.argmin(distances[:,i])
        print(word_id)
        get_key(word_id)





class CBOWModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOWModeler, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        
        out1 = F.relu(self.linear1(embeds))
        
        out2 = self.linear2(out1)
        
        
        log_probs = F.log_softmax(out2, dim=1)
        return log_probs

    def predict(self,input):
        
        context_idxs = torch.tensor([word_to_ix[w] for w in input], dtype=torch.long)
        res = self.forward(context_idxs)
        
        res_arg = torch.argmax(res)

        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        
        for arg in zip(res_val,res_ind):
            print(arg)
            print([(key,val,arg[0]) for key,val in word_to_ix.items() if val == arg[1]])

    def freeze_layer(self,layer):
        for name,child in model.named_children():
            print(name,child)
            if(name == layer):
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())
                    params.requires_grad= False

    def print_layer_parameters(self):
        for name,child in model.named_children():
                print(name,child)
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())

    def write_embedding_to_file(self,filename):
        #zmienna "i" reprezentuje token
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(filename,weights)



losses = []

loss_function = nn.NLLLoss()

model = CBOWModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(400):
    total_loss = 0

    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        
        model.zero_grad()

        log_probs = model(context_idxs)
        
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(total_loss)
    losses.append(total_loss)


#Print the model layer parameters
model.print_layer_parameters()


#Predict the next word given n context words
print(model.predict(['of','all','human']))

print(model.predict(['he','was','not']))



model.write_embedding_to_file('embeddings.npy')
embeddings = np.load('embeddings.npy')
embeddings