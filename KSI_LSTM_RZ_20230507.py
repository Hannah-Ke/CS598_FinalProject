import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(1)

# Load data and mappings
lab2ind = np.load('label_to_ix.npy', allow_pickle=True).item()
train_data = np.load('training_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)
val_data = np.load('val_data.npy', allow_pickle=True)
wor2ind = np.load('word_to_ix.npy', allow_pickle=True).item()
wikiinput = np.load('newwikivec.npy', allow_pickle=True)
wikivoc = np.load('wikivoc.npy', allow_pickle=True).item()

# Set wikivec and batch size
wikisize, rvocsize = wikiinput.shape
wikivec = autograd.Variable(torch.FloatTensor(wikiinput))
embed_size, hidden_dim, batch_size, top = 100, 200, 32, 10

def func_data_preprocess(data):
    # Convert labels to one-hot encoding and store in new_data
    new_data = []
    for i, note, j in data:
        temp_label = np.zeros(len(lab2ind), dtype=float)
        for label in j:
            if label in wikivoc:
                temp_label[lab2ind[label]] = 1.0
        new_data.append((i, note, temp_label))
    new_data = np.array(new_data, dtype=object)

    # Sort new_data by the length of the first element (i.e., the text)
    new_data = new_data[np.argsort([len(i[0]) for i in new_data])]

    # Create batches
    batch_data = []
    for start_ix in range(0, len(new_data) - batch_size + 1, batch_size):
        this_block = new_data[start_ix:start_ix + batch_size]
        my_bsize, max_num_words = len(this_block), max(len(ii[0]) for ii in this_block)
        
        # Create main_matrix with word indices
        main_matrix = np.zeros((my_bsize, max_num_words), dtype=np.int64)
        for i in range(my_bsize):
            for j in range(len(this_block[i][0])):
                word = this_block[i][0][j]
                if word in wor2ind:
                    main_matrix[i, j] = wor2ind[word]

        # Extract xxx2 and yyy (features and labels)
        xxx2 = np.array([ii[1] for ii in this_block])
        yyy = np.array([ii[2] for ii in this_block])

        # Convert to torch variables and append to batch_data
        batch_data.append((autograd.Variable(torch.from_numpy(main_matrix)), autograd.Variable(torch.FloatTensor(xxx2)), autograd.Variable(torch.FloatTensor(yyy))))

    return batch_data

class LSTM(nn.Module):

    def __init__(self, batch_size, vocab_size, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size + 1, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

        self.layer2 = nn.Linear(embed_size, 1, bias=False)
        self.embedding = nn.Linear(rvocsize, embed_size)
        self.vattention = nn.Linear(embed_size, embed_size, bias=False)

        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.embed_drop = nn.Dropout(p=0.2)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda())

    def forward(self, vec1, nvec, wiki, simlearning):
        # Get word embeddings and apply dropout
        thisembeddings = self.embed_drop(self.word_embeddings(vec1).transpose(0, 1))

        # Compute new if simlearning is 1
        if simlearning == 1:
            nvec = nvec.view(batch_size, 1, -1).expand(batch_size, wiki.size()[0], -1)
            wiki = wiki.view(1, wiki.size()[0], -1).expand(nvec.size()[0], -1, wiki.size()[1])
            new = self.embedding(wiki * nvec)
            vattention = self.sigmoid(self.vattention(new))
            new = new * vattention
            vec3 = self.layer2(new).view(batch_size, -1)

        # Pass embeddings through LSTM
        lstm_out, self.hidden = self.lstm(thisembeddings, self.hidden)

        # Compute output1 with MaxPool1d
        lstm_out = lstm_out.transpose(0, 2).transpose(0, 1)
        output1 = nn.MaxPool1d(lstm_out.size()[2])(lstm_out).view(batch_size, -1)

        # Compute final tag scores
        vec2 = self.hidden2tag(output1)
        tag_scores = self.sigmoid(vec2.detach() + vec3) if simlearning == 1 else self.sigmoid(vec2)

        return tag_scores
        
def func_train_model(model, sim):
    print('start_training')
    
    model_saved = []
    model_perform = []
    top_k = 10
    best_results = -1
    best_iter = -1
    
    for epoch in range(5000):
        model.train()
        losses_train = []
        recall = []
        
        # Training loop
        for sentence in train_data_batch:
            model.zero_grad()
            model.hidden = model.init_hidden()
            targets = sentence[2].cuda()
            tag_scores = model(sentence[0].cuda(), sentence[1].cuda(), wikivec.cuda(), sim)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.data.mean())
        
        print(epoch)
        model_saved.append(copy.deepcopy(model.state_dict()))
        print("-------------------------------------------------------------")
        
        model.eval()
        recall = []
        
        # Validation loop
        for inputs in val_data_batch:
            model.hidden = model.init_hidden()
            targets = inputs[2].cuda()
            tag_scores = model(inputs[0].cuda(), inputs[1].cuda(), wikivec.cuda(), sim)
            loss = loss_function(tag_scores, targets)
            
            targets = targets.data.cpu().numpy()
            tag_scores = tag_scores.data.cpu().numpy()
            
            # Calculate recall
            for i in range(len(tag_scores)):
                temp = {idx: tag_scores[i][idx] for idx in range(len(tag_scores[i]))}
                temp1 = sorted(temp.items(), key=lambda x: x[1], reverse=True)
                this_top = int(np.sum(targets[i]))
                hit = sum([1 for idx_score in temp1[0:max(this_top, top_k)] if targets[i][idx_score[0]] == 1.0])
                
                if this_top != 0:
                    recall.append(hit / this_top)
        
        print('validation top-', top_k, np.mean(recall))
        model_perform.append(np.mean(recall))
        
        if model_perform[-1] > best_results:
            best_results = model_perform[-1]
            best_iter = len(model_perform) - 1
        
        # Early stopping
        if (len(model_perform) - best_iter) > 5:
            print(model_perform, best_iter)
            return model_saved[best_iter]

def func_test_model(model_state, sim):
    model = LSTM(batch_size, len(wor2ind), len(lab2ind))
    model.cuda()
    # model.load_state_dict(model_state)
    loss_function = nn.BCELoss()
    model.eval()

    recall = []
    losses_test = []
    y_true = []
    y_scores = []

    # Testing loop
    for inputs in test_data_batch:
        model.hidden = model.init_hidden()
        targets = inputs[2].cuda()
        tag_scores = model(inputs[0].cuda(), inputs[1].cuda(), wikivec.cuda(), sim)
        loss = loss_function(tag_scores, targets)

        targets = targets.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

        losses_test.append(loss.data.mean())
        y_true.append(targets)
        y_scores.append(tag_scores)

        # Calculate recall
        for i in range(len(tag_scores)):
            temp = {idx: tag_scores[i][idx] for idx in range(len(tag_scores[i]))}
            temp1 = sorted(temp.items(), key=lambda x: x[1], reverse=True)
            this_top = int(np.sum(targets[i]))
            hit = sum([1 for idx_score in temp1[0:max(this_top, top)] if targets[i][idx_score[0]] == 1.0])

            if this_top != 0:
                recall.append(hit / this_top)

    y_true, y_scores = np.concatenate(y_true, axis=0), np.concatenate(y_scores, axis=0)
    y_true, y_scores = y_true.T, y_scores.T

    # Remove columns with no true labels
    temp_true = [col for col in y_true if np.sum(col) != 0]
    temp_scores = [y_scores[i] for i, col in enumerate(y_true) if np.sum(col) != 0]
    y_true, y_scores = np.array(temp_true).T, np.array(temp_scores).T

    y_pred = (y_scores > 0.5).astype(np.int64)
    
    print('test loss', np.mean([loss.cpu().item() for loss in losses_test]))
    print('top-', top, np.mean(recall))
    print('macro AUC', roc_auc_score(y_true, y_scores, average='macro'))
    print('micro AUC', roc_auc_score(y_true, y_scores, average='micro'))
    print('macro F1', f1_score(y_true, y_pred, average='macro'))
    print('micro F1', f1_score(y_true, y_pred, average='micro'))

#===========================================Main Model===============================================

# Preprocess datasets
train_data_batch = func_data_preprocess(train_data)
test_data_batch = func_data_preprocess(test_data)
val_data_batch = func_data_preprocess(val_data)

model = LSTM(batch_size, len(wor2ind), len(lab2ind))
model.cuda()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
basemodel= func_train_model(model, 0)
torch.save(basemodel, 'LSTM_model')

print ('LSTM alone:           ')
func_test_model(basemodel, 0)
print ('====================================================')

model = LSTM(batch_size, len(wor2ind), len(lab2ind))
model.cuda()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
KSImodel= func_train_model(model, 1)
torch.save(KSImodel, 'KSI_LSTM_model')

print ('KSI+LSTM:           ')
func_test_model(KSImodel, 1)