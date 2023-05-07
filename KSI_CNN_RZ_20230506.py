import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import copy
torch.manual_seed(1)

# Load the data
lab2ind = np.load('label_to_ix.npy', allow_pickle=True).item()
train_data = np.load('training_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)
val_data = np.load('val_data.npy', allow_pickle=True)
wor2ind = np.load('word_to_ix.npy', allow_pickle=True).item()
wikiinput = np.load('newwikivec.npy', allow_pickle=True)
wikivoc = np.load('wikivoc.npy', allow_pickle=True).item()

# Set the hyperparameters
wikisize, rvocsize = wikiinput.shape
wikivec = autograd.Variable(torch.FloatTensor(wikiinput))
embed_size, hidden_dim, batch_size, conv_out, top = 100, 200, 32, 100, 10
kernel_size = [3, 4, 5]

# Define the func_data_preprocess function for the dataset
def func_data_preprocess(data, batch_size=batch_size):
    processed_data = []

    # convert labels to vectors
    for i, note, j in data:
        temp_label = np.zeros(len(lab2ind), dtype=float)
        temp_label[[lab2ind[jj] for jj in j if jj in wikivoc]] = 1.0
        processed_data.append((i, note, temp_label))

    processed_data = np.array(processed_data, dtype=object)

    # sort by length of i
    lengths = np.array([len(item[0]) for item in processed_data])
    sorted_indices = lengths.argsort()
    processed_data = processed_data[sorted_indices]

    # create batches
    batched_data = []
    for start_ix in range(0, len(processed_data) - batch_size + 1, batch_size):
        block = processed_data[start_ix:start_ix+batch_size]
        max_word_num = max(len(item[0]) for item in block)

        # create main matrix
        main_matrix = np.zeros((len(block), max_word_num), dtype=np.int64)
        for i, item in enumerate(block):
            ix = [wor2ind[word] for word in item[0][:max_word_num] if word in wor2ind]
            main_matrix[i, :len(ix)] = ix

        # gather notes and labels
        notes = np.array([item[1] for item in block])
        labels = np.array([item[2] for item in block])

        # wrap in torch variables
        batched_data.append((
            autograd.Variable(torch.from_numpy(main_matrix)),
            autograd.Variable(torch.FloatTensor(notes)),
            autograd.Variable(torch.FloatTensor(labels))
        ))

    return batched_data

class CNN(nn.Module):
    def __init__(self, batch_size, vocab_size, target_size):
        super(CNN, self).__init__()
        
        # Embedding layer with dropout
        self.embeddings = nn.Sequential(
            nn.Embedding(vocab_size + 1, embed_size, padding_idx=0),
            nn.Dropout(p=0.2)
        )
        
        # Convolutional layers with varying kernel sizes
        self.convs1 = nn.Conv1d(embed_size, conv_out, kernel_size[0])
        self.convs2 = nn.Conv1d(embed_size, conv_out, kernel_size[1])
        self.convs3 = nn.Conv1d(embed_size, conv_out, kernel_size[2])
        
        # Linear layers
        self.hidden2tag = nn.Linear(300, target_size)
        self.layer2 = nn.Linear(embed_size, 1, bias=False)
        self.embedding = nn.Linear(rvocsize, embed_size)
        self.attention = nn.Linear(embed_size, embed_size)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, vec1, nvec, wiki, simlearning):
        # Apply embedding and dropout to the input
        x = self.embeddings(vec1).transpose(1, 2)

        # Apply convolutional layers
        output1 = self.tanh(self.convs1(x))
        output1 = nn.MaxPool1d(output1.size()[2])(output1)

        output2 = self.tanh(self.convs2(x))
        output2 = nn.MaxPool1d(output2.size()[2])(output2)

        output3 = self.tanh(self.convs3(x))
        output3 = nn.MaxPool1d(output3.size()[2])(output3)

        # Concatenating convolutional outputs
        output4 = torch.cat([output1, output2, output3], 1).squeeze(2)

        # Similarity learning block
        if simlearning:
            # Expand nvec and wiki to match dimensions
            nvec = nvec.view(batch_size, 1, -1).expand(batch_size, wiki.size()[0], -1)
            wiki = wiki.view(1, wiki.size()[0], -1).expand(nvec.size())
            
            # Apply linear layers and activation function
            new = self.embedding(wiki * nvec)
            attention = self.sigmoid(self.attention(new))
            vec3 = self.layer2(new * attention).view(batch_size, -1)
            
            # Compute final tag scores with similarity learning
            tag_scores = self.sigmoid((self.hidden2tag(output4).detach() + vec3))
        else:
            # Compute final tag scores without similarity learning
            tag_scores = self.sigmoid(self.hidden2tag(output4))

        return tag_scores

def func_train_model(model, sim):
    print('start_training')
    modelsaved = []
    modelperform = []
    top = 10

    bestresults = -1
    bestiter = -1
    for epoch in range(5000):
        model.train()
        lossestrain = []

        # Training loop
        for mysentence in train_data_batch:
            model.zero_grad()
            targets = mysentence[2].cuda()
            tag_scores = model(mysentence[0].cuda(), mysentence[1].cuda(), wikivec.cuda(), sim)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            lossestrain.append(loss.data.mean())

        print(epoch)
        modelsaved.append(copy.deepcopy(model.state_dict()))
        print("--------------------------")
        model.eval()

        recall = []
        
        # Validation loop
        for inputs in val_data_batch:
            targets = inputs[2].cuda()
            tag_scores = model(inputs[0].cuda(), inputs[1].cuda(), wikivec.cuda(), sim)
            loss = loss_function(tag_scores, targets)

            targets = targets.data.cpu().numpy()
            tag_scores = tag_scores.data.cpu().numpy()

            # Compute recall for each validation example
            for iii in range(len(tag_scores)):
                temp = {iiii: tag_scores[iii][iiii] for iiii in range(len(tag_scores[iii]))}
                temp1 = sorted(temp.items(), key=lambda x: x[1], reverse=True)
                thistop = int(np.sum(targets[iii]))
                hit = sum([1.0 for ii in temp1[:max(thistop, top)] if targets[iii][ii[0]] == 1.0])

                if thistop != 0:
                    recall.append(hit / thistop)

        # Calculate and print validation recall
        avg_recall = np.mean(recall)
        print(f'validation top-{top} {avg_recall}')

        # Update model performance
        modelperform.append(avg_recall)
        if avg_recall > bestresults:
            bestresults = avg_recall
            bestiter = len(modelperform) - 1

        # Early stopping
        if (len(modelperform) - bestiter) > 5:
            print(modelperform, bestiter)
            return modelsaved[bestiter]
        
def filter_empty_columns(y_true, y_scores):
    temptrue = []
    tempscores = []
    for col in range(len(y_true)):
        if np.sum(y_true[col]) != 0:
            temptrue.append(y_true[col])
            tempscores.append(y_scores[col])
    return np.array(temptrue).T, np.array(tempscores).T        
 
def func_test_model(modelstate, sim):
    model = CNN(batch_size, len(wor2ind), len(lab2ind))
    model.cuda()
    loss_function = nn.BCELoss()
    model.eval()

    # Initialize metrics
    recall = []
    lossestest = []
    y_true = []
    y_scores = []

    # Test loop
    for inputs in test_data_batch:
        targets = inputs[2].cuda()
        tag_scores = model(inputs[0].cuda(), inputs[1].cuda(), wikivec.cuda(), sim)
        loss = loss_function(tag_scores, targets)

        targets = targets.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

        lossestest.append(loss.data.mean())
        y_true.append(targets)
        y_scores.append(tag_scores)

        # Compute recall for each test example
        for iii in range(len(tag_scores)):
            temp = {iiii: tag_scores[iii][iiii] for iiii in range(len(tag_scores[iii]))}
            temp1 = sorted(temp.items(), key=lambda x: x[1], reverse=True)
            thistop = int(np.sum(targets[iii]))
            hit = sum([1.0 for ii in temp1[:max(thistop, top)] if targets[iii][ii[0]] == 1.0])

            if thistop != 0:
                recall.append(hit / thistop)

    # Prepare data for metric calculation
    y_true = np.concatenate(y_true, axis=0).T
    y_scores = np.concatenate(y_scores, axis=0).T
    y_true, y_scores = filter_empty_columns(y_true, y_scores)
    y_pred = (y_scores > 0.5).astype(np.int64)

    # Calculate and print metrics
    print('test loss', np.mean([loss.cpu().item() for loss in lossestest]))
    print(f'top-{top}', np.mean(recall))
    print('macro AUC', roc_auc_score(y_true, y_scores, average='macro'))
    print('micro AUC', roc_auc_score(y_true, y_scores, average='micro'))
    print('macro F1', f1_score(y_true, y_pred, average='macro'))
    print('micro F1', f1_score(y_true, y_pred, average='micro'))

#===========================================Main Model===============================================

train_data_batch=func_data_preprocess(train_data)
test_data_batch=func_data_preprocess(test_data)
val_data_batch=func_data_preprocess(val_data)

# Initialize and train the base model
model = CNN(batch_size, len(wor2ind), len(lab2ind))
model.cuda()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
basemodel = func_train_model(model, 0)
torch.save(basemodel, 'CNN_model')
print("--------------------------")
print('CNN alone:')
func_test_model(basemodel, 0)
print('===============================')

# Initialize and train the KSI model
model = CNN(batch_size, len(wor2ind), len(lab2ind))
model.cuda()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
KSImodel = func_train_model(model, 1)
torch.save(KSImodel, 'KSI_CNN_model')
print("--------------------------")
print('KSI+CNN:')
func_test_model(KSImodel, 1)