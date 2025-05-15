import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from utilities import train_model

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = 'frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
BEAM_SIZE = 8
NUM_LETTERS = 11
MAX_LENGTH = 50
MODEL_PATH = "frankenstein_lstm.pth"

# Load and clean text
file = open(INPUT_FILE_NAME, 'r', encoding='utf-8-sig')
text = file.read()
file.close()
text = text.lower().replace('\n', ' ').replace('  ', ' ')

# Encode characters
unique_chars = list(set(text))
char_to_index = dict((ch, index) for index, ch in enumerate(unique_chars))
index_to_char = dict((index, ch) for index, ch in enumerate(unique_chars))
encoding_width = len(char_to_index)

# Create training data
fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])

X = np.zeros((len(fragments), WINDOW_LENGTH, encoding_width), dtype=np.float32)
y = np.zeros(len(fragments), dtype=np.int64)
for i, fragment in enumerate(fragments):
    for j, char in enumerate(fragment):
        X[i, j, char_to_index[char]] = 1
    y[i] = char_to_index[targets[i]]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.05, random_state=0)
trainset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
testset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))

# Model definition
class LastTimestep(nn.Module):
    def forward(self, inputs):
        return inputs[1][0][1]  # Get last layer hidden state

model = nn.Sequential(
    nn.LSTM(encoding_width, 128, num_layers=2, dropout=0.2, batch_first=True),
    LastTimestep(),
    nn.Dropout(0.2),
    nn.Linear(128, encoding_width)
)

optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# Train or load model
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(device)
else:
    print("Training new model...")
    train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset, optimizer, loss_function, 'acc')
    torch.save(model.state_dict(), MODEL_PATH)

# Beam search prediction
letters = 'the body '
one_hots = []
for char in letters:
    x = np.zeros(encoding_width)
    x[char_to_index[char]] = 1
    one_hots.append(x)
beams = [(np.log(1.0), letters, one_hots)]

for i in range(NUM_LETTERS):
    minibatch_list = [triple[2] for triple in beams]
    minibatch = np.array(minibatch_list, dtype=np.float32)
    inputs = torch.from_numpy(minibatch).to(device)
    outputs = model(inputs)
    outputs = F.softmax(outputs, dim=1)
    y_predict = outputs.cpu().detach().numpy()

    new_beams = []
    for j, softmax_vec in enumerate(y_predict):
        triple = beams[j]
        for k in range(BEAM_SIZE):
            char_index = np.argmax(softmax_vec)
            new_prob = triple[0] + np.log(softmax_vec[char_index])
            new_letters = triple[1] + index_to_char[char_index]
            x = np.zeros(encoding_width)
            x[char_index] = 1
            new_one_hots = triple[2].copy()
            new_one_hots.append(x)
            new_beams.append((new_prob, new_letters, new_one_hots))
            softmax_vec[char_index] = 0
    new_beams.sort(key=lambda tup: tup[0], reverse=True)
    beams = new_beams[0:BEAM_SIZE]

# Output results
for item in beams:
    print(item[1])
