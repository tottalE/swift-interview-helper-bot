import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r', encoding="UTF-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
#각 문장에 loop을 돌면서 패턴을 찾아냄
for intent in intents['intents']:
    tag = intent['tag']
    # tag list에 추가
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize
        w = tokenize(pattern)
        # word list에 추가
        all_words.extend(w)
        # xy pair에 추가
        xy.append((w, tag))

# stem 하고 필터링을 함
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# 반복되는 것을 삭제하고 정렬함
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

#트레이닝 데이터 생성
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# 하이퍼 파라미터
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # 인덱싱 지원 함수
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    #사이즈 리턴 함수
    def __len__(self):
        return self.n_samples

#데이터 셋과 모델을 불러오고 모델 인스턴트 생성
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#인스턴스를 model에 있는 클래스로 생성함
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss와 optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#모델을 실제로 train
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # 포워드 패스
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # backword와 optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

#학습 저장
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
