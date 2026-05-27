# 기말고사 코드 채우기 문제은행

시험 범위:

- `p1ch6/1_neural_networks.ipynb`
- `p1ch7/2_birds_airplanes.ipynb`
- `p1ch8/1_convolution.ipynb`

이 문서는 개념 설명용이 아니라 **코드 채우기 시험 대비용**이다.  
문제는 보통 "이런 상황일 때 필요한 코드를 2줄, 3줄, 4줄, 5줄 내로 작성하시오"처럼 나온다고 보고 준비한다.

공부 방법:

1. `방향`을 보고 어떤 코드 흐름인지 떠올린다.
2. `정답 코드`를 손으로 따라 쓴다.
3. `암기 포인트`만 마지막에 다시 외운다.

---

## 1. p1ch6 - 신경망과 학습 루프

### 문제 1. 온도 리스트를 텐서로 바꾸고 열 벡터로 만드는 코드를 2줄로 작성하시오.

방향: `nn.Linear(1, 1)`에 넣으려면 데이터 모양이 `(샘플 수, 1)`이어야 한다.

정답 코드:

```python
t_c = torch.tensor(t_c).unsqueeze(1)  # 정답 온도 리스트를 열 벡터로 변환
t_u = torch.tensor(t_u).unsqueeze(1)  # 입력 온도 리스트를 열 벡터로 변환
```

암기 포인트: 리스트 -> `torch.tensor(...)` -> `.unsqueeze(1)`

### 문제 2. 전체 데이터 개수와 검증 데이터 개수를 구하는 코드를 2줄로 작성하시오.

방향: 전체 샘플 수에서 20%를 검증 데이터로 사용한다.

정답 코드:

```python
n_samples = t_u.shape[0]        # 전체 샘플 개수
n_val = int(0.2 * n_samples)    # 검증 데이터 개수
```

암기 포인트: 개수는 `shape[0]`, 20%는 `int(0.2 * n_samples)`

### 문제 3. 데이터를 무작위로 섞은 뒤 학습/검증 인덱스로 나누는 코드를 3줄로 작성하시오.

방향: `torch.randperm`으로 인덱스를 섞고, 앞쪽은 학습용, 뒤쪽은 검증용으로 나눈다.

정답 코드:

```python
shuffled_indices = torch.randperm(n_samples)  # 인덱스를 무작위로 섞음
train_indices = shuffled_indices[:-n_val]     # 앞부분은 학습용
val_indices = shuffled_indices[-n_val:]       # 뒷부분은 검증용
```

암기 포인트: `[:-n_val]`은 학습, `[-n_val:]`은 검증

### 문제 4. 인덱스를 이용해 학습/검증 데이터를 만드는 코드를 4줄로 작성하시오.

방향: 입력 `t_u`와 정답 `t_c`를 같은 인덱스로 나누어야 한다.

정답 코드:

```python
t_u_train = t_u[train_indices]  # 학습 입력
t_c_train = t_c[train_indices]  # 학습 정답
t_u_val = t_u[val_indices]      # 검증 입력
t_c_val = t_c[val_indices]      # 검증 정답
```

암기 포인트: 입력과 정답은 같은 인덱스를 사용한다.

### 문제 5. 입력 온도를 정규화하는 코드를 2줄로 작성하시오.

방향: 온도 값이 너무 크면 학습이 불안정하므로 0.1을 곱해 크기를 줄인다.

정답 코드:

```python
t_un_train = 0.1 * t_u_train  # 학습 입력 정규화
t_un_val = 0.1 * t_u_val      # 검증 입력 정규화
```

암기 포인트: `t_u`를 바로 쓰지 않고 `t_un`을 만들어 사용한다.

### 문제 6. 입력 1개를 받아 출력 1개를 내는 선형 모델을 만드는 코드를 2줄로 작성하시오.

방향: PyTorch의 기본 신경망 층은 `torch.nn` 안에 있다.

정답 코드:

```python
import torch.nn as nn             # 신경망 모듈 import
linear_model = nn.Linear(1, 1)    # 입력 1개, 출력 1개
```

암기 포인트: `nn.Linear(입력 개수, 출력 개수)`

### 문제 7. 선형 모델의 파라미터를 SGD로 학습하도록 optimizer를 만드는 코드를 2줄로 작성하시오.

방향: optimizer에는 모델의 학습 대상 파라미터와 learning rate를 넣는다.

정답 코드:

```python
import torch.optim as optim  # optimizer 모듈 import
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)  # SGD 설정
```

암기 포인트: `optim.SGD(model.parameters(), lr=...)`

### 문제 8. MSE loss를 직접 정의하는 코드를 3줄로 작성하시오.

방향: 예측값과 정답의 차이를 제곱하고 평균을 낸다.

정답 코드:

```python
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2  # 차이를 제곱
    return squared_diffs.mean()       # 평균 loss 반환
```

암기 포인트: `(예측 - 정답) ** 2` 후 `.mean()`

### 문제 9. 학습 데이터에 대해 forward와 loss 계산을 하는 코드를 2줄로 작성하시오.

방향: 모델에 입력을 넣어 예측값을 만들고, 예측값과 정답으로 loss를 계산한다.

정답 코드:

```python
t_p_train = model(t_u_train)              # forward
loss_train = loss_fn(t_p_train, t_c_train)  # 학습 loss 계산
```

암기 포인트: `예측 = model(입력)`, `loss = loss_fn(예측, 정답)`

### 문제 10. 학습 loop 안에서 파라미터 업데이트 핵심 3줄을 작성하시오.

방향: gradient 초기화, gradient 계산, 파라미터 업데이트 순서다.

정답 코드:

```python
optimizer.zero_grad()  # 이전 gradient 초기화
loss_train.backward()  # gradient 계산
optimizer.step()       # 파라미터 업데이트
```

암기 포인트: `zero_grad -> backward -> step`

### 문제 11. 검증 데이터에 대해 forward와 loss 계산을 하는 코드를 2줄로 작성하시오.

방향: 검증 loss는 성능 확인용이며, 보통 파라미터 업데이트에는 쓰지 않는다.

정답 코드:

```python
t_p_val = model(t_u_val)            # 검증 데이터 forward
loss_val = loss_fn(t_p_val, t_c_val)  # 검증 loss 계산
```

암기 포인트: 검증은 `model(...)`과 `loss_fn(...)`까지만 기억한다.

### 문제 12. PyTorch 내장 MSE loss를 사용하는 코드를 1줄로 작성하시오.

방향: 직접 만든 `loss_fn` 대신 `nn.MSELoss()`를 넣을 수 있다.

정답 코드:

```python
loss_fn = nn.MSELoss()  # 평균제곱오차 loss
```

암기 포인트: 회귀 문제는 `nn.MSELoss()`

### 문제 13. Sequential로 은닉층이 있는 신경망을 만드는 코드를 5줄 내로 작성하시오.

방향: `Linear -> Tanh -> Linear` 순서로 입력 1개를 출력 1개로 바꾼다.

정답 코드:

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),  # 입력 1개 -> 은닉층 13개
    nn.Tanh(),         # 비선형 활성화
    nn.Linear(13, 1)   # 은닉층 13개 -> 출력 1개
)
```

암기 포인트: `Linear -> Tanh -> Linear`

### 문제 14. Sequential 모델의 optimizer를 만드는 코드를 1줄로 작성하시오.

방향: 선형 모델과 똑같이 `seq_model.parameters()`를 넘긴다.

정답 코드:

```python
optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)  # Sequential 모델 학습 설정
```

암기 포인트: 모델 이름만 바뀌고 `parameters()`는 그대로다.

---

## 2. p1ch7 - CIFAR2 비행기/새 분류

### 문제 15. PyTorch 기본 import를 3줄로 작성하시오.

방향: 텐서, 신경망, optimizer를 각각 import한다.

정답 코드:

```python
import torch                 # PyTorch 기본
import torch.nn as nn        # 신경망 층과 loss
import torch.optim as optim  # optimizer
```

암기 포인트: `torch`, `torch.nn as nn`, `torch.optim as optim`

### 문제 16. torchvision의 데이터셋과 transform을 import하는 코드를 1줄로 작성하시오.

방향: CIFAR10을 불러오려면 `datasets`, 전처리는 `transforms`가 필요하다.

정답 코드:

```python
from torchvision import datasets, transforms  # 데이터셋과 전처리 도구
```

암기 포인트: `from torchvision import datasets, transforms`

### 문제 17. CIFAR10 학습 데이터를 불러오면서 Tensor 변환과 Normalize를 적용하는 코드를 5줄 내로 작성하시오.

방향: `datasets.CIFAR10` 안에 `transform=transforms.Compose([...])`를 넣는다.

정답 코드:

```python
cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))  # 정규화
    ]))
```

암기 포인트: `ToTensor()` 다음 `Normalize(mean, std)`

### 문제 18. CIFAR10 검증 데이터를 불러오는 코드를 5줄 내로 작성하시오.

방향: 학습 데이터와 거의 같지만 `train=False`를 사용한다.

정답 코드:

```python
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),  # 텐서 변환
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))  # 정규화
    ]))
```

암기 포인트: 학습은 `train=True`, 검증은 `train=False`

### 문제 19. CIFAR10에서 airplane과 bird만 뽑는 코드를 5줄 내로 작성하시오.

방향: 원래 라벨 0과 2만 남기고, 새 라벨을 0과 1로 바꾼다.

정답 코드:

```python
label_map = {0: 0, 2: 1}  # airplane은 0, bird는 1
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
```

암기 포인트: `label_map = {0: 0, 2: 1}`

### 문제 20. 검증 데이터에서도 airplane과 bird만 뽑는 코드를 3줄 내로 작성하시오.

방향: `cifar10_val`에도 같은 `label_map`을 적용한다.

정답 코드:

```python
cifar2_val = [(img, label_map[label])  # 검증 데이터 라벨도 0/1로 변경
              for img, label in cifar10_val
              if label in [0, 2]]
```

암기 포인트: 학습은 `cifar2`, 검증은 `cifar2_val`

### 문제 21. Linear 기반 분류 모델을 5줄 내로 작성하시오.

방향: CIFAR 이미지는 `3 * 32 * 32 = 3072`개 입력이고, 출력 클래스는 2개다.

정답 코드:

```python
model = nn.Sequential(
    nn.Linear(3072, 512),  # 이미지 3072개 숫자 -> 은닉층
    nn.Tanh(),             # 활성화 함수
    nn.Linear(512, 2),     # airplane/bird 2개 출력
    nn.LogSoftmax(dim=1)   # 클래스 차원 log 확률
)
```

암기 포인트: `3072 -> 512 -> 2`

### 문제 22. LogSoftmax 모델에 맞는 loss를 만드는 코드를 1줄로 작성하시오.

방향: 모델 마지막이 `LogSoftmax`이면 loss는 `NLLLoss`를 쓴다.

정답 코드:

```python
loss_fn = nn.NLLLoss()  # LogSoftmax와 짝
```

암기 포인트: `LogSoftmax + NLLLoss`

### 문제 23. CrossEntropyLoss를 쓰는 모델의 마지막 부분을 4줄 내로 작성하시오.

방향: `CrossEntropyLoss`를 쓸 때는 마지막에 `LogSoftmax`를 붙이지 않는다.

정답 코드:

```python
model = nn.Sequential(
    nn.Linear(512, 128),  # 은닉층
    nn.Tanh(),            # 활성화
    nn.Linear(128, 2)     # 최종 클래스 점수
)
```

암기 포인트: `CrossEntropyLoss`는 마지막 `LogSoftmax` 없이 사용한다.

### 문제 24. CrossEntropyLoss를 만드는 코드를 1줄로 작성하시오.

방향: 모델 출력이 raw score일 때 사용한다.

정답 코드:

```python
loss_fn = nn.CrossEntropyLoss()  # raw score용 분류 loss
```

암기 포인트: `LogSoftmax 없음 + CrossEntropyLoss`

### 문제 25. 이미지 한 장을 펼쳐서 모델에 넣는 코드를 2줄로 작성하시오.

방향: `img`는 `(3, 32, 32)`이므로 1차원으로 펼치고 배치 차원을 추가한다.

정답 코드:

```python
img_batch = img.view(-1).unsqueeze(0)  # 3072개로 펼친 뒤 배치 차원 추가
out = model(img_batch)                 # 모델 예측
```

암기 포인트: 한 장은 `img.view(-1).unsqueeze(0)`

### 문제 26. 이미지 한 장에 대해 loss를 계산하는 코드를 2줄로 작성하시오.

방향: label도 배치 형태로 맞추기 위해 리스트처럼 감싼다.

정답 코드:

```python
out = model(img.view(-1).unsqueeze(0))  # 이미지 한 장 예측
loss = loss_fn(out, torch.tensor([label]))  # label도 배치 형태로 변환
```

암기 포인트: `torch.tensor([label])`

### 문제 27. 학습 DataLoader를 만드는 코드를 2줄 내로 작성하시오.

방향: 학습 데이터는 보통 섞어서 배치로 꺼낸다.

정답 코드:

```python
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=True)  # 학습 데이터는 섞음
```

암기 포인트: 학습 loader는 `shuffle=True`

### 문제 28. 검증 DataLoader를 만드는 코드를 2줄 내로 작성하시오.

방향: 검증 데이터는 성능 측정용이라 보통 섞지 않는다.

정답 코드:

```python
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
                                         shuffle=False)  # 검증 데이터는 섞지 않음
```

암기 포인트: 검증 loader는 `shuffle=False`

### 문제 29. 배치 이미지를 Linear 모델에 넣기 위해 펼치는 코드를 2줄로 작성하시오.

방향: 배치 크기는 유지하고, 이미지 부분만 1차원으로 펼친다.

정답 코드:

```python
outputs = model(imgs.view(imgs.shape[0], -1))  # 배치 크기는 유지하고 펼침
loss = loss_fn(outputs, labels)                # 배치 전체 loss 계산
```

암기 포인트: 배치는 `imgs.view(imgs.shape[0], -1)`

### 문제 30. 배치 학습 루프 안의 핵심 5줄을 작성하시오.

방향: forward, loss, gradient 초기화, backward, step 순서다.

정답 코드:

```python
outputs = model(imgs.view(imgs.shape[0], -1))  # forward
loss = loss_fn(outputs, labels)                # loss 계산
optimizer.zero_grad()                          # gradient 초기화
loss.backward()                                # gradient 계산
optimizer.step()                               # 파라미터 업데이트
```

암기 포인트: `forward -> loss -> zero_grad -> backward -> step`

### 문제 31. 모델 출력에서 예측 클래스를 구하는 코드를 2줄로 작성하시오.

방향: 클래스 차원 `dim=1`에서 가장 큰 값의 위치가 예측 클래스다.

정답 코드:

```python
outputs = model(imgs.view(imgs.shape[0], -1))  # 모델 출력
_, predicted = torch.max(outputs, dim=1)       # 가장 큰 값의 인덱스가 예측 클래스
```

암기 포인트: 예측 클래스는 `torch.max(..., dim=1)`의 두 번째 값

### 문제 32. 정확도 계산에서 맞힌 개수와 전체 개수를 누적하는 코드를 2줄로 작성하시오.

방향: 전체 개수는 label 개수이고, 맞힌 개수는 `predicted == labels`의 합이다.

정답 코드:

```python
total += labels.shape[0]                    # 전체 샘플 수 누적
correct += int((predicted == labels).sum()) # 맞힌 개수 누적
```

암기 포인트: `total`은 개수, `correct`는 맞은 개수

### 문제 33. 평가할 때 gradient 계산을 끄는 코드를 4줄 내로 작성하시오.

방향: 검증/테스트에서는 파라미터를 업데이트하지 않으므로 `torch.no_grad()`를 사용한다.

정답 코드:

```python
with torch.no_grad():                         # 평가 중 gradient 계산 끔
    for imgs, labels in val_loader:
        outputs = model(imgs.view(imgs.shape[0], -1))  # forward만 수행
        _, predicted = torch.max(outputs, dim=1)        # 예측 클래스
```

암기 포인트: 평가는 `with torch.no_grad():`

### 문제 34. 모델의 전체 파라미터 개수를 세는 코드를 1줄로 작성하시오.

방향: 각 파라미터 텐서의 원소 개수를 모두 더한다.

정답 코드:

```python
sum([p.numel() for p in model.parameters()])  # 전체 파라미터 개수
```

암기 포인트: 파라미터 개수는 `p.numel()`

---

## 3. p1ch8 - CNN 합성곱 신경망

### 문제 35. CNN에서 자주 쓰는 import를 4줄로 작성하시오.

방향: CNN에서는 `nn`, `F`, `optim`을 같이 사용한다.

정답 코드:

```python
import torch                       # PyTorch 기본
import torch.nn as nn              # Conv2d, Linear 등
import torch.nn.functional as F    # max_pool2d, relu 등 함수형 API
import torch.optim as optim        # optimizer
```

암기 포인트: CNN은 `torch.nn.functional as F`를 자주 쓴다.

### 문제 36. RGB 이미지를 받아 16개 특징맵을 만드는 Conv2d를 1줄로 작성하시오.

방향: CIFAR 이미지는 RGB라서 입력 채널이 3개다.

정답 코드:

```python
conv = nn.Conv2d(3, 16, kernel_size=3)  # 입력 채널 3개, 출력 채널 16개
```

암기 포인트: `nn.Conv2d(입력채널, 출력채널, kernel_size=3)`

### 문제 37. 이미지 한 장에 Conv2d를 적용하는 코드를 2줄로 작성하시오.

방향: CNN 입력은 `(N, C, H, W)`이므로 배치 차원을 추가한다.

정답 코드:

```python
img, _ = cifar2[0]              # 이미지 한 장 가져오기
output = conv(img.unsqueeze(0)) # 배치 차원 추가 후 Conv2d 적용
```

암기 포인트: CNN에 한 장 넣을 때는 `img.unsqueeze(0)`

### 문제 38. padding을 넣어 출력 크기를 유지하는 Conv2d를 2줄로 작성하시오.

방향: `kernel_size=3`일 때 `padding=1`을 주면 가로/세로 크기를 유지하기 쉽다.

정답 코드:

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)  # padding으로 크기 유지
output = conv(img.unsqueeze(0))                   # 이미지 한 장 통과
```

암기 포인트: `kernel_size=3`이면 `padding=1`

### 문제 39. Conv2d의 bias를 0으로 만드는 코드를 2줄로 작성하시오.

방향: weight를 직접 수정할 때는 gradient 계산을 끈다.

정답 코드:

```python
with torch.no_grad():  # 직접 값 수정이므로 gradient 계산 끔
    conv.bias.zero_()  # bias를 0으로 초기화
```

암기 포인트: 직접 수정은 `with torch.no_grad():`

### 문제 40. 3x3 평균 필터를 만드는 코드를 2줄로 작성하시오.

방향: 3x3 칸이 9개이므로 모든 weight를 `1.0 / 9.0`으로 채운다.

정답 코드:

```python
with torch.no_grad():            # gradient 계산 없이 직접 수정
    conv.weight.fill_(1.0 / 9.0) # 평균 필터 값 채우기
```

암기 포인트: 평균 필터는 `fill_(1.0 / 9.0)`

### 문제 41. 세로 경계 검출 필터를 넣는 코드를 5줄 내로 작성하시오.

방향: 왼쪽은 음수, 가운데는 0, 오른쪽은 양수로 두면 세로 방향 변화가 강조된다.

정답 코드:

```python
with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0]])  # 세로 경계 필터
    conv.bias.zero_()  # bias 제거
```

암기 포인트: `[-1, 0, 1]`을 세 줄 반복

### 문제 42. MaxPool2d로 이미지 크기를 줄이는 코드를 2줄로 작성하시오.

방향: `MaxPool2d(2)`는 가로/세로 크기를 절반으로 줄인다.

정답 코드:

```python
pool = nn.MaxPool2d(2)          # 2x2 영역에서 최댓값 선택
output = pool(img.unsqueeze(0)) # 배치 차원 추가 후 pooling
```

암기 포인트: pooling은 `nn.MaxPool2d(2)`

### 문제 43. CNN Net 클래스의 `__init__`에서 층을 정의하는 코드를 5줄 내로 작성하시오.

방향: `conv1 -> conv2 -> fc1 -> fc2` 순서로 층을 준비한다.

정답 코드:

```python
self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # RGB -> 16채널
self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)   # 16채널 -> 8채널
self.fc1 = nn.Linear(8 * 8 * 8, 32)                       # 펼친 특징 -> 은닉층
self.fc2 = nn.Linear(32, 2)                               # 은닉층 -> 2클래스
```

암기 포인트: `3 -> 16 -> 8`, 그리고 `8 * 8 * 8 -> 32 -> 2`

### 문제 44. CNN forward의 합성곱과 pooling 부분을 2줄로 작성하시오.

방향: Conv2d 결과에 `tanh`를 적용하고, 바로 max pooling을 한다.

정답 코드:

```python
out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)    # conv1 -> tanh -> pool
out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)  # conv2 -> tanh -> pool
```

암기 포인트: `F.max_pool2d(torch.tanh(self.conv...), 2)`

### 문제 45. CNN forward에서 flatten 후 완전연결층을 통과시키는 코드를 4줄로 작성하시오.

방향: Conv 결과 `(N, 8, 8, 8)`을 `(N, 512)`로 펼친 뒤 Linear 층으로 보낸다.

정답 코드:

```python
out = out.view(-1, 8 * 8 * 8)  # CNN 특징을 1차원으로 펼침
out = torch.tanh(self.fc1(out)) # 첫 번째 완전연결층 + 활성화
out = self.fc2(out)             # 최종 클래스 점수
return out                      # 출력 반환
```

암기 포인트: CNN의 핵심은 `out.view(-1, 8 * 8 * 8)`

### 문제 46. CNN 전체 forward 함수를 5줄 내로 작성하시오.

방향: 합성곱/풀링 두 번, flatten, 완전연결층 두 번 순서다.

정답 코드:

```python
out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)    # conv1
out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)  # conv2
out = out.view(-1, 8 * 8 * 8)                       # flatten
out = torch.tanh(self.fc1(out))                     # fc1
out = self.fc2(out)                                 # fc2
```

암기 포인트: `conv -> pool -> conv -> pool -> view -> fc`

### 문제 47. CNN 학습 루프 안의 핵심 5줄을 작성하시오.

방향: CNN은 이미지 shape를 그대로 모델에 넣고, flatten은 모델 내부에서 한다.

정답 코드:

```python
outputs = model(imgs)       # CNN forward
loss = loss_fn(outputs, labels)  # loss 계산
optimizer.zero_grad()       # gradient 초기화
loss.backward()             # gradient 계산
optimizer.step()            # 파라미터 업데이트
```

암기 포인트: CNN에서는 `model(imgs)` 그대로 사용한다.

### 문제 48. CNN 학습 준비 코드를 4줄로 작성하시오.

방향: DataLoader, 모델, optimizer, loss를 준비한다.

정답 코드:

```python
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)  # 학습 loader
model = Net()                                             # CNN 모델 생성
optimizer = optim.SGD(model.parameters(), lr=1e-2)        # optimizer
loss_fn = nn.CrossEntropyLoss()                           # 분류 loss
```

암기 포인트: `DataLoader -> Net -> SGD -> CrossEntropyLoss`

### 문제 49. 학습 loss를 누적하는 코드를 2줄로 작성하시오.

방향: 배치마다 나온 loss를 숫자로 바꿔 더한다.

정답 코드:

```python
loss_train += loss.item()          # 현재 배치 loss를 누적
loss_train / len(train_loader)     # 평균 학습 loss
```

암기 포인트: loss 숫자는 `.item()`

### 문제 50. CNN 검증 함수 안에서 예측과 정확도 누적을 4줄로 작성하시오.

방향: 검증에서는 forward만 하고, `torch.max`로 예측 클래스를 구한다.

정답 코드:

```python
outputs = model(imgs)                       # forward
_, predicted = torch.max(outputs, dim=1)    # 예측 클래스
total += labels.shape[0]                    # 전체 개수
correct += int((predicted == labels).sum()) # 맞힌 개수
```

암기 포인트: 정확도는 `correct / total`

### 문제 51. 모델을 저장하는 코드를 1줄로 작성하시오.

방향: 모델 전체가 아니라 학습된 파라미터 사전인 `state_dict()`를 저장한다.

정답 코드:

```python
torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt')  # 모델 파라미터 저장
```

암기 포인트: 저장은 `model.state_dict()`

### 문제 52. 저장한 모델을 다시 불러오는 코드를 2줄로 작성하시오.

방향: 같은 구조의 모델을 먼저 만들고, 저장된 파라미터를 넣는다.

정답 코드:

```python
loaded_model = Net()  # 같은 구조의 모델 생성
loaded_model.load_state_dict(torch.load(data_path + 'birds_vs_airplanes.pt'))  # 파라미터 불러오기
```

암기 포인트: `Net()` 먼저, 그 다음 `load_state_dict`

### 문제 53. GPU가 있으면 GPU를 사용하고 아니면 CPU를 사용하는 코드를 2줄로 작성하시오.

방향: `torch.cuda.is_available()`로 CUDA 사용 가능 여부를 확인한다.

정답 코드:

```python
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))  # GPU 가능하면 cuda, 아니면 cpu
```

암기 포인트: `cuda if torch.cuda.is_available() else cpu`

### 문제 54. 모델과 배치 데이터를 같은 device로 보내는 코드를 3줄로 작성하시오.

방향: 모델, 입력, 정답이 모두 같은 device에 있어야 한다.

정답 코드:

```python
model = Net().to(device=device)  # 모델을 device로 이동
imgs = imgs.to(device=device)    # 입력 이미지 이동
labels = labels.to(device=device) # 정답 라벨 이동
```

암기 포인트: 모델과 데이터는 모두 `.to(device=device)`

### 문제 55. device를 사용하는 학습 루프 안에서 데이터 이동과 forward/loss를 4줄로 작성하시오.

방향: 먼저 `imgs`, `labels`를 device로 옮긴 뒤 모델에 넣는다.

정답 코드:

```python
imgs = imgs.to(device=device)      # 이미지 device 이동
labels = labels.to(device=device)  # 라벨 device 이동
outputs = model(imgs)              # forward
loss = loss_fn(outputs, labels)    # loss 계산
```

암기 포인트: `.to(device=device)`를 forward 전에 한다.

### 문제 56. 저장된 모델을 device에 맞게 불러오는 코드를 3줄 내로 작성하시오.

방향: CPU/GPU 환경이 달라질 수 있으므로 `map_location=device`를 사용한다.

정답 코드:

```python
loaded_model = Net().to(device=device)  # 모델을 device로 이동
loaded_model.load_state_dict(torch.load(data_path + 'birds_vs_airplanes.pt',
                                        map_location=device))  # 위치 맞춰 불러오기
```

암기 포인트: 불러올 때 device 문제는 `map_location=device`

### 문제 57. L2 regularization을 loss에 더하는 코드를 4줄로 작성하시오.

방향: 모든 파라미터를 제곱해서 더한 값을 원래 loss에 추가한다.

정답 코드:

```python
l2_lambda = 0.001                         # L2 강도
l2_norm = sum(p.pow(2.0).sum()
              for p in model.parameters()) # 모든 파라미터 제곱합
loss = loss + l2_lambda * l2_norm          # 원래 loss에 L2 penalty 추가
```

암기 포인트: L2는 `p.pow(2.0).sum()`

### 문제 58. Dropout2d 층을 만들고 forward에서 적용하는 코드를 2줄로 작성하시오.

방향: convolution 특징맵 일부를 학습 중 꺼서 과적합을 줄인다.

정답 코드:

```python
self.conv1_dropout = nn.Dropout2d(p=0.4)  # dropout 층 생성
out = self.conv1_dropout(out)             # forward에서 dropout 적용
```

암기 포인트: `nn.Dropout2d(p=0.4)`

### 문제 59. BatchNorm2d 층을 만들고 forward에서 적용하는 코드를 2줄로 작성하시오.

방향: 채널 수에 맞춰 BatchNorm2d를 만들고 conv 결과에 적용한다.

정답 코드:

```python
self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)  # 채널 수 기준
out = self.conv1_batchnorm(self.conv1(x))                     # conv 뒤 batchnorm
```

암기 포인트: `BatchNorm2d(num_features=채널수)`

### 문제 60. Residual 연결을 만드는 코드를 3줄로 작성하시오.

방향: 중간 출력 `out1`을 저장해 두고 나중 출력에 더한다.

정답 코드:

```python
out1 = out                                      # 이전 출력 저장
out = torch.relu(self.conv3(out)) + out1        # 현재 출력에 이전 출력 더하기
out = F.max_pool2d(out, 2)                      # pooling
```

암기 포인트: residual은 `현재 출력 + 이전 출력`

### 문제 61. ResBlock의 forward 핵심 코드를 4줄로 작성하시오.

방향: conv, batchnorm, relu를 거친 결과에 원래 입력 `x`를 더한다.

정답 코드:

```python
out = self.conv(x)        # convolution
out = self.batch_norm(out) # batch normalization
out = torch.relu(out)     # ReLU 활성화
return out + x            # 입력을 더해 residual 연결
```

암기 포인트: ResBlock 마지막은 `return out + x`

---

## 4. 진짜 최종 암기 세트

시험 직전에는 아래 코드를 손으로 여러 번 쓰는 것이 가장 중요하다.

### 1) 공통 학습 5줄

```python
outputs = model(imgs)              # forward
loss = loss_fn(outputs, labels)    # loss 계산
optimizer.zero_grad()              # gradient 초기화
loss.backward()                    # gradient 계산
optimizer.step()                   # 파라미터 업데이트
```

### 2) Linear 이미지 분류 학습 5줄

```python
outputs = model(imgs.view(imgs.shape[0], -1))  # 이미지 펼치기 + forward
loss = loss_fn(outputs, labels)                # loss 계산
optimizer.zero_grad()                          # gradient 초기화
loss.backward()                                # gradient 계산
optimizer.step()                               # 업데이트
```

### 3) CNN forward 5줄

```python
out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)    # conv1
out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)  # conv2
out = out.view(-1, 8 * 8 * 8)                       # flatten
out = torch.tanh(self.fc1(out))                     # fc1
out = self.fc2(out)                                 # fc2
```

### 4) 예측과 정확도 4줄

```python
outputs = model(imgs)                       # forward
_, predicted = torch.max(outputs, dim=1)    # 예측 클래스
total += labels.shape[0]                    # 전체 개수
correct += int((predicted == labels).sum()) # 맞힌 개수
```

### 5) CIFAR2 만들기 5줄

```python
label_map = {0: 0, 2: 1}  # airplane -> 0, bird -> 1
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
```

### 6) 저장/불러오기 3줄

```python
torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt')  # 저장
loaded_model = Net()  # 모델 구조 생성
loaded_model.load_state_dict(torch.load(data_path + 'birds_vs_airplanes.pt'))  # 불러오기
```

### 7) GPU 사용 4줄

```python
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))  # device 선택
model = Net().to(device=device)      # 모델 이동
imgs = imgs.to(device=device)        # 입력 이동
labels = labels.to(device=device)    # 정답 이동
```

### 8) 외워야 할 짝

```python
nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()
```

```python
loss_fn = nn.CrossEntropyLoss()
```

암기 포인트: `LogSoftmax + NLLLoss`, `LogSoftmax 없음 + CrossEntropyLoss`
