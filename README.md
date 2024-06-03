# INTRO
## Ⅰ. Deep Learing (딥러닝)
<img src='./a_intro//images/deep_learning.png'>

<br>

- 인공 신경망(Artificial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
- 인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
- 인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
- 자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
- 입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
- 초기 인공 신경망(Perceptron = 뉴런)에서 깊게 층을 쌓아 학습하는 딥러닝으로 발전한다.
- 딥러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 세 가지 층이 존재한다.

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

# PERCEPTRON
## Ⅰ. SLP
#### 1. SLP (Single Layer Perceptron), 단층 퍼셉트론, 단일 퍼셉트론
가장 단순한 형태의 신경망으로써, Hidden Layer가 없고 Single Layer로 구성되어 있다.  
퍼셉트론의 구조는 입력 features와 가중치, activation function, 출력 값으로 구성되어 있다.

신경 세포에서 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신하고,  
입력 값과 가중치 값은 모두 인공 뉴런(활성 함수)으로 도착한다.

가중치의 값이 클수록 해당 입력 값이 중요하다는 뜻이고,  
인공 뉴런(활성 함수)에 도착한 각 입력 값이 가중치 값을 곱한 뒤 전체 합한 값을 구한다.

인공 뉴런(활성 함수)는 보통 시그모이드 함수와 같은 계단 함수를 사용하여, 합한 값을 확률로 변환하고  
이 때, 임계치를 기준으로 0 또는 1을 출력한다.

<br>

<img src="./b_perceptron/images/perceptron01.png">

<br>

로지스틱 회귀 모델이 인공 신경망에서는 하나의 인공 뉴런으로 볼 수 있다.  
결과적으로 퍼셉트론은 회귀 모델과 마찬가지로 실제 값과 예측 값의 차이가 최소가 되는 가중치 값을 찾는 과정이 퍼셉트론 학습과정이다.

최초 가중치 값을 설정한 뒤 입력 feature 값으로 예측 값을 계산하고, 실제 값과 차이를 구한 뒤 이를 줄일 수 있도록 가중치 값을 변경한다.

퍼셉트론의 활성화 정도를 편향(bias)으로 조절할 수 있으며, 편향을 통해 어느정도의 자극을 미리 주고 시작할 수 있다.

뉴런이 활성화되기 위해 필요한 자극이 1000이라고 가정하면, 입력 값을 500만 받아도 편향을 2로 주어 10000을 만들 수 있다.

<br>

<div style="display: flex; width: 500px; margin-left: 20px;">
    <div style="width: 200px;">
        <img src="./b_perceptron/images/perceptron02.png" width="200px" style="margin-top: 3px;"> 
    </div>
    <div style="width: 225px">
        <img src="./b_perceptron/images/perceptron03.png" width="225px" style="margin-left: 50px;">
    </div>
</div>

<br>

퍼셉트론의 출력 값과 실제 값의 차이를 줄여나가는 방향성으로 계속해서 가중치 값을 변경하며, 이 때 경사하강법을 사용한다.

<br>

<img src='./b_perceptron/images/gd01.gif' width='600px'>

<br></br>
<br></br>

#### 2. SGD (Stochastic Gradient Descent), 확률적 경사 하강법
경사 하강법 방식은 전체 학습 데이터를 기반으로 계산한다. 하지만, 입력 데이터가 크고 레이어가 많을 수록 많은 자원이 소모된다.  
일반적으로 메모리 부족으로 인해 연산이 불가능하기 때문에, 이를 극복하기 위해 SGD 방식이 도입되었다.

전체 학습 데이터 중 단 한 건만 임의로 선택하여 경사 하강법을 실시하는 방식을 의미한다.  
많은 건 수 중에 한 건만 실시하기 때문에, 빠르게 최적점을 찾을 수 있지만 노이즈가 심하다.  
무작위로 추출된 샘플 데이터에 대해 경사 하강법을 실시하기 때문에 진폭이 크고 불안정해 보일 수 있다.  
일반적으로 사용되지 않고, SGD를 얘기할 때에는 보통 미니 배치 경사 하강법을 의미한다.

<br>

<img src='./b_perceptron/images/gd02.png' width='600px'>

<br></br>
<br></br>

#### 3. Mini-Batch Gradient Descent, 미니 배치 경사 하강법
전체 학습 데이터 중, 특정 크기(Batch 크기)만큼 임의로 선택해서 경사 하강법을 실시한다.  
이 또한, 확률적 경사 하강법이다.

<br>

<img src='./b_perceptron/images/gd03.png' width='800px'>

<br>

- 전체 학습 데이터가 1000건이라고 하고, batch size를 100건이라 가정하면,  
  전체 데이터를 batch size만큼 나눠서 가져온 뒤 섞고, 경사하강법을 계산한다.  
  이 경우, 10번 반복해야 1000개의 데이터가 모두 학습되고 이를 epoch라고 한다. 즉, 10 epoch * 100 batch이다.  
<sub>*batch size는 제곱에 따라 개수를 정해주면 좋음</sub>

<br>

<img src='./b_perceptron/images/gd04.png' width='650px'>

<br></br>
<br></br>
<br></br>

## Ⅱ. MLP
### 1. Multi Layer Perceptron, 다층 퍼셉트론, 다중 퍼셉트론
보다 복잡한 문제의 해결을 위해서 입력층과 출력층 사이에 은닉층이 포함되어 있다.  
퍼셉트론을 여러층 쌓은 인공 신경망으로서, 각 층에서는 활성함수를 통해 입력을 처리한다.  
층이 깊어질 수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.

<div style="display: flex">
    <div>
        <img src="./b_perceptron/images/mlp01.png" width="500">
    </div>
    <div>
        <img src="./b_perceptron/images/mlp02.png" width="600" style="margin-top:50px; margin-left: 10px">
    </div>
</div>

<br></br>
<br></br>

### 2. ANN (Artificial Neural Network), 인공 신경망
은닉층이 1개일 경우 이를 인공 신경망이라고 한다.

<br>

<img src="./b_perceptron/images/ann.png" width="300px">

<br></br>
<br></br>

### 3. DNN (Deep Neural Network), 심층 신경망
은닉층이 2개 이상일 경우 이를 심층 신경망이라고 한다.

<br>

<img src="./b_perceptron/images/dnn.png" width="450px">

<br></br>
<br></br>

### 4. BackPropagation, 역전파
심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만, 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.

즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트하며, 손실 함수의 최소 값을 찾아야 하는데,  
순방향으로는 복잡한 미분식을 계산할 수가 없다. 따라서, 미분의 연쇄 법칙(Chain Rule)을 사용하여 역방향으로 편미분을 진행한다.

<br></br>

#### 4-1. 합성 함수의 미분
<img src="./b_perceptron/images/chain_rule01.png" width="150px">  

<br>

---

<br>

<img src="./b_perceptron/images/chain_rule02.png" width="550px">

<br>

<img src="./b_perceptron/images/backpropagation01.png" width="800px"> 

<br> 

<img src="./b_perceptron/images/backpropagation02.png" width="800px"> 

<br>

<img src="./b_perceptron/images/backpropagation03.png" width="500px">  

<br></br>
<br></br>
<br></br>

## Ⅲ. Activation Function
### 1. Activation Function, 활성화 함수
인공 신경망에서 입력 값에 가중치를 곱한 뒤 합한 결과를 적용하는 함수이다.

<br>

---

<br>

#### 1-1. sigmoid, 시그모이드 함수 (이진분류)
은닉층이 아닌 최종 활성화 함수. 즉, 출력층에서 사용된다.  
은닉에서 사용 시, 입력 값이 양의 방향으로 큰 값일 경우 출력 값의 변화가 없으며, 음의 방향도 마찬가지이다.  
평균이 0이 아니기 때문에 정규 분포 형태가 아니고, 이는 방향에 따라 기울기가 달라져서 탐색 경로가 비효율적(지그재그)이 된다.

<br>

<img src='./b_perceptron/images/sigmoid.png' width='500px'>

<br></br>

#### 1-2. softmax, 소프트맥스 함수 (다중분류)
은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.  
시그모이드와 유사하게 0 ~ 1 사이의 값을 출력하지만, 이진 분류가 아닌 다중 분류를 통해 모든 확률 값이 1이 되도록 해준다.  
여러 개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.

<br>

<img src='./b_perceptron/images/softmax.png' width='450px'>

<br></br>

#### 1-3. tangent, 탄젠트 함수
은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.  
은닉층에서 사용 시, 시그모이드와 달리 -1 ~1 사이의 값을 출력해서 평균이 0이 될 수 있지만,  
여전히 입력 값의 양의 방향으로 큰 값일 경우 출력값의 변화가 미비하고 음의 방향도 마찬가지이다.

<br>

<img src='./b_perceptron/images/tanh.png' width='650px'>

<br></br>

#### 1-4. relu, 렐루 함수
대표적인 은닉층의 활성 함수이다.  
입력 값이 0보다 작으면 출력은 0이 되고, 0 보다 크면 입력값을 그대로 출력하게 된다.  
<sub>*max(0, input) = 최대값을 구해주는 함수, 음수가 나오면 안 되고, 음수 부분은 0으로 취급함</sub>

<br>

<img src='./b_perceptron/images/relu.png' width='480px'>

<br></br>
<br></br>

### 2. Cross Entropy (손실 함수)
실제 데이터의 확률 분포와, 학습된 모델이 계산한 확률 분포의 차이를 구하는데 사용된다.  
분류 문제에서 원-핫 인코딩을 통해 사용할 수 있는 오차 계산법이다.  
<sub>각 샘플에 대해 교차 엔트로피를 계산하고, 모두 더해서 평균을 내면 모델의 손실 값이 됨</sub>

<br>

<img src="./b_perceptron/images/cross_entropy01.png" width="350">
<img src="./b_perceptron/images/cross_entropy02.png" width="700" style="margin-left: 20px">

<br></br>
<br></br>
<br></br>

## Ⅳ. Optimizer
### 1. Optimizer, 최적화
최적의 경사 하강법을 적용하기 위해 필요하며, 최솟값을 찾아가는 방법들을 의미한다.  
loss를 줄이는 방향으로 최소 loss를 보다 빠르고 안정적으로 수렴할 수 있어야 한다.

<br>

<img src="./b_perceptron/images/optimizer.png" width="650px">

<br></br>

#### 1-1. Momentum
가중치를 계속 업데이트할 때마다 이전의 값을 일정 수준 반영시키면서 새로운 가중치로 업데이트한다.  
지역 최소값에서 벗어나지 못하는 문제를 해결할 수 있으며, 진행했던 방향만큼 추가적으로 더하여, 관성처럼 빠져나올 수 있게 해준다.

<br>

<img src="./b_perceptron/images/momentum.png" width="600px">

<br></br>

#### 1-2. AdaGrad (Adaptive Gradient)
가중치 별로 서로 다른 학습률을 동적으로 적용한다.  
적게 변화된 가중치는 보다 큰 학습률을 적용하고, 많이 변화된 가중치는 보다 작은 학습률을 적용시킨다.  
처음에는 큰 보폭으로 이동하다가 최소값에 가까워질 수록 작은 보폭으로 이동하게 된다.  
과거의 모든 기울기를 사용하기 때문에 학습률이 급격히 감소하여, 분모가 커짐으로써 학습률이 0에 가까워지는 문제가 있다.

<br>

<div style="display: flex">
    <div>
        <img src="./b_perceptron/images/adagrad01.png" width="100" style="margin-top: 20px;">
    </div>
    <div>
        <img src="./b_perceptron/images/adagrad02.png" width="400" style="margin-left: 20px">
    </div>
</div>

<br></br>

#### RMSProp (Root Mean Sqaure Propagation)
- AdaGrad의 단점을 보완한 기법으로서, 학습률이 지나치게 작아지는 것을 막기 위해 지수 가중 평균법(exponentially weighted average)을 통해 구한다.
- 지수 가중 평균법이란, 데이터의 이동 평균을 구할 때 오래된 데이터가 미치는 영향을 지수적으로 감쇠하도록 하는 방법이다.
- 이전의 기울기들을 똑같이 더해가는 것이 아니라 훨씬 이전의 기울기는 조금 반영하고 최근의 기울기를 많이 반영한다.  
  <sub>*과거의 것을 약하게, 최근의 것을 강하게 함(직전의 영향력이 크다)</sub>
- feature마다 적절한 학습률을 적용하여 효율적인 학습을 진행할 수 있고, AdaGrad보다 학습을 오래 할 수 있다.

#### Adam (Adaptive Moment Estimation)
- Momentum과 RMSProp 두 가지 방식을 결합한 형태로서, 진행하던 속도에 관성을 주고, 지수 가중 평균법을 적용한 알고리즘이다.
- 최적화 방법 중에서 가장 많이 사용되는 알고리즘이며, 수식은 아래와 같다.

<div style="display: flex">
    <div>
        <img src="./images/adam01.png" width="300" style="margin-top: 20px; margin-left: 0">
    </div>
    <div>
        <img src="./images/adam02.png" width="200" style="margin-top: 20px; margin-left: 80px">
    </div>
</div>