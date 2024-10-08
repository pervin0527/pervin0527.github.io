---
layout: single
title: "Generative Models"
categories: GenAI
tag: [Deep-Learning, ComputerVision, Generative]
use_math: true
---

생성형 모델들을 알아보자.

# 1.사전지식들

생성형 모델들은 unsupervised, semi-supervised learning을 기반으로 하는 경우가 대부분이라 복잡한 수식들이 굉장히 많다. 해당 수식들을 하나하나 따라갈 수 있는 능력을 기른다면 정말 좋겠지만, 단기간에 되는 것이 아니기 때문에 이번 글에서 소개, 정리할 내용들을 이해하기 위해 필요한 지식들을 정리하자.

## 1-1.확률변수와 확률변수

확률변수와 확률분포의 수학적 정의는 다음과 같다.

- 표본공간 : 시행에서 얻을 수 있는 모든 결과들의 집합.
- 확률변수 : 표본공간에 포함된 각각의 원소들을 목적에 맞는 하나의 실수로 대응 시키는 함수.
- 확률분포 : 확률변수의 원소들을 각 사건이 발생할 확률로 대응시키는 함수.

예를 들어 동전 두 개를 동시에 던졌을 때 얻은 시행 [TT, TH, HT, HH]에서 앞면의 개수라는 목적에 따라 [0, 1, 2]로 각각의 원소들을 정수로 대응시키는 것이 확률변수다. 이어서 확률변수의 정수형 원소들을 해당 사건이 일어날 확률로 대응시키는 것이 확률분포다.

그렇다면 **생성형 모델이 학습하는 학습 데이터셋의 분포**는 무엇일까??

주관적인 해석은 다음과 같다.
- 데이터셋이 포함하고 있는 패턴이나 특성들은 이미지의 픽셀이나 텍스트의 토큰과 같이 정수형 데이터들을 기반으로 잠재되어 있다고 볼 수 있다.   
- 쉽게 말해 픽셀들이 모여 전경의 형태나 경계, 배경의 색상 등의 여러가지 패턴들은 잠재되어 우리가 쉽게 구할 수 없는 복잡한 확률분포로 구성되어 있다고 볼 수 있는 것이다.   

## 1-2.Manifold

<img src="{{site.url}}/images/240820/0001.jpg" width="1200" height="300">

설명을 위해 각각의 데이터 샘플은 [28, 28, 1]의 이미지라고 해보자. 

이러한 데이터 샘플들을 점(point)으로 포함하는 784차원의 공간을 \$ R^m $\이라고 할 때,  Manifold는 이러한 전체공간 중에서 데이터 샘플들의 핵심 특성들을 포함하는 부공간(subspace)를 말한다. 쉽게 말해 전체공간처럼 모든 데이터 샘플들을 포함하지만 차원이 더 낮은 공간인 것이다.

<img src="{{site.url}}/images/240820/0002.jpg" width="1200" height="300">

정의를 보면 알겠지만 manifold를 찾는 것은 생성 작업에만 국한되는 것이 아니라 분류나 회귀 같은 모든 작업에 있어 최우선이 되는 핵심적인 과제인데, 다음과 같이 정리할 수 있다.

1. 데이터를 잘 압축(compress, encoding)하게 되면 불필요한 정보는 제거하고 필수 정보들만 남겨놓게 되어 연산 크기를 줄이는 효과가 있다.
2. 사람은 3차원을 넘어가는 공간을 시각적으로 표현하는데 굉장히 어려운데, 압축이 잘되어 2차원, 3차원 공간으로 줄여버리고 공간에 투영하면 시각화 자료를 완성할 수 있고 이를 이용해 설명할 수 있다.(explainable)
3. ```차원의 저주```를 피할 수 있다.
  - 우리가 현실에 사용하는 이미지는 HD, FHD 이상을 가볍게 넘어가버리기 때문에 전체 공간의 차원은 굉장히 큰 공간이라 할 수 있다. 
  - 문제는 차원의 증가에 따라 공간이 커지더라도 결국 각각의 이미지는 공간내 점이기 때문에 공간상 점들의 밀도가 낮아 모델 입장에서는 학습할 내용에 비해 양이 작은 상태가 되므로 많은 양의 데이터를 확보가 필수적이다.
  - manifold를 잘 찾으면 불필요한 차원을 배제하는 효과가 있어 전체공간보다 밀도가 높아지므로 결국 적은 데이터로도 좋은 성능의 모델을 만들 수 있다.

<img src="{{site.url}}/images/240820/0003.jpg" width="1200" height="300">

manifold가 크리티컬한 또 다른 이유는 ```의미적 거리 또는 유사도```에 있다. 

그림의 왼쪽을 보면 전체공간의 데이터 A1, A2, B간의 거리가 표시되어 있는데 전체공간에서는 A1과 B과 B가 가깝지만 manifold 공간인 오른쪽 그림에서는 A2와 B가 더 가까운 것을 볼 수 있다. 

이는 전체공간(고차원 공간)에서 데이터 샘플들간 거리가 가까워도 의미적(semantic)으로는 가깝지 않다는 것을 말하는데, 자세한 내용은 이어지는 VAE에서 설명하겠다.

## 1-3.Likelihood

Likelihood(가능도)의 정의는 **주어진 데이터나 관찰된 결과가 특정한 모형이나 가정 아래에서 발생할 확률**이다.

무슨 말인지 이해하기 어려울텐데 쉽게 말하면 관측치(데이터)가 주어지면 특정 파라미터 $\theta$를 갖는 모델에서 관측될 확률이 가능도이다.

<img src="{{site.url}}/images/240820/0006.png" width="1200" height="300">

예를 들어 앞면과 뒷면이 나올 확률이 동일한 동전을 10번 던졌을 때 앞면이 7번 나왔을 때,
- 모델(특정한 모형이나 가정) : 앞면과 뒷면이 나올 확률이 동일하다.
- 데이터(관측치, 결과) : 앞면이 7번, 뒷면이 3번 나왔다.

이 상황에서 가능도는 ```모델이 가진 파라미터 $\theta$를 조절해서 관측치가 발생할 확률이다.```

참고로 첫번째 식이 가능도이고 두번째 식은 로그 가능도(Log Likelihood)이다. 로그를 적용함으로써 곱셈이 덧셈으로 바뀌었고 이로 인해 미분이 가능해지기 때문에 DNN에서는 로그 가능도를 목적으로 설정한다.

## 1-4.Maximum Likelihood Estimation(MLE)

가능도에 대해 이해했다면 최대 가능도 추정법에 대해 이해할 수 있다.

<img src="{{site.url}}/images/240820/0006-1.png" width="1200" height="300">

결국 우리가 만드는 DNN은 입력에 따른 출력을 반환하는 함수이며 출력의 형태는 우리가 함수를 어떻게 정의하는가에 따라 달라진다. 
- 함수의 정의는 우리가 풀고자하는 문제에 해당하니 회귀 문제를 풀어야 한다면 연속적인 값을, 분류 문제를 풀어야 한다면 확률을 반환한다.
- 또한 본 적 없는 데이터가 입력되더라도 정확한 예측을 해야하고, 출력에 대한 신뢰도가 높을수록 강건한 모델이라 할 수 있다.
- 이를 Supervised Learning 기준에서 해석해보면, 입력 데이터 $ x $가 주어졌을 때 정해진 정답 $ y $를 출력할 확률 $ p(y \mid x) $을 최대화하는 것이 모델의 목적이자 Maximum Likelihood Estimation과 같다. 
- 즉, 최대 가능도 추정법은 우리가 가정한 모델에서 단순한 확률이나 조건부 확률을 최대화하는 파라미터 $ \theta $를 추정하는 것이다.

<img src="{{site.url}}/images/240820/0006-2.png" width="1200" height="300">

참고로 두 분포를 유사한 정도를 계산하는 지표가 KL-Divergence이며, KL Divergence를 최소화하는 것이 Maximum Likelihood를 최대화 하는 것과 같다.


# 2.AutoEncoder

## 2-1.Basic AutoEncoder

기본적인 오토인코더는 비선형 차원축소(Dimension Reduction)와 재건(Reconstruction)을 수행하는 모델이다.

<img src="{{site.url}}/images/240820/0007.jpg" width="1200" height="300">

- 구조는 그림과 같이 Encoder-Decoder로 구성되어 있으며 Encoder는 차원축소, Decodeer는 재건하는 방법을 학습한다.
- Encoder는 입력된 이미지에서 핵심적인 특징을 추출하는 Feature Extraction을 수행하고, 압축된 결과물인 latent vector(잠재벡터)를 출력한다.
- Decoder는 잠재벡터를 입력 받아서 원래의 이미지록 복원한다.

Decoder의 작업을 보면 알겠지만 encoder의 입력이 곧 decoder가 만들어야할 정답이다. 따라서 오토인코더는 Self-supervised learning이 아닌 supervised learning을 수행한다.

>손실함수를 BCE나 MSE 중에서 어떤 것을 사용해야할까??
>{: .notice--info}

## 2-2.Variational AutoEncoder

<img src="{{site.url}}/images/240820/0008.jpg" width="1200" height="300">

VAE(변분 오토인코더)는 오토인코더와 마찬가지로 Encoder-Decoder 구조를 가지지만 이미지 생성이 목적인 모델이다.

- 그에 따라 Encoder는 학습 데이터셋의 분포 $p_{data}$의 평균을 예측해 $p_{data}$와 근사하는 분포를 추정하는 것이 목적이다.
- 이 때 Encoder가 예측하는 분포와 이상적인 분포는 우리가 샘플링 하기 쉽게 하기 위해 정규분포(가우시안)이라 가정한다.
- Decoder는 encoder에서 추정한 분포로 샘플링을 수행하여 잠재변수(latent variable)을 입력으로 받아 $p_{data}$내 데이터를 생성한다.

### 2-3.Manifold 공간상 거리
encoder로 이상적인 분포 $p_{data}$와 근사하는 분포를 추정하는 것은 어떤 의도로 설계된 것일까??

<img src="{{site.url}}/images/240820/0009.jpg" width="1200" height="300">

일단 모델이 추정하는 분포도 $p_{data}$처럼 정규분포라고 가정할 수 있을 것이고 연구진은 이를 실험했을 것이다.

위 그림에 내용은 원본 이미지 a를 일부 지운 b와 오른쪽으로 shift한 c간 MSE를 계산했을 때 a와 b보다 a와 c에서 더 큰 손실이 발생했음을 설명한다.

분명 b가 c보다 더 큰 차이를 갖고 있음에도 원본인 a와 MSE 손실이 상대적으로 더 큰 이유는 Manifold 단원에서 말한대로 의미적 거리가 반영되지 못했기 때문이다.

- manifold 공간에서 b는 이미지의 일부가 지워졌으므로 a와의 거리가 단순 shift만 한 c보다 더 멀어야 하고 이것이 우리가 원하는 결과와 해석적으로 동일하다.
- 반면에 MSE 손실은 단순 픽셀간 차이를 구하는 것이기 때문에 전체공간상 데이터의 거리를 비교한 것과 마찬가지이므로 이상적인 분포를 추정하기에 단순히 정규분포의 평균을 모델이 추정하게 만드는 것은 적절하지 못하다.

### 2-4.Evidence of Lower BOund(ELBO)


<img src="{{site.url}}/images/240820/0011.png" width="1200" height="300">
이러한 문제를 해결하기 위해 이상적인 분포(그림에서 True Posterior로 표기) $p(z|x)$의 하한에 근사하는 분포를 추정한다.

<img src="{{site.url}}/images/240820/0011-1.png" width="1200" height="300">

- q : 우리가 가정하는 분포로 정규분포와 같이 샘플링하기 쉬운 것을 설정.
- $\emptyset$ : 가정한 분포의 평균(혹은 평균과 분산)으로 이를 조절하면서 True Posterior와 유사한 분포를 추정하는 variance inference를 수행.
- True posterior를 알 수 없기 때문에 세번째 항을 제거하게 되고, 그에 따라 부등식으로 바뀌게된다. 즉, 이상적인 분포를 모르더라도 Lower Bound가 posterior와 같거나 하한선일 것이다.

### 2-5.Loss Function

<img src="{{site.url}}/images/240820/0012.png" width="1200" height="300">
ELBO를 기반으로 만들어진 손실함수는 다음과 같은 의미를 갖는다. 첫번째 항은 모델이 추정한 잠재변수가 decoder에 입력되었을 때, 학습 데이터셋의 분포에 속하는 데이터를 만들 확률을 최대화하는 Maximum Likelihood이고 두번째 항은 정규분포인 p(z)와 예측분포인 q(z|x) 간 유사정도를 계산하는데 이는 여러가지 예측분포들 중 이왕이면 정규분포와 비슷한 것을 선택하도록 강제한다.

참고로 decoder로 입력되는 잠재변수는 예측분포에서 샘플링된 값이기 때문에 Backpropagation 과정에서 손실값에 따른 입력값의 미분을 구할 수 없기 때문에 ```Reparameterization Trick```이라는 것을 사용한다.  

>Reparameterization Trick??
>{: .notice--info}

# 3.Generative Adversarial Networks(GAN)

## 3-1.Generator, Discriminator
<img src="{{site.url}}/images/240820/0013.png" width="1200" height="300">
VAE는 variational inference로 Lower Bound $q_\emptyset(z|x)$를 추정하면서 정규분포인 p(z)가 되도록 규제한다. 하지만 GAN은 이러한 분포 추정이 이루어지지 않고 생성된 데이터와 실제 데이터를 판별하고 속이는 과정을 거치면서 생성모델이 학습 데이터셋의 분포를 학습해간다.

<img src="{{site.url}}/images/240820/0018.png" width="1200" height="300">

판별기 모델은 우리가 이미지 분류에서 사용하는 것과 거의 같다.

- 수식을 보면 실제 데이터 x가 입력되었을 때 판별기 D(x)의 확률을 최대로 높혀 log(D(x))를 최대화한다.
- 반대로 생성기가 만든 가짜 데이터 G(x)가 입력으로 주어졌을 때는 확률을 최소화해 log(1-D(G(z)))를 최소화한다.

<img src="{{site.url}}/images/240820/0017.png" width="1200" height="300">

- 생성기는 반대로 판별기가 가짜임을 구분하는 확률을 최소화하려고한다.
- 하지만 이 수식은 학습 초기에는 작은 기울기를, 판별기를 잘 속이는 학습 후반에는 큰 기울기를 갖도록 구성되어 있다.
- 따라서 판별기가 틀릴 확률을 최대화 하는 것으로 수식을 개선한다.


## 3-2.학습 전략

GAN의 학습은 다음과 같이 정리할 수 있다.

<img src="{{site.url}}/images/240820/0014.png" width="1200" height="300">

분포 변화 시각화를 보면 좀 더 명확하다.

<img src="{{site.url}}/images/240820/0015.png" width="1200" height="300">

분명 굉장히 창의적인 아이디어지만 적대적인 관계 때문에 학습이 쉽지 않다. 대표적인 문제가 Mode Collapse인데 생성기는 결국 판별기의 예측결과를 피드백으로 학습하기 때문에 이 과정에서 판별기가 구분하지 못하는 특정 패턴을 발견한 경우 해당 패턴만 계속해서 사용하고 더 이상의 다양성을 고려하지 않게 된다.

엄밀히 말해 틀린 전략은 아니다. 각각의 모델 정의와 손실함수 정의를 보면 생성기는 판별기를 속이기만 하면 목표를 이루는 것이고, 판별기는 생성기를 제한하는 능력이 없기 때문이다. 이러한 문제를 해결하기 위한 전략으로 손실함수를 개선하거나 보다 효과적인 함수로 변경한다.


# 4.Diffusion
Diffusion Model은 확산 모델의 시작으로 이미지에 가우시안 노이즈를 추가하는 Forward diffusion, 노이즈 이미지에서 시작해 원본 이미지로 되돌아가는 Reverse diffsuion 두 단계로 나뉜다. 

이것이 가능한 이유는 물리학적으로 Forward 단계에서 충분히 긴 시간동안 굉장히 작은 노이즈를 조금씩 더했다면 Reverse 단계에서 원본으로 되돌아가기 위해 더해지는 노이즈도 정규분포를 따르기 때문에 다음 위치(다음 단계 이미지)를 추정할 수 있다는 배경을 기반으로 한다.

## 4-1.DPM, DDPM
<img src="{{site.url}}/images/240820/0019.jpg" width="1200" height="300">

DPM과 DDPM은 다음 단계의 이미지는 현재 단계 이미지에만 의존한다는 Markov Chain을 전제로 Forward, Reverse 모두 현재보다 과거의 값에는 영향을 받지 않고 현재 추정한 분포에서 샘플링한 노이즈를 현재의 이미지에 더해 다음 단계 이미지를 만들어낸다는 것도 핵심이다.(이름에 Probablisitc이 있는 이유는 추정한 분포에서 샘플링한 노이즈를 더하기 때문)

### Forward
<img src="{{site.url}}/images/240820/0020.jpg" width="1200" height="300">

이미지로부터 노이즈를 더하는 Forward 과정의 수식을 보면 
- 이미지와 동일한 형상의 가우시안 노이즈를 그냥 더하는 것이 아니라 t단계의 이미지를 $ 1-\beta_t $ 만큼 감쇠한다.
- 노이즈 $ \beta_t $ (스칼라)를 단위행렬에 상수배한 행렬을 감쇠된 이미지에 더하는 식으로 정의되어 있다.
- 그 결과 데이터의 분포 $ q(x_0) $는 가우시안 노이즈가 점점 더해져 최종 단계 $ q(x_T) $는 가우시간 분포(정규분포)에 근사하게 된다.
- 식에서는 노이즈의 변화에 대한 것은 없다. 즉, 노이즈 $ \beta_t $는 사전 정의된 스케쥴링 방식에 따라 스텝 T별로 값이 변경된다.

이렇게 정의한 이유는 변화하는 분포의 분산을 1로 유지하기 위함인데 아마도 분산이 크면 그만큼 변동성이 커지게 되서 학습의 안정성이 낮아져 추정이 어려워지기 때문일 것 같다. 또한 단순히 가우시안 노이즈를 이미지에 더하는 과정의 연속이기 때문에 학습으로 파라미터를 업데이트할 필요가 없다.

<img src="{{site.url}}/images/240820/0021.jpg" width="1200" height="300">

DPM의 다음 버전인 DDPM에서는 forward 수식을 개선해 $ x_t $번째 이미지를 $ x_0 $부터 t단계까지 sequential하게 변환하지 않고 $x_0$에서 $x_t$를 바로 구할 수 있게 했다. 이에 따라 inference에서 reverse 단계인 이미지를 생성할 때 스텝 수를 줄일 수 있게 되고 결과적으로 생성하는 시간을 줄일 수 있게 된다.


### Reverse
<img src="{{site.url}}/images/240820/0022.jpg" width="1200" height="300">

Reverse 단계에서는 가우시안 분포와 근사하는 $ x_T $로부터 원본 $ x_0 $로 복원하는 과정을 수행한다.
- 여기서도 문제는 데이터의 원래 분포를 알기 어렵다는 점.
- 따라서 VAE와 동일하게 Variational Inference로 ELBO인 분포의 평균을 예측.
- 추정한 분포에서 노이즈를 샘플링해서 $ x_t $에 더한 결과가 $ x_{t-1}$.

결과적으로 Diffusion 모델이 학습하는 것은 $ x_T $에서 시작해 데이터의 분포 $ p_{data} $를 생성하기 위한 Maximum Likelihood이다.

### 정리

<img src="{{site.url}}/images/240820/0023.jpg" width="1200" height="300">

### ELBO

참고로 아래 식을 유도하는 과정을 이해하는 것은 굉장히 높은 수학적 지식을 요구하기 때문에 필기한 내용을 따라가는 것을 추천한다.(물론 모델을 그냥 쓰기만 할거면 이해할 필요 없다.)

<img src="{{site.url}}/images/240820/0024.jpg" width="1200" height="300">
DDPM에서는 True Posterior인 $ q(x_{t-1} | x_t) $는 구하기 어려우니 $ q(x_{t-1} | x_t, x_0) $로 이상적인 분포를 정의한다. $ x_0 $를 활용할 수 있는 이유는 forward의 개선된 식에 의해 $ x_0 $에서 $ x_t $를 바로 구할 수 있다는 것이고, 해당 식을 $ x_0 $에 대한 식으로 정리하면 $ x_t $를 이용해 $ x_0 $를 구할 수 있다.  이 과정에서 $ q(x_{t-1} | x_0),  q(x_{t} | x_0) $의 평균과 분산은 모델이 예측하도록 학습하지 않고 $ q(x_{t-1} | x_t, x_0) = N(x_{t-1} l; \tilde \mu(x_t, x_0), \tilde \beta_t) $에 대한  확률밀도함수의 정의에 따라 계산된다.

<img src="{{site.url}}/images/240820/0025.jpg" width="1200" height="300">
결과적으로 DDPM에서 $ q(x_{t-1} | x_t, x_0) $가 의미하는 것은 평균을 예측해 분포를 추정하는 것보다 스텝 t에 어떤 노이즈가 더해졌는지를 예측하는 것으로 문제를 간소화하게 되고 forward에서 개선했던 $ x_0 $에서 $ x_t $를 직접 구할 때 모델이 예측한 노이즈를 적용할 수 있게 된다. DPM은 분포의 평균을 추정하고 노이즈를 샘플링했고, DDPM은 샘플링되는 노이즈 자체를 예측하게 되는 Stochastic한 과정을 수행하므로 Reparameterization Trick이 수식에 반영된다.


### Loss function

<img src="{{site.url}}/images/240820/0026.jpg" width="1200" height="300">

ELBO를 기반으로 정의된 손실함수는 다음과 같이 3개의 항으로 구성된다.
- DDPM에서의 forward process는 $x_T$ 가 항상 gaussian distribution을 따르도록 하기 때문에 사실상 tractable한 distribution이라 $q(x_T∣x_0)$ 는 prior p(x_T) 와 거의 유사하다. 
- 또한, DDPM에서는 forward process variance를 constant로 고정시킨 후 approximate posterior를 정의하기 때문에 이 posterior에는 learnable parameter가 없다.
- 따라서 첫번째 항은 항상 0에 가까운 상수이며, 학습과정에서 무시된다.
- 마지막 항은 두 개의 정규분포간 KL Divergence를 계산하는 것이고, VAE에서 봤던 것처럼 두 개의 정규분포에 대한 KLD는 계산 방법이 정해져 있으므로 상수취급한다.

<img src="{{site.url}}/images/240820/0027.png" width="1200" height="300">

논문에서는 Simplifed Loss라고 해서 $ \epsilon - \epsilon_\theta ( \sqrt{\alpha^\hbar_t}x_0 + \sqrt{1-\alpha^\hbar_t}\epsilon)$로 정의하고 있다.

## 4-2.DDIM

앞서 본 DPM, DDPM은 Markov Chain에 의해 Sequential한 절차를 수행해야만 했고 그에 따라 학습과 추론 속도가 매우 느리다. 

DDIM은 Markov Chain을 제거해 T=1000보다 더 작은 스텝으로 학습, 생성을 가능하게 하는 것이 목적이다.

<img src="{{site.url}}/images/240820/0028.png" width="1200" height="300">
- 식 1, 2, 3은 모두 논문에서 정의한 식으로 따로 유도 과정을 공부하지 않았고, 의미를 이해하고자 했다.
- 식 1에서는 $ q_\simga $라는 확률이 조건으로 $x_0$를 활용하고 있음을 볼 수 있다. 이는 markov chain을 사용하지 않고 forward, reverse를 수행하고자 함을 표상한다.
- 여기서 $ \sigma $는 forward 단계에서 얼마나 stochastic한지를 통제하는 변수로 0에 가까울수록 $ x_{t-1} $이 고정되는, Deterministic하게 된다. 즉, 노이즈가 더해지지 않는다.
- 이 때 식 1은 2에 나온 두 가지 식을 만족하게 되는데 그에 대한 전제 조건은 reverse conditional distribution의 Mean이 식 3과 같아야한다.

<img src="{{site.url}}/images/240820/0029.png" width="1200" height="300">
핵심은 $ x_{t-1} $ 을 예측할 때 $ x_t, x_0$ 가 필요한데 reverse 단계에서는 $x_0$를 모르니 $ x_t $를 활용한다는 것이다. 앞서 DDPM의 개선된 forward는 $ x_t $로부터 $ x_0 $를 구했는데 우리는 더이상 markov chain을 사용하지 않기 때문에 시점 $ t $에 적용되는 노이즈 $ \epsilon_\theta^t $를 예측하는 방식으로 $ x_0 $를 예측한다.

<img src="{{site.url}}/images/240820/0030.png" width="1200" height="300">