---
layout: single
title: "Data Centric Ai"
categories: Data-Centric
tag: [Data-Centric]
use_math: true
---

데이터 중심 인공지능 개발

# 1.데이터의 중요성

<img src="{{site.url}}/images/240925/0000.png" width="1200" height="300">

Data Centric Ai는 이름 그대로 데이터 중심의 ai개발을 뜻한다. 이전까지의 개발에서는 벤치마크 데이터로 고정시켜두고 모델의 구조 및 학습 알고리즘에 변화를 중점적으로 연구했으나 이제는 준수한 성능의 모델로 고정을 시켜두고 데이터에 집중하자는 것이다.

이러한 변화가 일어난 이유는 연구에 머물지 않고 서비스에 활용되기 위해서는 벤치마크 데이터에서 벗어나 기업의 서비스에 맞는 데이터에서도 높은 성능을 보여야하며 벤치마크상에서의 모델 성능은 어느정도 포화되었기 때문이라 추측한다.

또한 데이터는 곧 모델을 학습하는데 필요한 재료이기 때문에 양질의 재료 확보와 손질이 필요하다.

# 2.서비스 개발 과정

<img src="{{site.url}}/images/240925/0001.png" width="1200" height="300">

모델을 서비스에 도입하는 배포 단계 전까지는 다음과 같은 과정들이 수행된다.

- 베이스라인 코드 구축
- 모델의 성능 파악
- 데이터 수집 및 전처리 과정
- 이러한 것들이 모두 수행되기 때문에 모델 중심, 데이터 중심이 5대5 비율이다.

<img src="{{site.url}}/images/240925/0002.png" width="1200" height="300">

모델을 배포한다는 것은 어느정도 요구사항들이 모두 충족되었을 것을 의미한다.

이후에는 서비스가 시작되고 발생하는 문제점 즉, 모델이 잘못된 예측을 반환하게 되는 Edge case들을 수집하고 분석하여 다시 학습에 반영하는 단계다.

따라서 모델의 변화보다는 데이터를 중심으로 개발을 진행하게 된다.

참고로 이러한 과정은 결국 모델을 안정적이고 효율적으로 배포하며 유지 관리하는 것을 목표로 하는 MLOps와 거의 같다.

# 3.데이터셋 구축 과정

<img src="{{site.url}}/images/240925/0003.png" width="1200" height="300">

모델 학습을 위한 데이터셋 구축은 위와 같이 6개 과정으로 이루어진다.

1. 직접 수집, 크롤링, 크라우드 소싱 등의 방법으로 데이터를 수집한다.(원시 데이터, raw data)
2. 이상치, 결측치와 같은 부분들을 제거해 데이터셋의 품질을 높힌다.(원천 데이터, source data)
3. 데이터별로 라벨을 부여한다.(라벨링 데이터, labeled data)
4. 라벨링된 데이터의 품질을 검수하고 기준 이하의 데이터를 정제하는 최종 전처리 단계.
5. 데이터셋 릴리즈

# 4.주의사항

## 4-1.저작권

직접 수집외에 오픈소스 데이터, 크롤링, 크라우드 소싱 같은 방법으로 데이터를 수집하게 되는 경우 저작권에 주의해야한다.

<img src="{{site.url}}/images/240925/0004.png" width="1200" height="300">

여러 종류의 저작권이 있지만 간단하게 인지할 것은 ```CCL(Creative Commons, License)```이다. 이는 자유이용을 허락하는 표시를 의미한다.

<img src="{{site.url}}/images/240925/0005.png" width="1200" height="300">

## 4-2.개인정보

<img src="{{site.url}}/images/240925/0006.png" width="1200" height="300">

- 새로운 서비스에 가입할 때 개인정보 수집 및 이용 동의서에 동의를 하도록 요구하는데 사용자 입장에서도 해당 내용을 충분히 인지하고 동의를 하는 것이 좋다.
- 이름, 주민등록번호, 거주지 등의 개인정보는 개인정보 보호법령에 의해 보호 받기 때문에 이를 학습에 반영하지 못하게 해야한다.

<img src="{{site.url}}/images/240925/0007.png" width="1200" height="300">

즉, 기업에서는 회원 및 학습 데이터에 명시되어 있는 개인정보에서 개인을 식별할 수 있는 요소를 제거함으로써 특정 개인을 알아볼 수 없게 조치해야한다.

## 4-3.윤리

<img src="{{site.url}}/images/240925/0008.png" width="1200" height="300">

LLM이 발전함에 따라 사용자의 다양한 질문이 데이터로 입력되게 되는데 그에 따른 답변이 문제가 되는 경우들이 많다.

이러한 이슈를 사전에 방지하기 위해 과학기술정보통신부와 정보통신정책연구원에서는 인공지능 윤리 기준을 발표했다.

간단하게 정리해보면 다음과 같다.

- 개인정보, 저작권을 보호해야하며 비속어나 폭력적, 선정적인 내용을 포함하지 않아야 한다.
- 잘못된 예측이나 답변을 사용자에게 제공함으로써 발생하는 비인권, 성차별 등의 문제가 발생하지 않아야함.
- "폭탄을 어떻게 만드나요", "해킹하는 방법을 알려주세요"등의 잠재적인 위협을 일으킬 수 없도록 조치해야함.
- 데이터의 분포가 어느 한 쪽으로 치우쳐진 편향이 되지 않도록 구성해 인종 차별이나 사회적인 이슈를 발생시키지 않아야함.

# 5.데이터 클렌징

라벨링 에러는 굉장히 다양한 이유로 발생할 수 있다.

- 라벨링 가이드라인이나 교육 과정에서 잘못 설명한 경우
- 라벨링 규칙이 중의적이라 작업자마다 서로 다르게 해석될 수 있는 경우
- 라벨링 도중에 규칙이 변경되는 경우
- 라벨링 규칙이 다양한 예외들을 처리하지 못하는 경우
- 라벨링 규칙과 관련된 질의 응답에서 일관성이 없는 경우

따라서 라벨링을 할 때는 모든 작업자들이 제대로 이해하고 라벨링을 할 수 있을만큼 규칙이 일관적이어야 한다.

어떤 이유로든 라벨링 에러가 발생하는 경우 다음과 같은 방법들을 통해 검수할 수 있다.
1. 샘플링 후 직접 검수
    - 가장 쉽게 할 수 있는 방법으로 라벨링 결과를 사람이 직접 검수.
    - 데이터 규모가 너무 큰 경우 전부를 검사할 수 없으니 샘플링을 하고 검사한다.
2. 모델 결과 분석
    - 정량평가 또는 정성평가를 통해 성능이 낮은 부분을 찾아 해당 부분을 중심으로 직접 검수.
    - 성능이 낮은 원인이 문제의 난이도가 높은지, 모델 자체의 문제인건지, 라벨링 에러인지 판단.
3. IAA 분석
    - Inter-Annotator Agreement
    - 동일한 라벨링 작업에 참여한 작업자들의 일치정도를 의미함.
    - 일치도가 낮은 작업자 또는 일치도가 낮은 라벨을 찾아낼 수 있음.
4. Confident Learning

# 6.IAA

동일한 작업에 할당된 작업자들 간의 일치 정도를 나타내며 범주형 또는 명목형 데이터에 주로 활용한다.

- IAA가 높다는 것은 라벨링이 일관성 있게 되었다는 것을 의미한다.
- 즉, 데이터에 노이즈가 적다는 것을 의미하기 때문에 모델 성능에 긍정적인 영향을 줄 수 있다.
- 사람이 직접 평가하는 정성적, 휴리스틱한 데이터 품질 평가 방법은 많으나 IAA처럼 정량적인 데이터 품질 평가 방법은 많지 않다. 따라서 IAA를 사용하면 자체적으로 데이터의 품질을 정량적으로 평가할 수 있다.

단점이라면 IAA는 일치하는 정도를 평가하는 것이기 때문에 IAA가 높다고 해서 라벨링의 정확도가 높다는 것은 아니다.

즉, 모든 작업자가 틀리게 라벨링을 했더라도 모두 일치했다면 IAA는 높아질 수 있다는 것에 주의해야 한다.

- Cojen's Kappa
- Fleiss Kappa
- F1 Score

# 7.Active Learning

데이터 샘플링 방법의 일종으로 모델 학습에 가장 유익한 데이터 지점을 점진적으로 선택하는 방법으로 모델에 유익한 데이터를 선택적으로 파악해 라벨링할 수 있기 때문에 더 작은 라벨 데이터로도 목표 성능에 도달할 수 있게 된다.

1. 라벨이 지정되지 않은 데이터 세트 준비.(적은 양의 라벨이 있는 데이터로 기본적인 모델을 학습)
2. 모델이 학습을 완료한 후, 라벨이 없는 데이터 풀에서 어떤 데이터를 라벨링할지 선택.
    - 불확실성 샘플링(Uncertainty Sampling): 모델이 가장 자신 없어 하는 데이터를 선택.
    - 가장 큰 마진(Largest Margin): 두 개의 가장 가능성 높은 클래스 간 차이가 작은 데이터를 선택.
    - 엔트로피 기반 샘플링(Entropy Sampling): 모델의 예측 확률 분포에서 가장 혼란스러워하는 데이터를 선택.
3. 라벨러에게 해당 데이터를 할당해 라벨링
4. 라벨링된 데이터로 예측 모델의 성능 평가.
5. 성능 기준을 만족할 때까지 2~4 과정 반복.

액티브 러닝을 활용하면 목표 성능에 도달하거나 성능을 향상시키는데 필요한 데이터 양을 줄여주는 효과가 있다. 즉, 데이터 수집 예산 및 라벨링 소요 시간에 관계 없이 리소스를 최대한 활용할 수 있게 해준다.

다만 초기 학습을 위해 라벨링을 해야하며, 모델이 반드시 필요하고 불확실성을 올바르게 추정하지 못하는 경우 라벨링에 유익하지 않은 데이터들을 선택할 수 있다.