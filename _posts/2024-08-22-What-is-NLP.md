---
layout: single
title: "Natural Language Processing??"
categories: NLP
tag: [Deep-Learning, NLP]
use_math: true
---

자연어 처리는 무엇인가??

# 1.정의

<img src="{{site.url}}/images/240822/0001.png" width="1200" height="500">

자연어는 사람들이 일상생활에서 자연스럽게 사용하는 언어다. 즉, 우리 인간이 사용하는 언어를 자연어라고 한다.

자연어 처리는 컴퓨터가 자연언어의 의미를 분석하고, 이해하여 언어를 생성할 수 있게 만들어주는 기술이다.

# 2.자연어 처리는 어렵다.

<img src="{{site.url}}/images/240822/0002.png" width="1200" height="500">

자연어 처리는 꽤 많이 어려운 작업이다. 왜냐하면 인간의 언어는 여러 가지 특성들을 갖기 때문이다.

대표적인 케이스들에 대해 살펴보면 다음과 같다.

- 다른 단어지만 유사하거나 동일한 의미를 갖는 경우.(개, 강아지와 같은 동의어)
- 다른 표현이지만 유사하거나 동일한 의미를 갖는 경우.(집에 가다 / 귀가하다, 피곤하다 / 지친다)
- 문법적으로 다르지만 같은 의미를 갖는 경우.(수동태와 능동태)
- 동일한 단어지만 문맥에 따라 다른 의미를 갖는 경우.(나는 배를 먹었다. 배를 타고 왔다.)

<img src="{{site.url}}/images/240822/0003.png" width="1200" height="500">

- 중의적인 의미의 문장은 보통 주어진 문장만으로는 정보가 부족해 정확한 의미를 파악하기 어렵다.
- 고유명사처리(```문맥```에 따라 의미가 달라진다.)
- 신조어와 같은 문제들도 추가적으로 존재한다.(사람도 뜻을 모를때가 많은데 컴퓨터는 어떻게 이해해야할지 너무 어렵다.)
- 뿐만 아니라 속담, 사자성어, 숙어처럼 문장을 구성하는 단어들이 보유한 의미와 문장 자체가 의미하는 것이 다른 경우도 존재한다.

## 2-1.한국어는 영어보다 더 어렵다.

<img src="{{site.url}}/images/240822/0004.png" width="1200" height="300">

- 한국어는 ```교착어```로 ```어근```과 ```접사```에 의해 단어의 의미와 기능이 정해지는데 어떤 조사가 붙는가에 따라 문장의 의미가 달라지게 되고 다른 언어에 비해 조합의 수가 더 많다.
- 단어의 순서가 문장의 의미를 결정하는 결정적인 요소는 아니기 때문에 단어 순서가 달라지더라도 문법적인 문제가 없거나 맥락을 이해하는데 문제가 없는 경우도 있다.
- 영어와 다르게 띄어쓰기의 기준이 획일화 되어 있지 않은데, 이에 의해 자연어 전처리인 토큰화 과정이 어렵다.
- "밥 먹었어", "밥 먹었어?"와 같이 평서문과 의문문이 문장부호 하나 외에는 다른 점이 없다. 즉, 문장부호가 나와있지 않았다면 의미가 명확하지 못한 경우도 존재한다.

# 3.언어의 단위들

그렇다면 언어라는 데이터들은 어떤 구성을 가지고 있는지 단위에 대해 먼저 정리해보자.

## 3-1.음절

<img src="{{site.url}}/images/240822/0005.png" width="1200" height="500">

- 음절은 ```언어를 말하고 들을 때, 하나의 덩어리로 여겨지는 가장 작은 말소리의 단위```이다.

- 사과 -> 사, 과 총 2음절
- cat(캣) -> 1음절
- computer(컴퓨터) -> 3음절

## 3-2.형태소

<img src="{{site.url}}/images/240822/0006.png" width="1200" height="500">

- 언어에서 ```의미를 가지는 가장 작은 단위```다. 
- 일반적으로 자연언어 처리에서는 분석의 기본이 되는 토큰으로써 형태소를 많이 이용한다.
- 의미를 갖는 가장 작은 단위이기 때문에 더 쪼개게 되면 의미가 사라지게 된다.

## 3-3.어절

- 한 개 이상의 형태소가 모여 구성된 단위.('뛰-' + '다' = '뛰다') 
- 자연언어는 어절단위로 띄어쓰기 되어 발화 또는 서술된다.(즉, 띄어쓰기가 적용되는 기준은 어절이다.)
- ‘우리는 오늘 동해로 간다’ -> "우리는", "오늘", "동해로", "간다" 4어절
- '김밥은 정말 맛있어.' -> "김밥은', '정말', '맛있어'

## 3-4.품사

<img src="{{site.url}}/images/240822/0009.png" width="1200" height="500">

- 단어를 문법상 의미, 형태, 기능에 따라 분류한 종별(명사, 대명사, 동사 등)을 의미한다.
- 쉽게 말해 문장에서 단어가 어떤 역할을 하는지에 따라 분류된 단어의 종류.
- 의미에 따른 구분 : 명사(Noun), 대명사(Pronoun), 형용사(Adjective), 부사(Adverb), 조사(Postposition), 관형사(Determiner), 감탄사(Interjection), 접속사(Conjunction)
- 이외에도 역할, 형태에 따른 구분이 존재한다.

## 3-5.어간

- 동사나 형용사의 기본형태에서 어미가 추가되지 않은 형태.(뛰-, 칠-, 먹- 등등)
- 자체적인 의미를 가지고 있지만 단독으로 쓰이지 않고 항상 어미와 결합하여 나타낸다.

## 3-6.문맥

문맥은 언어의 단위에는 포함되지 않으나 자주 사용되는 개념 중 하나다.

문맥 또는 맥락은 context라고도 하며, 문장이나 문단이 갖는 전반적인 의미를 말한다.(어떤 글을 한 편 읽었을 때, 이 글이 말하고자하는 것이 무엇이구나로 생각할 수 있겠다.)

문맥이 중요한 이유는 앞서 살펴봤듯 동일한 단어가 문맥에 따라 뜻이 달라질 수 있기 때문이다.

- 언어적 문맥 : 한 단어나 문장이 다른 단어나 문장과 함께 사용되었을 때 그 의미를 결정하는 요소.
- 상황적(물리적) 문맥 : 말이나 글이 사용되는 실제 상황.(걷다가 굉장히 큰 bank를 보았다. 글에는 나타나나 있지 않으나 도시라면 은행일 가능성이 높고, 숲이라면 둑일 가능성이 높다.)

# 4.언어학적 분석

## 4-1.언어의 구성요소

언어는 형태, 내용, 사용 3가지 구성요소가 결합되어 이루어지며 각각에 대해 연구하는 세부적인 분야들이 존재한다.

1. 형태는 실체인 의미를 물리적으로 표현할 수 있는 방법을 말한다.
    - 음운론 : 말소리
    - 형태론 : 형태소, 단어
    - 통사론 : 문장

2. 내용 : 의미론, 단어나 문장이 갖는 실제 의미

3. 사용 : 화용론, 언어를 사용하는 상황.

## 4-2.형태론

<img src="{{site.url}}/images/240822/0010.png" width="1200" height="500">

```형태소(morpheme)```는 언어에서 의미를 갖는 가장 작은 단위이며, 형태론은 형태소를 분석하면서 형태소 간의 상관관계를 규명하는 학문을 말한다.

talks, talker, talked, talking과 같은 단어에서 형태소는 talk와 뒤에 붙는 -s, -er, -ed, -ing와 같은 접사들도 형태소에 해당한다.

자립성을 갖는지에 따라 자립 형태소, 의존 형태소로 나뉜다.
- 자립 형태소 : 홀로 자립하여 쓸 수 있는 형태소.
- 의존 형태소 : 다른 형태소에 의존하여 쓰이는 형태소.(이/가, 맨- 과 같은 조사, 접사들도 형태소에 해당함에 유의)

또한 의미를 가지고 있는가에 따라 실질 형태소, 형식 형태소로 나뉜다.
- 실질 형태소 : 실질적인 의미를 가지고 구체적인 대상이나 동작을 표시하는 형태소.(뛰-, 칠- 은 각각 뛰는것, 칠하는것 이라는 의미를 자체적으로 보유.)
- 형식 형태소 : 시질형태소에 결합하여 말과 말 사이의 관계를 형식적으로 표시하는 형태소.

마지막으로 형태소는 복수형, 과거형과 같은 여러 개의 변이형태를 가질 수 있는데 이를 이형태(allomorph)라고 한다.

## 4-3.통사론

통사론은 단어가 결합하여 구와 문장을 형성하는 규칙과 방법에 대해 연구하는 학문이다.

- "구"는 문장 내에서 하나의 단위로 기능하는 단어들의 집합을 의미한다.
- 단어보다 큰 단위지만 문장보다는 작은 단위로, 특정한 문법적 역할을 수행한다.
- 명사구, 형용사구 등등 -> "큰 집", "내 친구", "책을 읽는다"

### 구조적 모호성

구조적으로는 크게 심층구조와 표층구조로 구분할 수 있다.
- 표층구조 : 실생활에서 사용하는 단어들의 규칙적인 구조.
- 심층구조 : 화자가 문장에 대해 갖는 추상적인 정보(의미)를 담은 구조.

"나는 그녀에게 차였다", "그녀는 나를 찼다"와 같이 의미적으로는 동일하지만(심층구조는 동일하지만) 문장의 구조가 다른(표층구조가 다른) 경우도 있고, 반대로 표층구조는 하나지만 여러 가지 심층구조를 갖는 경우도 존재한다. 즉, 구조적으로 모호한 경우들이 많다.

뿐만 아니라 특정한 규칙을 기반으로 구 구조규칙이라는 것도 존재하지만 이것 역시 변형규칙이 존재하기 때문에 텍스트 데이터를 처리하는 것은 더 어렵다고 볼 수 있다.

### 반복성

언어는 반복이라는 속성을 갖고 있기 때문에 특정 구조를 반복적으로 적용할 수도 있고, 하나의 문장을 다른 문장으로 넣을 수도 있다.

- 나는 어제 꽃을 사려고 했다. -> 나는 어제 맑은 하늘 아래, 깨끗한 공기를 마시며 꽃을 사려고 했다.
- 나는 어제 친구와 밥을 먹었다. -> 엄마는 내가 어제 친구와 밥을 먹었다는 것을 모른다.

## 의미론 

<img src="{{site.url}}/images/240822/0011.png" width="1200" height="500">

말 그대로 단어, 구, 문장의 의미를 연구하는 분야로 이들이 사용될 때 전달되는 일반적인 의미를 다룬다. 즉, 특별한 상황에서 말하는 사람이 의도하는 의미는 제외한다.

- 개념적 의미 : 단어가 사용될 때 전달되는 기본적, 본질적인 의미. 
- 연상적 의미 : 사람에 따라 다른 의미를 떠올린다.

예를 들어 "겨울"이라는 단어는 개념적으로 사계졀 중 하나지만 사람에 따라 추운 날씨, 한파와 같이 부정적인 의미일 수도 있고 하얀 눈이 내리는 즐거운 날일 수도 있다.

또한 일반적인 의미를 연구하는 것이므로 "사람이 하늘을 난다"라는 문법적으로 문제가 없는 문장이 의미적으로는 어색한 문장이다.

### 의미자질

<img src="{{site.url}}/images/240822/0012.png" width="1200" height="500">

Semantic features.

- 단어의 의미를 차별화하기 위한 기본적인 구성요소로 Man, Woman, King, Queen 4개의 단어는 성별로 구분될 수도 있으나 귀족이라는 특성으로도 구분될 수 있다.
- 또한 행위를 하는 주체나 행위를 당하는, 영향을 받는 개체, 날짜, 위치 등과 같은 의미도 함축하고 있다.
- Word Embedding과 같이 단어를 벡터로 표현하는 때에 이러한 의미들을 파악하는 것 같다.
- 동의어(synonymn), 반의어(antonym), 상하관계(hyponymy), 동음이철어(homophones), 동음이의어(homonyms), 다의어(polysemy), 연어(collocation)와 같은 의미적인 관계를 갖는 개념도 포함된다.

## 4-4.화용론

<img src="{{site.url}}/images/240822/0013.png" width="1200" height="500">

화용론은 언어적으로 보이지 않는 의미, 실제로 말하거나 쓰지 않았더라도 화자가 의미하는 바가 무엇인지 연구한다. 즉, 화자와 청자가 어디서, 어떤 때에 나눈 대화의 문맥과 관련해 문장의 의미를 체계적으로 분석하는 것이다.

- 대표적으로 물리적인 문맥(physical context)가 화용론의 주요 내용 중 하나다. 
- 맥락 : 어떤 주어진 언어 표현이 나타나는 부분과 연관 -> 단어, 구, 문장 등의 언어 표현이 사용된 환경이나 상황을 의미
- 고층 건물 숲을 걷다가 bank라는 단어를 보게 되었다면 "둑", "은행" 중에 은행에 해당한다는 것이 맥락을 통해 알 수 있다.