---
layout: single
title: "AiLab 3기 - 대화 요약 경진대회"
categories: Competition
tag: [Deep-Learning, Natural-Language-Processing, Competition]
use_math: true
---

Upsatage x FastCampus AiLab 3기 - 문서 이미지 분류 경진대회

# 1.대회소개

본 대회는 upstage와 fastcampus가 주관하는 AiLab 3기에서 진행된 Dialogue Summarization의 개인 레포트입니다.

<img src="{{site.url}}/images/240911/0000.png" width="1200" height="300">

팀은 저를 포함해 총 4명으로 구성되어 있으며, public 7위, private 4위로 마무리하였습니다.

# 2.데이터 개요

- [DialogueSum](https://github.com/cylnlp/dialogsum)을 한국어로 번역한 데이터입니다.
- train, dev, test로 총 3개의 csv 파일이 제공됩니다.
- train : 12457, dev : 499, test : 499
- train과 dev는 [fname, dialogue, summary, topic]으로, test는 [fname, dialogue]로 구성됩니다.

<img src="{{site.url}}/images/240911/0001.png" width="1200" height="300">

train, dev 데이터는 위와 같이 두 명이상의 사람들이 대화한 내용(구어체)이 있었고, 그에 대한 요약문(문어체)이 제공됩니다.

반면 test 데이터는 dialogue가 대화체이기 때문에 학습 성능과 리더보드 점수간 불일치가 발생했습니다.

# 3.베이스라인 구축

## 3-1.데이터 전처리

학습 데이터는 다음과 같은 전처리 과정이 적용되었습니다.

<img src="{{site.url}}/images/240911/0005.png" width="1200" height="300">

1. 화자 #Personi#의 턴(turn)이 종료됨을 모델이 이해할 수 있게 하기 위해 ```<sep>``` 토큰을 추가합니다.
2. 'ㅋㅋ'와 같은 자음으로만 된 글자를 포함하는 경우 이를 제거합니다.
3. ```#문자열#```과 같은 특수토큰들이 존재하는데 잘못 표기된 경우는 적절하게 변경합니다.
4. '아아아', '스스스'와 같이 반복되는 글자들이 포함되는 경우 '아', '스'와 같이 한글자만 남겨놓도록 처리합니다.

<img src="{{site.url}}/images/240911/0002.png" width="1200" height="300">

모델 학습시 ```encoder_max_len```과 ```decoder_max_len```, 추론시 ```generate_max_len```을 설정해야합니다. 따라서 train, dev, test의 dialogue, summary에 대한 길이를 구해 시각화하고 가장 비중이 가장 많은 값으로 설정합니다.


## 3-2.모델 학습

학습 모델은 대회 Baseline 코드에 있는 Bart를 사용했고, 추가적으로 T5도 사용해봤습니다.

베이스라인 코드는 final result로 41.66, 전처리를 추가했을 때 41.67이라는 점수를 받았습니다.

# 4.문제점

- huggingface에 있는 pretrained weight를 사용했었는데, 한국어로 사전학습된 것들이 많지 않아서 선택지가 영어보다 작습니다.
- 자연어처리 작업에 대한 이해도가 부족해서 Tokenizer를 대회 데이터로 생성해놓고 pretrained weight를 사용하는 실수를 했는데, 이 작업에서 시간 소비를 굉장히 많이했습니다.
- 토크나이저를 새로 만들게 되면 Pretrained weight를 사용하지 못하고 새롭게 pretraining을 해야합니다.
- BartForConditionalGeneration은 [CrossEntropy 손실을 사용합니다.](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/bart/modeling_bart.py#L1967) 하지만 이는 단어 단위 손실을 계산하기 때문에 문장 전체에 대한 피드백이 부족해서 성능개선이 어렵다고 해석했습니다.

# 5.시도한 방법들

## 5-1.R3F

<img src="{{site.url}}/images/240911/0009.png" width="1200" height="300">

- input에 noise를 적용하고 얻은 추론과 noise를 적용하지 않은 추론간 KL-Divergence를 loss로 사용하는 방식입니다.
- 두 예측 분포간 발산을 계산하므로 전체적인 문장의 차이를 손실로써 반영할 수 있을 것이라 생각했습니다.
- 리더보드 점수는 39로 오히려 감소했는데, noise가 없는 input의 분포가 어느정도 좋은 성능을 보여야 의미가 있을 수 있는 방법이라 해석하고 있습니다.

## 5-2.Rouge loss와 Reinforcement Learning

- 대회 평가 지표인 Rouge score를 손실함수로 사용할 수는 없을까라는 생각이 있었습니다.
- Rouge-1은 단어, Rouge-2는 bigram이다 보니 rouge를 손실로 반영한다면 어느정도 문장 관점 손실을 계산할 수 있을 것이라 생각하고 실제로 학습을 시도했습니다.
- 학습 로그 자체는 준수했지만, 점수 향상은 이루어지지 않았고, 조사 끝에 rouge는 미분이 불가능하며 보통은 rouge를 reward로 해서 강화학습을 하는게 일반적이라고 합니다.
- 따라서 해당 방식을 시도해봤는데 학습 속도가 굉장히 느리면서 수렴도 이루어지지 않아서 기각했습니다.


## 5-3.영어 데이터 활용하기

<img src="{{site.url}}/images/240911/0011.png" width="1200" height="300">

- 한국어 자체적으로 갖는 여러 특성들로 인해 성능의 한계가 있을 것이라 생각되었으며, Back Translation 데이터를 확보할 수 있고, Pretrained weight 선택폭도 넓어지니 일석이조입니다.
- 대회 규정상 오리지널 데이터는 사용이 불가능합니다. 따라서 SoLA, Gpt4 API를 이용해 train, dev를 영어로 다시 번역했습니다.
- 영어로 학습시 더 빠르게 수렴되고 rogue score가 증가하는 양상을 보였습니다.
- 테스트 데이터에 대한 예측을 만들고, LLM 모델로 다시 한국어로 번역해 결과를 제출합니다.
- 하지만 리더보드 점수는 향상되지 못했습니다. 그 이유는 번역된 단어가 동의어 또는 유의어로 생성되서 정답 단어와 다르기 때문입니다. 즉, 요약문의 퀄리티는 꽤 괜찮았지만 리더보드 점수 향상으로 연결되지 못했습니다.

## 5-4.LLama3 + LoRA

[https://llama.meta.com/docs/how-to-guides/fine-tuning/](https://llama.meta.com/docs/how-to-guides/fine-tuning/)

Bart, T5 모델들은 이런저런 시도들을 하더라도 점수가 향상되지 못했기 때문에 LLM 모델을 사용해보기로 했습니다.

모델은 LLama3-8B를 사용했는데, 전체를 fine tuning하기엔 컴퓨팅 자원이 턱없이 모자랐기 때문에 LoRA를 채택해서 소량의 파라미터만 학습했습니다.

결과적으로 리더보드 점수가 43점을 달성할 수 있었습니다.

## 5-5.Back Translation + Data Generation

<img src="{{site.url}}/images/240911/0012.png" width="1200" height="300">

멘토링 시간에 받은 피드백은 train과 test간 어체의 불일치였습니다. 따라서 멘토님은 다음과 같은 피드백을 주셨습니다.
- 이전에 만든 영어 데이터를 한국어로 번역할 때 문어체, 구어체, 번역체를 반영할 것.
- 기존 한국어 데이터에서 topic을 기준으로 가장 많은 비중을 차지하는 것들을 선별하고 그에 대한 샘플들을 뽑아 fewshot으로 제공.
- 이를 기반으로 dialogue를 입력해 summary를 생성 또는 summary를 입력해 dialogue를 생성하면서 다양한 어체를 반영하도록 프롬프트 엔지니어링.

데이터는 SoLA보다 Gpt4가 더 좋은 퀄리티로 생성할 수 있었습니다. 다만 멘토링을 받은 시점이 월요일 밤이었기 때문에 사실상 학습할 수 있는 기간은 화요일 하루 뿐이었고, 7시 이전까지 열심히 학습했지만 데이터 양이 증가했기도 하고 기본적인 모델 학습 시간도 길었기 때문에 3epoch 정도만 학습했습니다.

결과적으로 점수 향상은 이루어지지 못했으나, 길게 학습했다면 유의미한 성과를 보였을 것 같습니다.

# 6.후기

일단 경험이 굉장히 중요하다는 점을 다시 한 번 느낄 수 있었습니다.

자연어처리에 대한 경험 및 이해도가 좀 있었다면 '토크나이징 + 사전학습'에 시간을 많이 허비하지 않았을 것이고, 그만큼 LLM 실험을 더 할 수 있었을 것 같습니다.

그래도 그만큼 경험치를 많이 쌓을 수 있었다고 생각하며, 성적도 7위에서 4위로 올라갔기 때문에 나름 유의미한 삽질이었다고 생각합니다.

- huggingface에서 pretrained weight를 사용해야하는데 이것 때문에 모델의 구조적인 변화를 주기에 굉장히 까다로웠다.
- 토크나이저를 새로 만들 때는 대규모 데이터셋을 이용한 Pretrain이 반드시 필요하다....
    - 연구할 때는 이 과정이 거의 필수이지 않을까??
    - 기업에서 서비스를 만들 때는 너무 많은 비용이 들어갈 것 같으므로 pretrained weight를 많이 사용할 것 같다.
- LLM 모델을 활용하는 방법에 대한 공부가 필요하겠다.(솔직히 튜토리얼 없었으면 LoRA라는게 있는지도 몰랐다.)