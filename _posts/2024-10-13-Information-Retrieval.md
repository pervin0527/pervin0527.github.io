---
layout: single
title: "AiLab 3기 - Information Retrieval 대회"
categories: Competition
tag: [Deep-Learning, Information-Retrieval, Competition]
use_math: true
---

# 0.Repository

[GitHub - pervin0527/IR](https://github.com/pervin0527/IR)

핵심 포인트

- 이번 대회에서는 생성문에 대한 평가는 이루어지지 않는다.
- 즉, 검색엔진이 입력된 쿼리와 가장 연관성이 높은 문서들을 잘 골라내는 것이 핵심이다.
    
    **대회 뿐만이 아니라 서비스적인 측면에서 봐도 당연히 핵심.**
    

# 1.RAG Pipeline

## 1-1.전체 구조

<img src="{{site.url}}/images/241013/SmartSelect_20241010_171712_Flexcil.jpg" width="1200" height="300">


업스테이지 강의 자료에서 이해한대로 크게 3가지 프로세스로 구분했다.

- 파란색 : 문서 임베딩, 벡터 데이터베이스(스토어)
- 주황색 : 유저가 입력한 쿼리를 처리한다.
- 초록색 : 검색

## 1-2.Query


[IR/src/search/query_processor.py at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/src/search/query_processor.py)

쿼리를 처리하기 위해 LLM을 두 번 호출하게 된다.

1. 멀티턴 대화인가?
    - True → `standalone query`로 정리.(핵심 : 전체 대화를 사용해서 하나의 질의로 만든다.)
    - False → `“응답할 수 없음”`
2. 과학상식에 맞는 대화인가??
    - True → 검색 엔진으로 입력
    - False → `“응답할 수 없음”`

서비스 측면에서 쿼리가 검색에 적절한지 판단하는 것도 중요하겠지만 이번 대회에서는 특히나 더 중요하다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 21.03.45.png" width="1200" height="300">

[intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)를 임베딩 모델로 사용했을 때 쿼리를 처리하는 단계를 추가로 적용했을 때 0.84로 7점 정도 더 개선되었다.


## 1-3.Document


문서처리는 카운팅기반이나 벡터 임베딩 기반 방식 두 가지로 나뉜다. 따라서 둘 중 어느 것이 더 효과적일지 비교하는 과정이 필요했다.

대회 베이스라인 코드에는 ElasticSearch + Nori를 사용했지만 elasticsearch를 사용하지 않을 것이기 때문에 다음 두 가지를 실험했다.

- Kiwi 형태소 분석기와 BM25(Langchain)
- [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) 기반 임베딩

리더보드 점수를 보면 `kiwi 형태소 분석기 + BM25`를 실험했을 때 벡터 임베딩 방식보다는 점수가 낮다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 17.42.15.png" width="1200" height="300">

원인 해석 : 단어의 빈도수 보다 문맥적인 의미를 기반으로 하는 것이 더 효과적이지 않을까??

# 2.Embedding Models

## 2-1.정량평가의 필요성


어떤 모델로 임베딩 하는 것이 좋을지 판단하기 위해서는 평가가 필요하다.

그러나 우리가 가진 쿼리와 문서간에는 매칭 정답이 없기 때문에 검색 결과를 하나하나씩 들여다보거나 리더보드 점수만으로 판단해야한다.

[IR/notebooks/13-embedding_mAP.ipynb at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/notebooks/13-embedding_mAP.ipynb)

이러한 문제를 개선하기 위해 GPT4o를 이용해 문서별로 3개의 질의를 생성하고 mAP를 계산하도록 만들었다.

- 생성된 쿼리는 문서와 동일한 docid를 갖는다.
- 따라서 검색으로 얻은 문서와 쿼리가 같은 docid를 갖는 경우 정답으로 간주한다.

다만 문서를 기반으로 만들어진 쿼리이기 때문에 **mAP가 리더보드보다 더 높게 측정되는 문제**가 있는데 어떻게 하면 좋을지 아직 모르겠다.

## 2-2.다른 사람들의 자료 참고하기

차선책으로 한국어 임베딩 기반 RAG를 연구하는 사람들의 자료를 참고하기 시작했다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 20.37.25.png" width="1200" height="300">

위 표는 RAG로 유명한 유튜버인 테디노트님이 `2024년 7월 기준` 한국어 임베딩 모델 성능을 정리해두신 것이다.

내용상 HuggingFace, Upstage, OpenAI로 정리될 수 있을 것 같아 코드를 간단하게 구현했다.

[IR/src/dense_retriever/model.py at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/src/dense_retriever/model.py)

더 구체적인 성능 비교를 원한다면 [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)팀의 블로그를 참고하자.

[어떤 한국어 임베딩 모델 성능이 가장 좋을까? 직접 벤치마크 해보자.](https://velog.io/@autorag/어떤-한국어-임베딩-모델-성능이-가장-좋을까-직접-벤치마크-해보자)

리더보드 점수로 성능을 비교했을 때 순위는 다음과 같다.(기재되지 않은 모델은 실험하지 않은 것.)

1. [upstage/solar-embedding-1-large](https://python.langchain.com/docs/integrations/text_embedding/upstage/) `mAP : 91.52, MRR : 91.7`
2. [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) `mAP : 84.17, MRR : 84.39`
3. [OpenAI/text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) `mAP : 82.88, MRR : 83.03`

# 3.성능 개선을 위한 시도

## 3-1.Query Expension


검색을 위한 쿼리가 명확해야 좋은 검색이 가능하다.

- 쿼리 : `버스는 무엇인가?`
- 도로 위를 달리는 버스인지, 컴퓨터 공학에서 말하는 버스인지 알 수 없다.
- 따라서 쿼리를 `사람들이 탑승하는 버스`, `시스템 버스` 와 같이 명확하게 재구성하는 과정이 필요하다.

[IR/src/search/query_processor.py at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/src/search/query_processor.py#L119)

이를 위해 GPT4o를 사용해서 검색 엔진에 쿼리를 입력하기 전에 쿼리를 확장하는 처리를 도입하였다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 22.10.25.png" width="1200" height="300">

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 22.20.33.png" width="1200" height="300">

- huggingface의  [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)과 활용했을 때는 약간의 성능 개선이 이루어졌다.
- 하지만 upstage 임베딩 모델과 조합했을 때는 upstage 임베딩 모델만 사용했을 때보다 점수가 낮았다.
    - 이유가 무엇인지….
    - 200개 가량의 쿼리를 사람이 개선해본다면 어떨까??


## 3-2.Document Summarization


[IR/notebooks/10-generate_doc_title.ipynb at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/notebooks/10-generate_doc_title.ipynb)

[IR/notebooks/15-document_summarization.ipynb at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/notebooks/15-document_summarization.ipynb)

문서를 한 줄 요약하여 임베딩을 하도록 하는 방식을 실험했다.

뿐만 아니라 각 문서별로 제목 `ex) 제목 : 글의 핵심, 문서 내용` 을 부여하거나 핵심 단어들을 해시태깅 하는 방법도 실험했다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 22.25.27.png" width="1200" height="300">

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 22.30.18.png" width="1200" height="300">

하지만 요약만을 문서로써 임베딩 하는 방식은 문맥 정보가 손실되기 때문인지 성능이 오히려 크게 감소하는 모습을 보였다.

- 제목이나 요약문을 문서의 초반부에 문장형태로 도입한다면 어떨까??
- 학습한 데이터처럼 문장 형태로 반영해야하는 것이 핵심.

## 3-3.ReRanker


[01. Cross Encoder Reranker](https://wikidocs.net/253836)

쿼리에 따라 문서들이 검색되고 나면 유사도를 기반으로 순위가 결정된다. Reranker는 1차적으로 검색된 모델들을 다시 평가하고 순위를 재조정하는 역할한다.

서비스를 위한 RAG를 설계에서는 1차 검색 때는 가벼운 검색 엔진을 사용해 빠르게 후보들을 선별하고, reranking 단계에서는 복잡하고 계산 비용이 큰 모델(Cross-Encoder)을 사용하여 문서와 쿼리 간의 미세한 상호작용을 고려하게끔 만든다.

[IR/src/dense_retriever/model.py at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/src/dense_retriever/model.py#L32)

[IR/src/search/answer_processor.py at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/src/search/answer_processor.py#L136)

langchain에서도 reranking을 위한 모듈을 구현해뒀기 때문에 huggingface나 다른 모델들과의 활용이 쉽게 이루어질 수 있다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 22.52.33.png" width="1200" height="300">

한가지 아쉬운 점이라면 한국어 기준 reranker 모델은 선택지가 많지 않았고, upstage 모델과 조합했을 때 성능이 mAP : 91보다 높아지지 못했다.

따라서 문서와 쿼리를 영어로 번역하고 영어 임베딩 모델 [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)과 reranker 모델 [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)을 조합으로 사용했을 때 mAP 84.17보다 높은 성능을 보여줬다.


## 3-4.Query Ensemble


쿼리 앙상블은 검색시 쿼리와 문서를 여러 개의 임베딩 모델로 임베딩한 후 **코사인 유사도**를 계산하고, 가중치를 적용하여 최종 유사도를 계산하는 방식이다.

이는 서로 다른 방법으로 쿼리와 문서 간의 관계를 파악할 수 있으며 개별 모델들이 가지는 단점은 보완하고 강점을 합칠 수 있게 된다.

쉽게 말해 단일 모델은 때때로 특정 유형의 노이즈나 편향된 결과를 반환할 수 있으나 여러 모델을 앙상블하여 쿼리를 처리하면, 개별 모델에서 발생할 수 있는 노이즈나 편향된 결과를 평균화하거나 완화할 수 있어 보다 안정적이고 신뢰성 있는 검색 결과를 얻을 수있다.

[IR/src/search/answer_processor.py at main · pervin0527/IR](https://github.com/pervin0527/IR/blob/main/src/search/answer_processor.py#L81)

<img src="{{site.url}}/images/241013/스크린샷 2024-10-10 23.15.23.png" width="1200" height="300">

모델을 여러가지 사용할 수 있고, 각 모델에 대한 가중치를 설정하는 방식이기 때문에 조합할 수 있는 경우의 수가 굉장히 많아 실험 시간이 길어지지만 전반적으로 성능 개선이 이루어지는 것을 확인할 수 있었다.

다만 한가지 고려할 사항은 역시 한국어 임베딩 모델들의 성능과 선택의 폭이 좁다는 점이므로 쿼리와 문서를 영어로 실험하게 되었다.

## 4.한계점 및 개선사항

## 4-1.한계


현재 상황을 정리해보면 어떤 모델을 사용하든, 어떤 방법을 사용하든 [upstage/solar-embedding-1-large](https://python.langchain.com/docs/integrations/text_embedding/upstage/)보다 점수가 높게 나올 수 없는 상황이 이어진다.

더 자세하게 보자면 `huggingface에 공개되어 있는 임베딩 모델들 + 쿼리 앙상블 + 청킹 등` 다양한 조합을 시도해도 mAP가 90을 넘지 못한다. 왜 그럴까?

주관적인 해석이지만  다른 임베딩 모델들은 [upstage/solar-embedding-1-large](https://python.langchain.com/docs/integrations/text_embedding/upstage/)에 비해 한국어에 대한 이해도(또는 실력)가 낮다고 본다.

왜냐하면 정량평가 단계에서는 임베딩 모델의 성능 자체만으로 mAP가 80~91로 좋은 성능을 보여주지만 실제 쿼리 즉, 리더보드 점수는 그에 상응하지 못하기 때문. 

더 쉽게 말하자면 문맥기반 이해도가 낮아 숫자만 바꿔서 똑같은 문제를 내도 틀리는 것과 비슷한 것 같고, 이로 인해 쿼리 임베딩에 대한 검색 역량이 부족한게 아닐까라고 추측한다.

## 4-2.Anthropic : Introducing Contextual Retrieval


[Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

해당 문서는 claude를 제작한 anthropic사에서 작성한 글로 RAG 성능을 높히기 위한 방법들을 소개한다.

### 1.Hybrid Retriever

<img src="{{site.url}}/images/241013/스크린샷 2024-10-12 11.56.53.png" width="1200" height="300">

임베딩 기반 검색은 쿼리 임베딩과 문서(또는 청크) 임베딩간 유사도를 계산하는 것이기 때문에 특정 키워드에 대한 검색 성능은 좋지 못하다고 한다.

(쿼리나 문서가 가진 의미를 수치적으로 표현한 것이기 때문에 키워드 매칭적인 능력은 떨어진다는 것 같다.)

이러한 문제를 개선하고자 BM25를 **함께 사용하면** 어휘 매칭기반의 단어나 구문 일치를 찾아낼 수 있게 되므로 검색시 오류를 줄일 수 있다고 제시한다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-12 18.15.01.png" width="1200" height="300">

예시로는 “Error code TS-999”라는 것을 쿼리로 검색했을 때, 임베딩 기반 검색은 “에러 코드에 대한 검색”을 의미로 갖는 문서를 찾게 되지만 해당 쿼리에서 핵심은 TS-999라는 단어와 관련된 문서를 찾는 것인데, BM25를 함께 사용하게 되면 핵심 키워드 TS-999를 갖는 문서를 검색할 수 있다.

### 2.Contextual Retrieval

chunking은 길이가 긴 문서를 여러 개의 조각으로 나눠서 쿼리와의 관련성이 높은 부분이 더 효과적으로 검색될 수 있게 한다. 

쉽게 말해 문서에서는 쿼리와 연관된 핵심 부분이 있으나 다른 부분들은 노이즈로 적용될 수 있기 때문에 검색 품질이 저하될 수 있기 때문에 chunking을 통해 노이즈를 줄이는 것이 목적이라 볼 수 있다.

다만 여기서 발생하는 문제점은 청크가 충분한 정보를 가지지 못한다는 점이다.

아래 예시를 보면 좀 더 명확하다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-12 15.26.16.png" width="1200" height="300">

- 원본 청크 : "`회사` 수익은 `지난 분기`에 비해 3% 증가했습니다.”
    - “회사”가 정확히 어떤 회사인지 알 수 없다.
    - “지난 분기”가 언제인지 알 수 없다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-12 18.29.52.png" width="1200" height="300">

- anthropic은 **Contextual Retrieval**이라는 것을 소개하는데 이는 다음과 같이 검색 성능을 높히는 전처리 기술이다.
    - "이 청크는 `2023년 2분기` `ACME corp`의 실적에 대한 SEC 제출 자료에서 가져온 것입니다. 전 분기 매출은 3억 1,400만 달러였습니다. 회사의 매출은 전 분기 대비 3% 증가했습니다.”
    - 이를 위해 contextual retrieval은 원본 문서와 청크를 프롬프트에 제공해서 정보가 손실된 청크의 품질을 향상시킨다.
- 다만 문서별로 청크가 굉장히 많이 나오게 되면 그만큼 LLM이 contextual retrieval 과정을 수행해야하기 때문에 비효율적이다.
- 따라서 Claude에서는 **prompt caching**을 제안하며 동일한 문서로부터 파생된 청크들을 처리하는 경우 문서가 caching되어 있기 때문에 chunk에 대한 토큰 비용만 지불하면 된다고 한다.

해당 글에 코드도 첨부되어 있으니 참고하면 좋겠다.

[anthropic-cookbook/skills/contextual-embeddings/guide.ipynb at main · anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb)

<img src="{{site.url}}/images/241013/스크린샷 2024-10-12 18.35.24.png" width="1200" height="300">

결과적으로 BM25와 임베딩 방식의 hybrid retriever를 사용하면서 contextual retriever를 함께 사용한다면 성능 향상이 중첩으로 이루어져 검색 오류가 Embedding + BM25의 5.0%보다 더 낮은 2.9%를 보이고 있다.

## 4-3.검색결과 중복 제거

<img src="{{site.url}}/images/241013/스크린샷 2024-10-14 15.17.32.png" width="1200" height="300">

chunking(`chunk_size = 100`, `chunk_overlap = 50`)으로 설정하고 검색을 하게 되면 하나의 쿼리에 동일한 문서로부터 파생된 청크가 두 개 이상 top3에 포함되는 문제가 있었다.

따라서 이러한 문제를 해결하고자 k개의 검색 결과에서 동일한 `docid`를 가지는 중복 청크들 중 가장 점수가 높은 하나만 선택하도록 옵션을 수정했다.

<img src="{{site.url}}/images/241013/스크린샷 2024-10-14 15.29.43.png" width="1200" height="300">

- 중첩 제거 테스트
    <img src="{{site.url}}/images/241013/스크린샷 2024-10-14 15.18.29.png" width="1200" height="300">

    - upstage 임베딩
    - chunking(chunk_size=100, chunk_overlap=50)
    - query_ensemble
    - 중복문서 제거
    - 점수도 전반적으로 준수하고 top3를 모두 채우는 형태를 보인다.
    
- only-upstage-rm-dup

    <img src="{{site.url}}/images/241013/스크린샷 2024-10-14 15.42.59.png" width="1200" height="300">

    - upstage 임베딩
    - chunking(chunk_size=100, chunk_overlap=50)
    - 중복문서 제거
    - 이상한 점은 유사도 점수가 낮고, top3를 모두 채우지 못하는 경우가 많아졌다.

- UP-CH-ER
    - `documents.jsonl`
    - Chunking(`chunk_size=100`, `chunk_overlap=50`)
    - Ensemble Retriever(Upstage Embedding + BM25)
    - `MAP=0.8788`,`MRR=0.8803`

- UP-CH-ER-CR
    - `documents.jsonl`
    - Ensemble Retriever(Upstage Embedding + BM25)
    - Contextual Retriever
    - `MAP=0.9053`,`MRR=0.9091`

- UP-CH-ER-QE
    - `documents.jsonl`
    - Chunking(`chunk_size=100`, `chunk_overlap=50`)
    - Ensemble Retriever(Upstage Embedding + BM25)
    - Query Embedding
        - `"BAAI/bge-m3”`
        - `"intfloat/multilingual-e5-large”`
        - `"solar-embedding-1-large-query”`
    - `MAP=0.9197`,`MRR=0.9258`

- UP-ER-QE-CR
    - `gpt_contextual_retrieval_documents.jsonl`
    - Ensemble Retriever(Upstage Embedding + BM25)
    - Query Embedding
        - `"BAAI/bge-m3”`
        - `"intfloat/multilingual-e5-large”`
        - `"solar-embedding-1-large-query”`
    - Contextual Retriever
    - `MAP=0.9333`,`MRR=0.9394`