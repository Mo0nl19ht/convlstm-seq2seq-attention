# convlstm-seq2seq-attention
미래의 교통상황을 빠르게 예측하기 위해교통 네트워크와 속도를 Heatmap화 시키고 이를 ConvLSTM으로 예측하였습니다.
ConvLSTM의 성능을 높히기 위해 Encoder-Decoder구조와 Attention을 도입하였습니다

### 원본 데이터

미래의 교통파의 변화를 예측하기 위해 하루동안의 내부순환로 속도를 각각 Heatmap화 시켰습니다.
Heatmap에서 교통 혼잡이 있는 부분은 검은색으로 짙게 칠해져서 그 교통 혼잡이 전파되는 양상을 파악할 수 있습니다. 그리고 이미지를 예측하는데 좋은 CNN과 시계열 예측에 좋은 RNN을 합친 ConvLSTM모델을 구축하여 3년간의 내부순환로 Heatmap을 데이터로 넣고 교통 혼잡 예측 모델을 만들어 예측을 하였습니다.
![image](https://user-images.githubusercontent.com/81469045/196760557-d8922f96-e120-4eed-b46f-51dabfc76f51.png)


## 데이터를 Heatmap으로 변환

![image](https://user-images.githubusercontent.com/81469045/196760579-1175f04f-b0fb-4df8-bb0c-cb430742d9a4.png)


## 예측 결과
![image](https://user-images.githubusercontent.com/81469045/196760647-532cf0ac-624a-4d3a-a999-490be7d4fb94.png)

1번 열 : 기존 Heat map

2~4열 : 여러가지 모델이 예측한 결과

3번위 예측이 가장 정확하였다

### 배운 것

- AI에서 가장 중요한 것은 데이터이다
- AI 논문 리서치 능력
- 필요한 데이터를 수집하는 능력
- 필요한 모델을 구현하는 능력
- 데이터를 내가 원하는 형태로 전처리하고 보간하는 능력
- MLflow를 이용한 모델 및 실험관리
- 내가 모르는 것을 잘 질문하는 법

### 역할

- AI 논문 리서치
- AI 모델 구현
- 데이터 수집 및 전처리
- 논문 작성

## 발생문제 및 해결방법

- 처음에는 컬러 heatmap으로 만들고 이를 훈련 및 예측에 사용함 →
**성능이 그다지 좋지 않았음** → 스스로 디버깅하고 데이터를 살펴보면서 교수님과 토론함
→ 굳이 컬러 heatmap으로 만들 필요가 없음을 발견함 → 흑백 heatmap으로 예측을 하니 RMSE성능이 5 오름 (15→10)
    - 이유 : 원본데이터는 x축은 장소 y축은 시간으로 이루어진 속도데이터임 → 이를 heatmap화 시킬 때  아래의 컬러 range사용(잘 보이기 위해서)
        
        ![image](https://user-images.githubusercontent.com/81469045/196760679-43cdd7fe-85d0-49ec-b739-e7d12df4efba.png)
        하지만 색상이 들어간 heatmap은 (r,g,b)의 벡터로 이루어짐 → 즉 1차원의 데이터(속력) 가 3차원의 데이터(r,g,b)로 확장되었기 떄문에 데이터가 더 복잡해져서 예측을 잘 할수 없었던것
        
- **위의 모델을 여러 방면으로 튜닝해도 성능이 10 이하로 떨어지지 않음**
    - 관련 최신논문을 리서치함 → Encoder - Decode구조를 도입함 → 성능 향상 RMSE(10→7)
    → 여기에 attnetion을 도입함 → 성능 향상 RMSE(7~3)
    
- 위의 모델을 튜닝하고 여러가지 데이터로 실험해야하는데 **실험 및 모델관리가 매우 힘듬**
    - Mlops를 공부하여 모델 및 실험 관리 라이브러리 MLflow를 도입 → 실험을 자동화 시키고 evaluate까지 저장하도록함
![image](https://user-images.githubusercontent.com/81469045/196760725-ac72e3ce-b639-495f-a784-26cbf5628de8.png)
![image](https://user-images.githubusercontent.com/81469045/196760764-e758bfd7-859d-4eb2-a779-841fb9ecda7c.png)
