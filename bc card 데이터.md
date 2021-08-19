# BC Card

## ◆ 데이터 분석 방향

---

**_MZ 세대가 가장 많이 소비하는 품목 종류는?_**

_전체 연령대를 기준으로 품목별 매출금액의 평균값보다_

**_MZ 세대를 기준으로 품목별 매출금액의 평균값이 높은 품목은?_**

== 다른 세대에 비해 **주목하고 있는 품목**은 무엇인지

⇒ 그 품목을 구매하는 **MZ 세대의 주가구생애주기와 주성별은 무엇인지**

**_코로나 19이전 2019년에 비해 2020년과 2021년에 MZ 세대가 주로 소비하는 품목은?_**

그 변화에 주목해 볼 것 → **포스트 코로나 시대를 대비해 MZ 세대의 소비패턴의 변화에 주목해볼 것**

## ◆ 데이터명세

---

![1](https://user-images.githubusercontent.com/54494622/128905964-0b439f5e-f0e4-41de-9fb1-568fc91ba768.PNG)

## ◆ 데이터 전처리

---

▩ 데이터 로드

```python
import pandas as pd

bc = pd.read_csv('C:\\Users\\sjjung\\Desktop\\contestData\\bccard.csv')
bc
```

▩ 결측치 확인

```python
#bc.info()
bc.isnull().sum()
```

▩ 라이브러리 및 환경 세팅

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
%matplotlib inline

import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname = 'C:/Windows/Fonts/malgun.ttf').get_name()
mpl.rc('font', family = font_name)

plt.style.use('seaborn') # seaborn 스타일로 변환
plt.rc('font',family='Malgun Gothic')
plt.rc('axes',unicode_minus=False)
pd.set_option('display.max_columns',None)
warnings.filterwarnings(action='ignore') # Warning Message 안 보이게
import pandas.util.testing as tm

import matplotlib as mpl # 고해상도 Plot 을 위해 DPI 조절
mpl.rcParams['figure.dpi']=150
```

▩ heatmap

```python
plt.figure(figsize=(16,9))
sns.heatmap(bc.corr(),cmap='YlGnBu',annot=True,fmt="0.1f")
plt.show()
```

![2](https://user-images.githubusercontent.com/54494622/128905967-9b033ced-0779-4369-80cd-c669a58687d8.PNG)

_⇒ 품목중분류코드와 품목대분류코드가 상관관계를 띄는건 당연한 사실이며, 매출금액과 매출건수가 상관관계를 띄는 거 역시 당연한 사실이기때문에 데이터 전처리의 필요성을 느낄 수 있었다._

**※ 데이터 전처리가 필요해 보이는 필드**

---

- 기준년월 ← 시계열 그래프에서 필요, 그 외의 모델링에서는 제외
- 품목대분류코드,품목중분류코드 or 품목대분류명,품목중분류명 ⇒ 품목대분류명,품목중분류명을 이용하여 더미화하여 수치형 데이터로 변환
- 성별 → 남 0, 여 1
- 연령 → 20대미만,20대,30대,,,60대이상 → 1,2,3,4,5,6 으로 변환
- 가구생애주기 → 더미화
- 고객소재지 광역시도, 시군구, 읍면동을 더미화하거나 수치형데이터로 바꿔줄 경우 유의미한 데이터값이 되지 않으리라 예상, 단 transpose 를 해 describe 로 freq 은 체크해줄 것
- 파생변수 추가 → 매출 평균 == 매출금액/매출건수
- 파생변수 추가 → mz세대 1 , 아닌세대 0

▩ 범주형 변수 분석

```python
bc[bc.columns[bc.dtypes.map(lambda x: x=='object')]].describe().transpose()
```

![3](https://user-images.githubusercontent.com/54494622/128905971-0e06edb9-37da-44f1-aa3d-f6f23089a3b7.PNG)

⇒ 품목대분류명에서 **e상품/서비스**의 빈도가 가장 높은것으로 관측되었다.

⇒ 품목중분류명에서 **o2o 서비스**가 가장 많이 관측된것으로 기록되었다.

▧ 피처 엔지니어링

```python
# 컬럼 재정렬
bc.drop(['고객소재지_광역시도','고객소재지_시군구','고객소재지_읍면동'],axis=1,inplace=True)
bc.drop(['품목대분류코드','품목중분류코드'],axis=1,inplace=True)
bc

# 연령 컬럼 <- 숫자형 데이터 (1,2,3,4,5,6) 으로 변경
bc['연령'].unique() # array(['40대', '20대', '30대', '50대', '60대 이상', '20세 미만'], dtype=object)

def age(x):
    if x == '20세 미만':
        return x.replace(x,'1')
    elif x == '20대':
        return x.replace(x,'2')
    elif x == '30대':
        return x.replace(x,'3')
    elif x == '40대':
        return x.replace(x,'4')
    elif x == '50대':
        return x.replace(x,'5')
    else:
        return x.replace(x,'6')

bc['연령'] = bc['연령'].apply(age)
bc['연령'] = bc['연령'].astype('int64')

# 성별 컬럼 <- 여성 1, 남성 0
bc['성별'] = bc['성별'].apply(lambda x:1 if x=='여성' else 0)

# 파생변수 추가 : 매출평균
pd.options.display.float_format = '{:,.0f}'.format
bc['매출평균'] = bc['매출금액']/bc['매출건수']

# 파생변수2 추가 mz세대 -> 연령(1,2,3) -> 1, 나머지 0
bc['mz세대'] = bc['연령'].apply(lambda x:1 if x in (1,2,3) else 0)
```

![4](https://user-images.githubusercontent.com/54494622/128905973-792ad9e9-da12-4c61-8c6d-0d611e7cb6ab.PNG)

## ◆ 데이터 분석

---

### Q1. **_MZ 세대가 가장 많이 소비하는 품목 종류는?_**

```python
bc_mz[bc_mz.columns[bc_mz.dtypes.map(lambda x: x=='object')]].describe().transpose()
```

![5](https://user-images.githubusercontent.com/54494622/128905975-6a5b67f9-649f-4f69-87b8-1d037429dd76.PNG)

**A1.**

- 품목대분류명에서는 e상품/서비스를 가장 많이 구매했다.
- 품목중분류명에서는 o2o 서비스를 가장 많이 구매했다.
- mz 세대 중 가구생애주기(가구유형)가 가장 많은 것은 신혼영유아가구였다.

_결론 : 전체 데이터의 모양새와 비슷한 분포를 띄고 있다._

### Q2. **_MZ 세대를 기준으로 품목별 매출금액의 평균값이 높은 품목은?_**

<품목대분류명 기준>

```python
bc_mz_big1 = bc_mz.groupby('품목대분류명')['매출금액','매출건수'].sum().reset_index()
bc_mz_big2 = bc_mz.groupby('품목대분류명')['매출평균'].mean().reset_index()

bc_mz_big = pd.merge(bc_mz_big1,bc_mz_big2)

bc_mz_big.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False)
```

![6](https://user-images.githubusercontent.com/54494622/128905977-1e383c16-ad2c-403d-9b63-4fb7fa2c36cb.PNG)

A2-1.

- e상품/서비스의 비중이 압도적이었으며, 많은 사람들이 이용한다는 사실을 알 수 있다.
- 여가/스포츠의 비중도 높다는 사실을 알 수 있다. == **개인을 우선시한다.**
- 의식주 중 식품에 소비하는 것에 많은 비중을 둔다는 사실을 알 수 있다.

**cf) e상품/서비스 및 여가/스포츠를 주로 구매한 mz세대들의 가구생애주기는?**

```python
bc_mz['가구생애주기'][(bc['품목대분류명'] =='e상품/서비스') | (bc['품목대분류명'] == '여가/스포츠')].describe().transpose().reset_index()
```

![7](https://user-images.githubusercontent.com/54494622/128905978-76340be7-478d-43f3-adcc-4145d38a36b7.PNG)

**_⇒ mz 세대의  1인가구의 경우 e상품/서비스 와 여가/스포츠를 주로 구매한다는 사실을 알 수 있다._**

ex) 모바일 상품권 이벤트의 경우 1인 가구 대상 모바일 교환권으로 + 1인이 할 수 있는 스포츠(헬스장 이용권 및 골프연습장 일일이용권)

<품목중분류명 기준>

```python
bc_mz_small1 = bc_mz.groupby('품목중분류명')['매출금액','매출건수'].sum().reset_index()
bc_mz_small2 = bc_mz.groupby('품목중분류명')['매출평균'].mean().reset_index()

bc_mz_small = pd.merge(bc_mz_small1,bc_mz_small2)

bc_mz_small.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False)
```

![8](https://user-images.githubusercontent.com/54494622/128905981-2606e621-f0da-40a9-83db-218335fa1eba.PNG)

A2-2.

- e상품 및 서비스는 주로 **o2o 서비스**를 이용한다**(생활밀착 플랫폼)**
- e머니/상품권에 많은 소비를 한다. (현금보다는 **모바일(온라인) 결제**를 위주로)

⇒ _온라인 플랫폼을 이용하는 MZ 세대의 유저의 경우 e머니 추가 적립이나 상품권 구입 유도를 통해 고객을 유치할 수 있다._

- 대분류에서 **여가/스포츠**에 해당되는 **취미/특기** 가 높은 결과를 보였다.

⇒ _mz 세대의 경우 취미나 특기로 스포츠 활동을 주로 한다고 예측해볼 수 있다. (ex : 골린이, 근손실)_

⇒ _여가/스포츠의 경우 대분류에서 2번째로 높은 수치를 보였으나 기존 여가에 해당되었던 여행의 경우 중분류에서 코로나19의 영향으로 인해 낮은 결과값을 보였다._

**cf) e머니/상품권 및 o2o서비스 그리고 취미/특기 분류명에서 주로 구매한 mz세대들의 가구생애주기는?**

```python
bc_mz['가구생애주기'][(bc['품목중분류명'] =='e머니/상품권') | (bc['품목중분류명'] == 'o2o서비스')
                | (bc['품목중분류명'] == '취미/특기')].describe().transpose().reset_index()
```

![9](https://user-images.githubusercontent.com/54494622/128905946-d2d57300-0fde-45db-ba79-79a9111aaaed.PNG)

**_⇒ 이 역시 1인가구 비중이 가장 큰 것으로 나타났다._**

### Q3. **_코로나 19이전 2019년에 비해 2020년과 2021년에 MZ 세대가 주로 소비하는 품목은?_**

```python
# 코로나19에 따라 2019년도와 2020,2021년도를 분리

mz2019 = bc_mz[(bc_mz['기준년월']==201903) | (bc_mz['기준년월']==201909)]
mz2019.drop(['기준년월'],axis=1,inplace=True)

mz2021 = bc_mz[(bc_mz['기준년월']==202003) | (bc_mz['기준년월']==202009) | (bc_mz['기준년월']==202103)]
mz2021.drop(['기준년월'],axis=1,inplace=True)

```

```python
# 2020/2021년도 매출분석
mz2021_total = mz2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
mz2021_avg = mz2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

mz2021_consume = pd.merge(mz2021_total,mz2021_avg)

mz2021_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(10)

# 2019년도 매출분석
mz2019_total = mz2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
mz2019_avg = mz2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

mz2019_consume = pd.merge(mz2019_total,mz2019_avg)

mz2019_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(10)
```

![12](https://user-images.githubusercontent.com/54494622/128906499-9aaa6549-efbd-4e58-8c92-07598af127e7.jpg)

- **_여행 관련 상품의 경우 코로나 이전에 비해 매출 금액이 급감하였다._**
- **_육아 및 어린이용품 관련 상품의 경우 코로나 19 이후 저출산의 영향을 더욱더 받아 전체 순위 10위권 바깥으로 밀려났다._**
- **_코로나 19 이후로 신선/요리재료, 가공식품, 건강식품과 같이 음식 및 건강에 사람들의 관심사가 높아졌다. ★_**

→ 시계열 그래프 및 막대그래프 그리는 방법 추가적으로 해볼 것

## ◆ 데이터 모델링

---

**※ 분석 방향**

위의 데이터 분석을 통해 **코로나 19 이후** **mz세대**의 경우, 대분류 기준 **식품 및 건강**에 관심이 높아진 것으로 드러났다. 중분류 기준 **신선/요리재료, 가공식품, 건강식품**의 관심사가 높아졌다.

Q. _신선/요리재료, 가공식품, 건강식품에 관심이 많은 mz세대의 경우 어떠한 성별이며, 연령대는 어떻게되며, 가구생애주기(가구원)이 어떻게 되는지 알아보자._

- 정답 라벨 : 품목중분류명
- 훈련 셋 컬럼 : [ 성별, 연령, 가구생애주기, 매출금액, 매출건수, 매출평균 ]
- 머신러닝 모델 : KNN

▣ 1 .데이터 셋

```python
# 위의 가공된 mz2021 데이터프레임 이용
mz2021.drop(['품목대분류명','mz세대'],axis=1,inplace=True)
df = mz2021[mz2021['품목중분류명'].apply(lambda x: x in (['신선/요리재료','가공식품','건강식품']))]
df

```

▣ 라이브러리

```python
# data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from pandas.plotting import parallel_coordinates

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# grid search
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
```

▣ Pairplot

```python
sns.pairplot(df,hue="품목중분류명")
plt.show()
```

![12](https://user-images.githubusercontent.com/54494622/128906388-4c19cdd3-d04c-4099-8c02-df5163e9d8d4.PNG)

- 분류가 쉽지 않아보이지만, 그나마 매출금액, 매출건수를 이용하여 분류가 가능할거 같다.

▣ 모델링

```python
# 명목형 변수 더미화
x = pd.get_dummies(df2.iloc[ : , 0:-1 ])
y = df2.iloc[ : , -1 ]

# 정규화 + 계산
from  sklearn.preprocessing  import MinMaxScaler

x_scaled = MinMaxScaler().fit(x).transform(x)

# 훈련 데이터와 테스트 데이터를 분리
from  sklearn.model_selection  import   train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)
#print(x_train.shape)  # (14570, 8)
#print(x_test.shape)  # (3643, 8)
#print(y_train.shape)  # (14570,)
#print(y_test.shape) # (3643,)

# 모델 생성
from  sklearn.neighbors   import  KNeighborsClassifier

knn_m = KNeighborsClassifier()  # knn 모델생성

grid_params = {
    'n_neighbors' : list(range(1,16,2)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski']
}

gs_m = GridSearchCV(knn_m, grid_params, cv=10)
gs_m.fit(x_train, y_train)
print("Best Parameters : ", gs_m.best_params_) # Best Parameters :  {'metric': 'euclidean', 'n_neighbors': 15, 'weights': 'uniform'}
print("Best Score : ", gs_m.best_score_) # Best Score :  0.527522306108442
print("Best Test Score : ", gs_m.score(x_test, y_test)) # Best Test Score :  0.5182541861103486
```

⇒ 약 50%의 정확도로 분류를 하였다. 정확도를 향상시키기 위해서 파생변수를 추가하든 다른 방향으로 고려해봐야겠다. 추가적으로 분석방향에 대해서도 다시 검토해봐야겠다.

※ 의사결정트리

```python
#4.   모델 생성
from  sklearn.tree  import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)

#5.   모델 훈련
model.fit(x_train, y_train)

#6.   모델 예측
result = model.predict(x_test)

#7.   모델 평가
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
print(model.feature_importances_)

print(x.columns)
```

```
0.5531914893617021
0.553390063134779
[0.0170977  0.04481727 0.05234271 0.09786981 0.76870185 0.00538932
 0.01378135 0.        ]
Index(['성별', '연령', '매출금액', '매출건수', '매출평균', '가구생애주기_1인가구', '가구생애주기_신혼영유아가구',
       '가구생애주기_초중고자녀가구'],
      dtype='object')
```

⇒ 의사결정트리도 약 55%의 정확도를 보이며 비슷한 결과를 도출하였다. 분석의 방향을 다시 한 번 고려해볼 필요가 있겠다.
