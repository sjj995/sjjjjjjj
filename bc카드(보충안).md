# BC Card

## ▣ 데이터 분석 방향(보충)

---

- **_MZ세대를 제외한 다른세대의 코로나 전과 후의 구매패턴 차이점_**

⇒ MZ세대의 경우 **출산/육아 관련 상품** 과 **여행** 관련 상품이 코로나 전 전체 항목 중 10위 안에 랭크했으나, 코로나 이후 **여행**의 경우 **5위 → 9위**, **출산/육아관련 상품**의 경우 **10위 바깥**으로 밀려났다. 또한, **식품관련** ( **신선/요리재료, 가공식품, 건강식품** ) 의 경우 (특히 **가공식품**) 의 경우 많은 상승폭을 보였다 ( cf: **가공식품** ⇒ **밀키트 매출 상승 반영** )

이러한 특징이 MZ세대만의 특징인지 아닌지를 분석하기 위해서 **다른 세대도 데이터 분석**

- **_신선요리제품, 가공식품 (이상 식품) 과 건강식품 ( 이상 건강) 을 주로 이용하는 MZ세대의 가구생애주기 전체 비중 확인 ( 원형 그래프 )_**

⇒ 가공식품의 판매 매출 건수와 매출금액이 상이하게 증가하였다면, 가공식품을 주로 이용하는 MZ세대의 가구생애주기는 무엇일지 알아볼 것, 이와 마찬가지로 건강식품, 신선요리 제품도 같이 알아보기

- **_코로나 전과 후를 비교해서 MZ세대의 가구생애주기 비중이 달라진 점_**

⇒ 왜 육아용품,어린이용품 서비스 구매가 하위로 떨어지고 식품관련 항목이 올라갔는지 그 근거에 대한 대립가설을 세우기 위해서

## ▣ 데이터 전처리

---

```python
# 데이터 로드
import pandas as pd

bc = pd.read_csv('C:\\Users\\sjjung\\Desktop\\contestData\\bccard.csv')
bc
```

```python
# 라이브러리
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

```python
# 컬럼 재정렬
bc.drop(['고객소재지_광역시도','고객소재지_시군구','고객소재지_읍면동'],axis=1,inplace=True)
bc.drop(['품목대분류코드','품목중분류코드'],axis=1,inplace=True)
bc
```

```python
# 연령 컬럼 1,2,3,4,5,6
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
```

```python
# 성별 컬럼 여 1, 남 0
bc['성별'] = bc['성별'].apply(lambda x:1 if x=='여성' else 0)
```

```python
# 파생변수 추가 매출평균
pd.options.display.float_format = '{:,.0f}'.format
bc['매출평균'] = bc['매출금액']/bc['매출건수']
```

```python
# 파생변수2 추가 mz세대 -> 연령(1,2,3) -> 1, 나머지 0
bc['mz세대'] = bc['연령'].apply(lambda x:1 if x in (1,2,3) else 0)
```

## ▣ 데이터 분석

---

### ▩ **_MZ세대를 제외한 다른세대의 코로나 전과 후의 구매패턴 차이점_**

⇒ MZ세대의 경우 **출산/육아 관련 상품** 과 **여행** 관련 상품이 코로나 전 전체 항목 중 10위 안에 랭크했으나, 코로나 이후 **여행**의 경우 **5위 → 9위**, **출산/육아관련 상품**의 경우 **10위 바깥**으로 밀려났다. 또한, **식품관련** ( **신선/요리재료, 가공식품, 건강식품** ) 의 경우 (특히 **가공식품**) 의 경우 많은 상승폭을 보였다 ( cf: **가공식품** ⇒ **밀키트 매출 상승 반영** )

이러한 특징이 MZ세대만의 특징인지 아닌지를 분석하기 위해서 **다른 세대도 데이터 분석**

```python
# MZ 아닌 다른세대 데이터프레임 구성
bc_ad = bc[bc['mz세대']==0]
bc_ad

# 2019년도 기성세대 데이터프레임 구성
ad2019 = bc_ad[(bc_ad['기준년월']==201903) | (bc_ad['기준년월']==201909)]
ad2019.drop(['기준년월'],axis=1,inplace=True)

# 2020년,2021년도 기성세대 데이터프레임 구성
ad2021 = bc_ad[(bc_ad['기준년월']==202003) | (bc_ad['기준년월']==202009) | (bc_ad['기준년월']==202103)]
ad2021.drop(['기준년월'],axis=1,inplace=True)

#2020년,2021년도 기성세대 매출금액, 매출건수, 매출평균 컬럼 구하기
ad2021_total = ad2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
ad2021_avg = ad2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

ad2021_consume = pd.merge(ad2021_total,ad2021_avg)

# 기성세대 2020년,2021년도 매출건수, 매출금액, 매출평균별 정렬 상위 Top 15 항목 (순위 포함)
temp_ad_2021 = ad2021_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)
temp_ad_2021['매출순위'] = temp_ad_2021['매출건수'].rank(method='min',ascending=False)
temp_ad_2021

# 기성세대 2019년도 매출건수, 매출금액, 매출평균별 정렬 상위 Top 15 항목 (순위 포함)
ad2019_total = ad2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
ad2019_avg = ad2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

ad2019_consume = pd.merge(ad2019_total,ad2019_avg)

temp_ad_2019 = ad2019_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

temp_ad_2019['매출순위'] = temp_ad_2019['매출건수'].rank(method='min',ascending=False)
temp_ad_2019
```

```python
# 기성세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
cov_ad = pd.merge(temp_ad_2019,temp_ad_2021,how='inner',on='품목중분류명')
cov_ad.drop(['품목대분류명_y'],axis=1,inplace=True)
cov_ad.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','매출평균前','순위前','매출금액後','매출건수後','매출평균後','순위後']

cov_ad['변동순위'] = cov_ad['순위前'] - cov_ad['순위後']
cov_ad
```

![기성세대자료](https://user-images.githubusercontent.com/54494622/130101153-a95c60b7-e82e-4902-90d4-cec46e7dac2b.png)

⇒ 기성세대의 경우 코로나 이전과 코로나 이후 매출건수를 기준으로 순위를 나열했을 경우, 순위의 변동폭이 **최대 2순위**였다. ( = 큰 변화를 나타내지않음 )

⇒ 2순위 변동을 나타낸 항목의 경우 **여가/스포츠** 의 **취미/특기** 의 경우 (**4위→6위**), **가공식품**의 경우 (**6위 → 4위**), **여행**의 경우 (**9위 → 11위**) 의 변화를 보였다.

```python
# mz세대 2019년도 매출건수, 매출금액, 매출평균별 정렬 상위 Top 15 항목 (순위 포함)
bc_mz = bc[bc['mz세대']==1]
mz2019 = bc_mz[(bc_mz['기준년월']==201903) | (bc_mz['기준년월']==201909)]
mz2019.drop(['기준년월'],axis=1,inplace=True)
mz2019_total = mz2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
mz2019_avg = mz2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

mz2019_consume = pd.merge(mz2019_total,mz2019_avg)

temp_mz_2019 = mz2019_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

temp_mz_2019['매출순위'] = temp_mz_2019['매출건수'].rank(method='min',ascending=False)
temp_mz_2019

# mz세대 2020년도,2021년도 매출건수, 매출금액, 매출평균별 정렬 상위 Top 15 항목 (순위 포함)
mz2021 = bc_mz[(bc_mz['기준년월']==202003) | (bc_mz['기준년월']==202009) | (bc_mz['기준년월']==202103)]
mz2021.drop(['기준년월'],axis=1,inplace=True)

mz2021_total = mz2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
mz2021_avg = mz2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

mz2021_consume = pd.merge(mz2021_total,mz2021_avg)
temp_mz_2021 = mz2021_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

temp_mz_2021['매출순위'] = temp_mz_2021['매출건수'].rank(method='min',ascending=False)
temp_mz_2021
```

```python
# mz세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
cov_mz = pd.merge(temp_mz_2019,temp_mz_2021,how='inner',on='품목중분류명')
cov_mz.drop(['품목대분류명_y'],axis=1,inplace=True)
cov_mz.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','매출평균前','순위前','매출금액後','매출건수後','매출평균後','순위後']

cov_mz['변동순위'] = cov_mz['순위前'] - cov_mz['순위後']
cov_mz
```

![mz세대특징](https://user-images.githubusercontent.com/54494622/130101143-3c337d83-d121-4c1a-95ab-9a5d22b529a3.jpg)

_⇒ MZ 세대의 경우 다른 세대에 비해 변화의 폭이 상대적으로 컸다._

1. _여행의 경우 (5위 → 9위) 4계단 하락하였다._

→ MZ 세대의 경우 해외여행을 가는 경우가 많았을텐데 코로나 이후로 많은 영향을 받은것으로 추측된다.

_2. 가공식품의 경우 (11위 → 7위) 4계단 상승하였다_

→ 다른세대의 경우도 가공식품이 2계단 상승하였다. 하지만 MZ세대의 경우 4계단 상승하며 매출이 큰 폭으로 증가하였다

**가설 1 X : MZ세대의 가구생애주기의 영향이 있었기 때문이다. → Y : 가공식품의 매출이 증가하였다**

3.  육아용품서비스 및 어린용품서비스 관련 상품의 경우 (11위 → 8위, 12위 → 9위) 3계단 하락하였다.

→ 코로나19 이후 MZ세대의 가구생애주기에 변화가 생겼을 것이다

**가설 2 X : 코로나19 이후 MZ세대의 가구생애주기에 변화가 생겨서 영향을 끼쳤다 → Y : 출산/육아 용품 매출이 줄었다.**

1. 신선 요리재료, 건강식품의 경우도 2계단씩 상승하였다.

→ 코로나 19이후 모든 세대에서 건강과 건강에 밀접한 먹거리에 관심을 갖고 있다.

### ▣ 가설 1 검증

---

**X : MZ세대의 가구생애주기의 영향이 있었기 때문이다. → Y : 가공식품의 매출이 증가하였다**

1. **코로나 19 이전과 이후의 MZ 세대의 가구생애주기의 변화 (원 그래프 시각화)**

![가구생애주기로_나눈](https://user-images.githubusercontent.com/54494622/130101151-fb716045-cb33-4e50-bb59-fd8ef0e7339d.jpg)

⇒ 코로나 19 이전과 코로나 19 이후를 비교해보았을 때, **1인 가구**의 비중이 42.3%에서 47.7%로 **5.4% 증가**했으며, **신혼영유아가구**의 경우 55.8%에서 49.6% 로 **6.2% 감소**했다. 초중고자녀 가구의 경우도 **0.8% 증가폭**이 있었다.(=이는 신혼영유아가구가 초중고자녀의 가구로 성장했을 경우라고 생각)

⇒ 따라서 최종적으로 신혼영유아가구의 경우 **55.8%**에서 **50.4%(49.6%+0.8%)**로 **5.4% 감소**했다. _5.4% 감소폭은 신혼영유아가구에서 1인가구로 비중확대를 의미한다._

```python
bc_mz
family_Type_mz2019 = mz2019.groupby('가구생애주기')['mz세대'].count()
family_Type_mz2021 = mz2021.groupby('가구생애주기')['mz세대'].count()
family_Type_mz2019.plot(kind='pie',autopct='%1.1f%%',fontsize=13)
family_Type_mz2021.plot(kind='pie',autopct='%1.1f%%',fontsize=13)
```

**1-2. 가공식품을 구매하는 MZ세대의 코로나 이전과 이후의 분포도 비교**

```python
pfood_mz2019 = mz2019[mz2019['품목중분류명']=='가공식품']
pfood_mz2019.groupby('가구생애주기')['mz세대'].count().plot(kind='pie',autopct='%1.1f%%',fontsize=13)

pfood_mz2021 = mz2021[mz2021['품목중분류명']=='가공식품']
pfood_mz2021.groupby('가구생애주기')['mz세대'].count().plot(kind='pie',autopct='%1.1f%%',fontsize=13)
```

![가공식품비교한것](https://user-images.githubusercontent.com/54494622/130101149-7319f87a-f4df-4e72-86fa-b9ac296c0e0d.jpg)

⇒ 가공식품을 구매한 MZ세대를 따로 분류했을 때,

**코로나 이전 2019년도** 1인가구의 경우 **41.7%**에서 코로나 이후 **20년,21년** **49.3%** 로 **7.6%** 증가하였다.

⇒ 가공식품을 주로 이용하는것은 신혼영유아가구이지만, **_코로나19 이전과 이후의 증가폭이 큰 것은 1인가구였다._**

**_이는, 다시말하면, 1인가구의 증가로 인해 가공식품의 매출이 증가하였다는 가설을 증명한다._**

**X : MZ세대의 가구생애주기의 영향이 있었기 때문이다.**

⇒ 가공식품 중 신혼영유아가구가 가장 많은 매출을 차지하지만 코로나19 이전과 이후 비중이 가장 많이 증가한 것은 1인가구이었다.

**Y : 가공식품의 매출이 증가하였다**

⇒ 밀키트와 같은 홈키트 상품의 매출이 증가하였다.

⇒ 1인~2인 가구를 대상으로 한 밀키트 상품을 집중적으로 공략할 필요가 있다.


### ▣ 가설 2 검증

---

**X : 코로나19 이후 MZ세대의 가구생애주기에 변화가 생겨서 영향을 끼쳤다 → Y : 출산/육아 용품 매출이 줄었다.**

![가구생애주기로_나눈](https://user-images.githubusercontent.com/54494622/130101151-fb716045-cb33-4e50-bb59-fd8ef0e7339d.jpg)

⇒ 코로나 19 이전과 코로나 19 이후를 비교해보았을 때, **1인 가구**의 비중이 42.3%에서 47.7%로 **5.4% 증가**했으며, **신혼영유아가구**의 경우 55.8%에서 49.6% 로 **6.2% 감소**했다. 초중고자녀 가구의 경우도 **0.8% 증가폭**이 있었다.(=이는 신혼영유아가구가 초중고자녀의 가구로 성장했을 경우라고 생각)

⇒ 따라서 최종적으로 신혼영유아가구의 경우 **55.8%**에서 **50.4%(49.6%+0.8%)**로 **5.4% 감소**했다. *5.4% 감소폭은 신혼영유아가구에서 1인가구로 비중확대를 의미한다.*

**X : 코로나19 이후 MZ세대의 가구생애주기에 변화가 생겨서 영향을 끼쳤다**

⇒ 신혼영유아가구의 비중을 줄고 1인가구의 비중이 커졌다.

**Y : 출산/육아 용품 매출이 줄었다.**

⇒ 가구생애주기에 변화가 생겼기 때문이다.
