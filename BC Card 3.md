# BC Card 4

## ※ 특이점을 찾기 위한 분석 ( = 이번 분석 목표 )

---

_⇒ 질문을 막 해본다,,,_

_⇒ 코난이 된 듯 추리하며 예측해본다,,,_

_⇒ 시간투자한다,,,_

⇒ 삽질일수도 있다...

## ▣ 데이터 불러오기

---

```python
import pandas as pd

bc = pd.read_csv('C:\\Users\\sjjung\\Desktop\\contestData\\bccard.csv')
bc

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

# 파생변수 추가 : 연령
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

# 성별 컬럼 여 1, 남 0
bc['성별'] = bc['성별'].apply(lambda x:1 if x=='여성' else 0)

# 파생변수 추가 매출평균
pd.options.display.float_format = '{:,.0f}'.format
bc['매출평균'] = bc['매출금액']/bc['매출건수']

# 파생변수2 추가 mz세대 -> 연령(1,2,3) -> 1, 나머지 0
bc['mz세대'] = bc['연령'].apply(lambda x:1 if x in (1,2,3) else 0)
```

### ※ 추가한 파생변수

---

a. 연령 컬럼 ( 문자형 → 숫자형) 으로 변환
b. 성별 컬럼 ( 문자형 → 숫자형) 으로 변환
c. 매출 평균 ( 매출금액 / 매출건수 )
d. mz세대 (mz세대 여부에 따라 1,0 으로 분리)
e. 분기당 평균 매출 건수 ( 매출 건수 / 분기 수 )

## ▣ 분석 방향

---

1. **bc 카드 데이터에 지역 컬럼을 추가하여 특이점 찾아보기 ( 특이점 X )**

_⇒ 기존의 경우 지역 컬럼을 통해서 큰 특이점을 찾을 수 없다고 생각했지만, 있는 컬럼을 그냥 버리는 것은 좋은 선택이 아니므로 추가해서 다시 분석해볼 것이다._

1. MZ 세대가 많은 지역
2. MZ 세대가 많은 지역에서 주로 구매하는 품목 (대분류, 중분류)
3. 지역별 소득금액에 따라 지역에 가중치를 두어 비교 ( 신한은행 데이터 이용 )
4. 지역별, 성별로 나눠서 많이 구매하는 품목 비교
5. 지역별, 성별로 나눠서 많이 구매하는 품목 비교 (코로나 전과 후로)

6. **bc 카드 데이터에서 MZ 세대 중 20대와 30대 나눠서 비교해보기**

⇒ M세대와 Z세대간의 차이점이 존재할 수도 있으니 나눠서 분석해볼것이다.

### 1-a. MZ 세대가 많은 지역

---

```python
x = bc[bc['mz세대']==1]
xx = x.groupby('고객소재지_광역시도')['mz세대'].count().reset_index()
y  = bc.groupby('고객소재지_광역시도')['mz세대'].count().reset_index()

pd.options.display.float_format = '{:.3f}'.format # 소숫점 3자리까지 표현해주는 판다스 옵션

mz_zone = pd.merge(xx,y,on='고객소재지_광역시도')
mz_zone.columns=['광역시도','mz세대','전체인원']

mz_zone['mz세대'] = mz_zone['mz세대'].astype('float')
mz_zone['전체인원'] = mz_zone['전체인원'].astype('float')
mz_zone['분포']= mz_zone.mz세대/mz_zone.전체인원

my_colors = [ '#ffd7ba','#2a9d8f']
mz_zone.plot.bar(x='광역시도',y=['mz세대','전체인원'],rot=45,color=my_colors,fontsize=8,title="시도별 mz세대 인원수",figsize=(6,4))
```

![1](https://user-images.githubusercontent.com/54494622/130847536-a7dc8448-5a6a-4754-b3db-56da43a79f28.png)

![2](https://user-images.githubusercontent.com/54494622/130847539-409a284b-ab5b-427e-92b2-0ab434361c77.png)

_전체 인원이 다르기 때문에 분포로 다시 한 번 확인해보았다. 그 결과, BC 카드 데이터에서는 MZ 세대의 경우 인천광역시-서울특별시-경기도 순으로 인원이 많았다._

```python
x = bc[bc['mz세대']==1]
xx = x.groupby('고객소재지_시군구')['mz세대'].count().reset_index()
y  = bc.groupby('고객소재지_시군구')['mz세대'].count().reset_index()

pd.options.display.float_format = '{:.3f}'.format # 소숫점 3자리까지 표현해주는 판다스 옵션

mz_zone2 = pd.merge(xx,y,on='고객소재지_시군구')
mz_zone2.columns=['시군구','mz세대','전체인원']

mz_zone2['mz세대'] = mz_zone2['mz세대'].astype('float')
mz_zone2['전체인원'] = mz_zone2['전체인원'].astype('float')
mz_zone2['분포']= mz_zone2.mz세대/mz_zone2.전체인원

my_colors = [ '#ffd7ba','#2a9d8f']
mz_zone2.plot.bar(x='시군구',y=['mz세대','전체인원'],rot=45,color=my_colors,fontsize=8,title="시군구별 mz세대 인원수",figsize=(20,10))
```

![Untitled](https://user-images.githubusercontent.com/54494622/130847563-c51076c3-b0c1-4624-96ca-a7da8c1ddd23.png)

```python
mz_zone2[['시군구','분포']].sort_values(by='분포',ascending=False).head(10)
mz_zone2[['시군구','분포']].sort_values(by='분포',ascending=False).tail(10)
```

![4](https://user-images.githubusercontent.com/54494622/130847540-006f45e7-6847-4de1-9b25-019d96eda08e.jpg)

_상위 10군데의 경우, 경기도 2군데, 서울 6군데, 인천 2군데이었으며, 하위의 경우도 고르게 분포되었다. 하지만 특이점을 찾기는 힘들었다._

### 1-.b MZ 세대가 많은 지역에서 주로 구매하는 품목 (대분류, 중분류)

---

_※ 데이터 분석에 앞서, 1-a 에서 특이점을 찾지 못했으므로, 1-c 에서 신한은행 데이터(서울지역 데이터만 있음) 를 사용할 것을 고려하며, 가장 많은 데이터가 있는 **서울특별시의 데이터**만을 가지고 분석을 해보기로 하였다._

```python
# 서울특별시 mz세대
bc_mz = bc[bc['mz세대']==1]
mz_seoul = bc_mz[bc_mz['고객소재지_광역시도']=='서울특별시']
mz_seoul

bc_seoul = bc[bc['고객소재지_광역시도']=='서울특별시']

x = mz_seoul.groupby('고객소재지_시군구')['mz세대'].count().reset_index()
y  = bc_seoul.groupby('고객소재지_시군구')['mz세대'].count().reset_index()

pd.options.display.float_format = '{:.3f}'.format # 소숫점 3자리까지 표현해주는 판다스 옵션

mz_zone3 = pd.merge(x,y,on='고객소재지_시군구')
mz_zone3.columns=['지역구','mz세대','전체인원']

mz_zone3['mz세대'] = mz_zone3['mz세대'].astype('float')
mz_zone3['전체인원'] = mz_zone3['전체인원'].astype('float')
mz_zone3['분포']= mz_zone3.mz세대/mz_zone3.전체인원

my_colors = [ '#ffd7ba','#2a9d8f']
mz_zone3.plot.bar(x='지역구',y=['mz세대','전체인원'],rot=45,color=my_colors,fontsize=8,title="서울시 지역구별 mz세대 인원수",figsize=(20,10))
```

![4](https://user-images.githubusercontent.com/54494622/130847543-ce699e58-5ef3-42a6-a82d-ae710ba9a6df.png)

_⇒ 서울특별시 지역구별 mz세대 인원을 비교해서 나타낸 차트이다._

![8](https://user-images.githubusercontent.com/54494622/130847544-c53bd762-7db8-4d82-a544-c6958af5756a.jpg)

1. **중구**(_mz세대 비율 상위 1개구_ ) 에서 mz세대가 주로 구매하는 품목 알아보기

---

```python
# 중구 만 따로 분류
junggu = bc_seoul[bc_seoul['고객소재지_시군구']=='중구']

# 중구 (mz세대 비율 상위 1개구 2019년 품목별 매출 순위 )
junggu2019 = junggu[(junggu['기준년월']==201903) | (junggu['기준년월']==201909)]
junggu2019.drop(['기준년월'],axis=1,inplace=True)
junggu2019_total = junggu2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
junggu2019_avg = junggu2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

junggu2019_consume = pd.merge(junggu2019_total,junggu2019_avg)

junggu_mz_2019 = junggu2019_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

junggu_mz_2019['매출순위'] = junggu_mz_2019['매출건수'].rank(method='min',ascending=False)
junggu_mz_2019

# 중구 (mz세대 비율 상위 1개구 2020년,21년 품목별 매출 순위 )
junggu2021 = junggu[(junggu['기준년월']==202003) | (junggu['기준년월']==202009) | (junggu['기준년월']==202103)]
junggu2021.drop(['기준년월'],axis=1,inplace=True)

junggu2021_total = junggu2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
junggu2021_avg = junggu2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

junggu2021_consume = pd.merge(junggu2021_total,junggu2021_avg)
junggu_mz_2021 = junggu2021_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

junggu_mz_2021['매출순위'] = junggu_mz_2021['매출건수'].rank(method='min',ascending=False)
junggu_mz_2021

# 중구 mz세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
junggu_cov_mz = pd.merge(junggu_mz_2019,junggu_mz_2021,how='inner',on='품목중분류명')
junggu_cov_mz.drop(['품목대분류명_y'],axis=1,inplace=True)
junggu_cov_mz.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','매출평균前','순위前','매출금액後','매출건수後','매출평균後','순위後']

junggu_cov_mz['변동순위'] = junggu_cov_mz['순위前'] - junggu_cov_mz['순위後']
junggu_cov_mz
```

![8](https://user-images.githubusercontent.com/54494622/130847545-35e0013c-c1b3-42c6-8e5a-74b82bf2cf42.png)

1. **종로구**(_mz세대 비율 상위 2위 지역구_ ) 에서 mz세대가 주로 구매하는 품목 알아보기

---

```python
# 종로구 만 따로 분류
jongro = bc_seoul[bc_seoul['고객소재지_시군구']=='중구']

# 종로구 (mz세대 비율 상위 2위 2019년 품목별 매출 순위 )
jongro2019 = jongro[(jongro['기준년월']==201903) | (jongro['기준년월']==201909)]
jongro2019.drop(['기준년월'],axis=1,inplace=True)
jongro2019_total = jongro2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
jongro2019_avg = jongro2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

jongro2019_consume = pd.merge(jongro2019_total,jongro2019_avg)

jongro_mz_2019 = jongro2019_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

jongro_mz_2019['매출순위'] = jongro_mz_2019['매출건수'].rank(method='min',ascending=False)
jongro_mz_2019

# 종로구 (mz세대 비율 상위 2위 2020년,21년 품목별 매출 순위 )
jongro2021 = jongro[(jongro['기준년월']==202003) | (jongro['기준년월']==202009) | (jongro['기준년월']==202103)]
jongro2021.drop(['기준년월'],axis=1,inplace=True)

jongro2021_total = jongro2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
jongro2021_avg = jongro2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

jongro2021_consume = pd.merge(jongro2021_total,jongro2021_avg)
jongro_mz_2021 = jongro2021_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

jongro_mz_2021['매출순위'] = jongro_mz_2021['매출건수'].rank(method='min',ascending=False)
jongro_mz_2021

# 종로구 mz세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
jongro_cov_mz = pd.merge(jongro_mz_2019,jongro_mz_2021,how='inner',on='품목중분류명')
jongro_cov_mz.drop(['품목대분류명_y'],axis=1,inplace=True)
jongro_cov_mz.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','매출평균前','순위前','매출금액後','매출건수後','매출평균後','순위後']

jongro_cov_mz['변동순위'] = jongro_cov_mz['순위前'] - jongro_cov_mz['순위後']
jongro_cov_mz
```

![jongrogu](https://user-images.githubusercontent.com/54494622/130847556-c14c0eda-acc9-44b7-95d1-3f0708e1fd5d.png)

1. **서초구**(_mz세대 비율 하위 1위 지역구_ ) 에서 mz세대가 주로 구매하는 품목 알아보기

---

```python
# 서초구 만 따로 분류
seocho = bc_seoul[bc_seoul['고객소재지_시군구']=='서초구']

# 서초구 (mz세대 비율 하위 1위 2019년 품목별 매출 순위 )
seocho2019 = seocho[(seocho['기준년월']==201903) | (seocho['기준년월']==201909)]
seocho2019.drop(['기준년월'],axis=1,inplace=True)
seocho2019_total = seocho2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
seocho2019_avg = seocho2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

seocho2019_consume = pd.merge(seocho2019_total,seocho2019_avg)

seocho_mz_2019 = seocho2019_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

seocho_mz_2019['매출순위'] = seocho_mz_2019['매출건수'].rank(method='min',ascending=False)
seocho_mz_2019

# 서초구 (mz세대 비율 하위 1위 2020년,21년 품목별 매출 순위 )
seocho2021 = seocho[(seocho['기준년월']==202003) | (seocho['기준년월']==202009) | (seocho['기준년월']==202103)]
seocho2021.drop(['기준년월'],axis=1,inplace=True)

seocho2021_total = seocho2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수'].sum().reset_index()
seocho2021_avg = seocho2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

seocho2021_consume = pd.merge(seocho2021_total,seocho2021_avg)
seocho_mz_2021 = seocho2021_consume.sort_values(by=['매출건수','매출금액','매출평균'],ascending=False).head(15)

seocho_mz_2021['매출순위'] = seocho_mz_2021['매출건수'].rank(method='min',ascending=False)
seocho_mz_2021

# 서초구 mz세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
seocho_cov_mz = pd.merge(seocho_mz_2019,seocho_mz_2021,how='inner',on='품목중분류명')
seocho_cov_mz.drop(['품목대분류명_y'],axis=1,inplace=True)
seocho_cov_mz.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','매출평균前','순위前','매출금액後','매출건수後','매출평균後','순위後']

seocho_cov_mz['변동순위'] = seocho_cov_mz['순위前'] - seocho_cov_mz['순위後']
seocho_cov_mz
```

![seocho](https://user-images.githubusercontent.com/54494622/130847557-2a408cd3-6ac2-45ca-9c89-367c13ffde8e.png)

⇒ 아*쉽게도 지역구별로 나눴을 때 품목 구매에 대한 매출 건수 차이가 있지 않았다. 심지어 중구와 종로구는 매출건수가 똑같았으며, ~~다만 차이가 있었던 것은 매출 건수 및 금액이 서초구가 중구와 종로구에 비해 정수배 이상 많았다.~~*

**_⇒ 더이상의 지역별 특징이 BC 카드의 데이터에서 나타나지 않았기 때문에 기존에 분석했던 방향인 세대 및 가구생애주기별로 분석하는 방향을 더 깊게 해봐야겠다._**

### 2. **bc 카드 데이터에서 MZ 세대 중 10대, 20대, 30대 나눠서 비교해보기**

---

**◈ 데이터 전처리**

```python
bc_10 = bc[bc['연령']==1]
bc_10.가구생애주기.unique()

array(['1인가구'], dtype=object)
```

⇒ _10대의 가구생애주기가 1인가구뿐인것으로 봐서, **만 18세** 또는 **만 19세**로만 이루어진 데이터로 추측_

_⇒ 따라서, 10대 20대를 Z세대로 지칭, 30대를 M세대로 구분하기로 함._

```python
bc_20 = bc[(bc['연령']==2) | (bc['연령']==1)]
bc_20.연령.unique() # array([2, 1], dtype=int64)

bc_30 = bc[bc['연령']==3]
```

### z세대 코로나 전과 후 많이 구매한 품목 비교

---

⇒ 코로나 전

```python
# Z세대 10,20대만 따로 분석
bc_20

# z세대 2019년도 품목별 매출건수 상위 top 15
bc_20_2019 = bc_20[(bc_20['기준년월']==201903) | (bc_20['기준년월']==201909)]
bc_20_2019.drop(['기준년월'],axis=1,inplace=True)

bc_20_2019['분기당_평균_매출건수'] = (bc_20_2019['매출건수'] / 2).astype('int64')
bc_20_2019_total = bc_20_2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수','분기당_평균_매출건수'].sum().reset_index()
bc_20_2019_avg = bc_20_2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

bc_20_2019_consume = pd.merge(bc_20_2019_total,bc_20_2019_avg)

bc_20_z_2019 = bc_20_2019_consume.sort_values(by=['분기당_평균_매출건수','매출건수','매출금액','매출평균'],ascending=False).head(15)

bc_20_z_2019['매출순위'] = bc_20_z_2019['분기당_평균_매출건수'].rank(method='min',ascending=False)

bc_20_z_2019
```

⇒ 코로나 후

```python
# z세대 2020,21년도 품목별 매출건수 상위 top 15
bc_20_2021 = bc_20[(bc_20['기준년월']==202003) | (bc_20['기준년월']==202009) | (bc_20['기준년월']==202103)]
bc_20_2021.drop(['기준년월'],axis=1,inplace=True)

bc_20_2021['분기당_평균_매출건수'] = (bc_20_2021['매출건수'] / 3).astype('int64')
bc_20_2021_total = bc_20_2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수','분기당_평균_매출건수'].sum().reset_index()
bc_20_2021_avg = bc_20_2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

bc_20_2021_consume = pd.merge(bc_20_2021_total,bc_20_2021_avg)
bc_20_z_2021 = bc_20_2021_consume.sort_values(by=['분기당_평균_매출건수','매출건수','매출금액','매출평균'],ascending=False).head(15)

bc_20_z_2021['매출순위'] = bc_20_z_2021['분기당_평균_매출건수'].rank(method='min',ascending=False)
bc_20_z_2021
```

⇒ 전과 후 합친 것

```python
# z세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
bc20_cov_mz = pd.merge(bc_20_z_2019,bc_20_z_2021,how='inner',on='품목중분류명')
bc20_cov_mz.drop(['품목대분류명_y'],axis=1,inplace=True)
bc20_cov_mz.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','분기당_평균_매출건수前','매출평균前','순위前','매출금액後','매출건수後','분기당_평균_매출건수後','매출평균後','순위後']

bc20_cov_mz['변동순위'] = bc20_cov_mz['순위前'] - bc20_cov_mz['순위後']
bc20_cov_mz
```

![11](https://user-images.githubusercontent.com/54494622/130847547-e26f1715-8553-4d3f-8e2f-c9809cb0b9aa.jpg)

**※ 분석 결과**

---

- _20대의 경우 성별에서 여성 고객이 남성 고객보다 2.07배 많았다. 따라서, **여성의류**, 스킨케어와 같은 품목들이 상대적으로 30대에 비해 높게 위치해있다_.

- **\*여성의류** 품목의 경우 코로나 전과 후의 순위가 차이가 많이 났다. 보다 정확하게 계산해보기위해, 코로나 전 데이터는 2분기의 데이터이고, 코로나 후의 데이터는 3분기의 데이터이기 때문에 **매출건수에서 각 분기수를 나눠** 코로나 전과 후의 **분기별 매출 평균**을 따로 계산해보겠다. 코로나 전의 경우 분기 당 **13,841** 건, 코로나 이후 분기 당 **111,86** 건으로 약 **19%의 매출 건수가 감소했다는 사실을 알 수 있었다.\***

- _반대로 분기별 평균 매출 건수가 **신선/요리재료**의 경우 **71% 증가**했으며, **건강식품**의 경우도 **41% 증가**, 그리고 **음료**의 경우도 **42%** 증가하며, 식품 및 건강식품의 경우 코로나 19 이후 많은 관심이 있었음을 알 수 있다._

- _20대의 경우 30대와 다르게 **스포츠 의류**의 경우도 **6% 가량 판매량**이 **증가**한 사실을 알 수 있었다. ( = 바디프로필 및 홈트레이닝, 헬스, 골프의 수요 증가로 인한 것으로 추측 )_

- \*20대의 경우 **가공식품**의 분기당 매출 건수가 **71% 증가하였다**. 71%가 증가하였음에도 불구하고 순위가 크게 변동이 없었던 이유는 **20대의 경우는 코로나 이전에도 전체 품목 중에 가공식품을 많이 소비했음을 시사한다.\***

### z세대 코로나 이전과 이후 가구 유형 변화 추이

---

```python
# z세대 코로나이전 가구유형
family_Type_z2019 = bc_20_2019.groupby('가구생애주기')['mz세대'].count()
family_Type_z2019

# z세대 코로나이후 가구유형
family_Type_z2021 = bc_20_2021.groupby('가구생애주기')['mz세대'].count()
family_Type_z2021
```

```python
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

llist = ['1인가구','신혼영유아가구']
explode = [0.05, 0.05]
color=['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
plt.rcParams["font.size"]=15

ax1.pie(family_Type_z2019,autopct='%1.1f%%',labels=llist,explode=explode,colors=color)
#ax1.bar(card1.카드사명,card1.비중,color=colors,label='단위 : %')

ax2.pie(family_Type_z2021,autopct='%1.1f%%',labels=llist,explode=explode,colors=color)
#ax2= family_Type_z2021.plot(kind='pie',autopct='%1.1f%%',fontsize=10)

ax1.set_title('z세대 코로나19 이전 가구유형')
ax2.set_title('z세대 코로나19 이후 가구유형')

ax1.legend(llist)
ax2.legend(llist)
```

![13](https://user-images.githubusercontent.com/54494622/130847553-8af9336d-1fb4-43a4-b238-6a5fac110469.png)

⇒ 20대의 경우 코로나 이전과 비교해 1인 가구가 전체 가구 대비 **3% 증가**하였다.

⇒ **_20대의 소비 변동 폭이 큰 품목들을 보면 20대의 경우 건강에 관심이 늘었다는 사실을 알 수 있다._**

- 식품 관련 : 신선/요리재료, 음료, 가공식품
- 건강 관련 : 건강식품, 스포츠의류

⇒ **_1인가구의 증가 및 코로나 19의 영향으로 인해 가공식품 및 신선/요리재료 매출이 늘어났다는 사실을 알 수 있다._**

- 가공식품(=간편 밀키트,냉동식품)
- 신선/요리재료 (코로나19로 인해 외식이 줄어들어 영향을 끼침)

※ _여성 의류 매출이 큰 폭으로 준 것은 오프라인 하이앤드(명품) 소비 추세로 의류 시장의 변화로 영향을 받은 것으로 추측_

### m세대 코로나 전과 후 많이 구매한 품목 비교

---

⇒ 코로나 전

```python
# m세대 30대만 따로 분석
bc_30

# m세대 2019년도 품목별 매출건수 상위 top 15
bc_30_2019 = bc_30[(bc_30['기준년월']==201903) | (bc_30['기준년월']==201909)]
bc_30_2019.drop(['기준년월'],axis=1,inplace=True)

bc_30_2019['분기당_평균_매출건수'] = (bc_30_2019['매출건수'] / 2).astype('int64')
bc_30_2019_total = bc_30_2019.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수','분기당_평균_매출건수'].sum().reset_index()
bc_30_2019_avg = bc_30_2019.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

bc_30_2019_consume = pd.merge(bc_30_2019_total,bc_30_2019_avg)

bc_30_m_2019 = bc_30_2019_consume.sort_values(by=['분기당_평균_매출건수','매출건수','매출금액','매출평균'],ascending=False).head(15)

bc_30_m_2019['매출순위'] = bc_30_m_2019['분기당_평균_매출건수'].rank(method='min',ascending=False)

bc_30_m_2019
```

⇒ 코로나 이후

```python
# m세대 2020,21년도 품목별 매출건수 상위 top 15
bc_30_2021 = bc_30[(bc_30['기준년월']==202003) | (bc_30['기준년월']==202009) | (bc_30['기준년월']==202103)]
bc_30_2021.drop(['기준년월'],axis=1,inplace=True)

bc_30_2021['분기당_평균_매출건수'] = (bc_30_2021['매출건수'] / 3).astype('int64')
bc_30_2021_total = bc_30_2021.groupby(['품목대분류명','품목중분류명'])['매출금액','매출건수','분기당_평균_매출건수'].sum().reset_index()
bc_30_2021_avg = bc_30_2021.groupby(['품목대분류명','품목중분류명'])['매출평균'].mean().reset_index()

bc_30_2021_consume = pd.merge(bc_30_2021_total,bc_30_2021_avg)
bc_30_m_2021 = bc_30_2021_consume.sort_values(by=['분기당_평균_매출건수','매출건수','매출평균','매출금액'],ascending=False).head(15)

bc_30_m_2021['매출순위'] = bc_30_m_2021['분기당_평균_매출건수'].rank(method='min',ascending=False)
bc_30_m_2021
```

⇒ 전과 후 합친 것

```python
# m세대 코로나 전과 후의 구매 매출건수별 변동순위 합친 데이터프레임 구성
bc30_cov_mz = pd.merge(bc_30_m_2019,bc_30_m_2021,how='inner',on='품목중분류명')
bc30_cov_mz.drop(['품목대분류명_y'],axis=1,inplace=True)
bc30_cov_mz.columns = ['품목대분류','품목중분류','매출금액前','매출건수前','분기당_평균_매출건수前','매출평균前','순위前','매출금액後','매출건수後','분기당_평균_매출건수後','매출평균後','순위後']

bc30_cov_mz['변동순위'] = bc30_cov_mz['순위前'] - bc30_cov_mz['순위後']
bc30_cov_mz
```

![12](https://user-images.githubusercontent.com/54494622/130847549-1d0ddb0e-68d9-49aa-a2a0-fbaecf86388a.jpg)

**※ 분석 결과**

---

- 3*0대의 경우 **여행상품**의 매출이 **-37%** 로 급감하였다. 20대의 경우도 -26% 매출건수가 급감하였지만 여행상품의 총 매출 건수에서 30대가 차지하는 비중이 크다. 코로나 이전에 전체 품목 중 많은 비중을 차지하던 여행 상품이 코로나 이후 코로나 19로 인해 전체 품목 중 현저히 비중이 낮아졌음을 시사한다.*

- \*30대의 경우 반대로 **가공식품** 품목에서 매출이 **37%** 급증하였다. 또한 매출 건수의 순위가 **6위**나 상승하였다. 이는 **30대의 분기당 전체 총 매출건수에서 가공식품이 차지하는 비중이 상당히 올라갔음을 시사하였다.\***

- **\*육아용품, 어린이용품**의 경우 코로나 19 이전에 비해 각각 **-25%, -24%** 감소하였다. 30대의 가구생애주기(=가구유형)에 변화가 있으리라 예상되는 부분이다. **30대의 가구 유형이 어떤 형식으로 바뀌었는지 알아보는 것이 중요하리라 생각된다.\***

- **\*신선/요리재료, 음료, 건강식품** 역시 20대와 마찬가지로 각각 **39%, 18%, 11%** 상승하였다. 이는 코로나 19로 인해 30대 역시 **식품 관련 품목을 온라인으로 많이 소비하며, 전보다 더 건강식품에 관심을 갖게 되었음을 시사한다.\***

### m세대 코로나 이전과 이후 가구 유형 변화 추이

---

```python
# m세대 코로나이전 가구유형
family_Type_m2019 = bc_30_2019.groupby('가구생애주기')['mz세대'].count()
family_Type_m2019

# m세대 코로나이후 가구유형
family_Type_m2021 = bc_30_2021.groupby('가구생애주기')['mz세대'].count()
family_Type_m2021
```

```python
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

llist = ['1인가구','신혼영유아가구','초중고가구']
explode = [0.05, 0.05,0.05]
color=['#e9c46a', '#f4a261', '#e76f51']
plt.rcParams["font.size"]=15

ax1.pie(family_Type_m2019,autopct='%1.1f%%',labels=llist,explode=explode,colors=color)

ax2.pie(family_Type_m2021,autopct='%1.1f%%',labels=llist,explode=explode,colors=color)

ax1.set_title('m세대 코로나19 이전 가구유형')
ax2.set_title('m세대 코로나19 이후 가구유형')

ax1.legend(llist)
ax2.legend(llist)
```

![Untitled 1](https://user-images.githubusercontent.com/54494622/130847560-a4a94245-4ee3-40a8-9f54-39b6f1594ad3.png)

⇒ 30대의 경우 전체 가구 대비 1인가구의 비중이 **5.6% 증가**하였다. 반대로 말하자면 신혼영유아가구의 비중은 **5.6% 감소**하였다는 것을 의미한다.

- _30대의 신혼영유아가구의 가구 수가 줄어들었기 때문에 육아용품 및 어린이용품 판매 매출에 영향을 끼쳤다고 예상한다._

- _30대의 1인가구 비율이 늘어났기 때문에 가공식품의 매출에 영향을 끼쳤다고 예상한다._

- _코로나 19로 인해 30대 역시 건강에 관심이 많아졌기 때문에, 신선/요리재료 및 건강식품, 음료 판매량이 늘어났다고 예상한다._

- _코로나 19로 인해 집에서 요리를 직접 해서 먹는 경우가 늘어났기 때문에 가공식품, 및 신선/요리재료의 매출이 늘어났을 것이다._

### ※ M세대와 Z세대 판매량 변화가 있는 품목

---

![15](https://user-images.githubusercontent.com/54494622/130847554-c4f72394-6fff-4f17-ada3-52bb0d3dac8d.png)

## ▣ 머신러닝 예측 방향

---

⇒ 검토
