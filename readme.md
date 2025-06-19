```python
import numpy as np
import pandas as pd
import os
import seaborn as sns
```


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv('dataset.csv')
```

## Первичный анализ


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>track_id</th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>energy</th>
      <th>...</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5SuOikwiRyPMVoIQDJUgSV</td>
      <td>Gen Hoshino</td>
      <td>Comedy</td>
      <td>Comedy</td>
      <td>73</td>
      <td>230666</td>
      <td>False</td>
      <td>0.676</td>
      <td>0.4610</td>
      <td>...</td>
      <td>-6.746</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.0322</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.7150</td>
      <td>87.917</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4qPNDBW1i3p13qLCt0Ki3A</td>
      <td>Ben Woodward</td>
      <td>Ghost (Acoustic)</td>
      <td>Ghost - Acoustic</td>
      <td>55</td>
      <td>149610</td>
      <td>False</td>
      <td>0.420</td>
      <td>0.1660</td>
      <td>...</td>
      <td>-17.235</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.9240</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.2670</td>
      <td>77.489</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1iJBSr7s7jYXzM8EGcbK5b</td>
      <td>Ingrid Michaelson;ZAYN</td>
      <td>To Begin Again</td>
      <td>To Begin Again</td>
      <td>57</td>
      <td>210826</td>
      <td>False</td>
      <td>0.438</td>
      <td>0.3590</td>
      <td>...</td>
      <td>-9.734</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.2100</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.1200</td>
      <td>76.332</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6lfxq3CG4xtTiEg7opyCyx</td>
      <td>Kina Grannis</td>
      <td>Crazy Rich Asians (Original Motion Picture Sou...</td>
      <td>Can't Help Falling In Love</td>
      <td>71</td>
      <td>201933</td>
      <td>False</td>
      <td>0.266</td>
      <td>0.0596</td>
      <td>...</td>
      <td>-18.515</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.9050</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.1430</td>
      <td>181.740</td>
      <td>3</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5vjLSffimiIP26QG5WcN2K</td>
      <td>Chord Overstreet</td>
      <td>Hold On</td>
      <td>Hold On</td>
      <td>82</td>
      <td>198853</td>
      <td>False</td>
      <td>0.618</td>
      <td>0.4430</td>
      <td>...</td>
      <td>-9.681</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.4690</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.1670</td>
      <td>119.949</td>
      <td>4</td>
      <td>acoustic</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113995</th>
      <td>113995</td>
      <td>2C3TZjDRiAzdyViavDJ217</td>
      <td>Rainy Lullaby</td>
      <td>#mindfulness - Soft Rain for Mindful Meditatio...</td>
      <td>Sleep My Little Boy</td>
      <td>21</td>
      <td>384999</td>
      <td>False</td>
      <td>0.172</td>
      <td>0.2350</td>
      <td>...</td>
      <td>-16.393</td>
      <td>1</td>
      <td>0.0422</td>
      <td>0.6400</td>
      <td>0.928000</td>
      <td>0.0863</td>
      <td>0.0339</td>
      <td>125.995</td>
      <td>5</td>
      <td>world-music</td>
    </tr>
    <tr>
      <th>113996</th>
      <td>113996</td>
      <td>1hIz5L4IB9hN3WRYPOCGPw</td>
      <td>Rainy Lullaby</td>
      <td>#mindfulness - Soft Rain for Mindful Meditatio...</td>
      <td>Water Into Light</td>
      <td>22</td>
      <td>385000</td>
      <td>False</td>
      <td>0.174</td>
      <td>0.1170</td>
      <td>...</td>
      <td>-18.318</td>
      <td>0</td>
      <td>0.0401</td>
      <td>0.9940</td>
      <td>0.976000</td>
      <td>0.1050</td>
      <td>0.0350</td>
      <td>85.239</td>
      <td>4</td>
      <td>world-music</td>
    </tr>
    <tr>
      <th>113997</th>
      <td>113997</td>
      <td>6x8ZfSoqDjuNa5SVP5QjvX</td>
      <td>Cesária Evora</td>
      <td>Best Of</td>
      <td>Miss Perfumado</td>
      <td>22</td>
      <td>271466</td>
      <td>False</td>
      <td>0.629</td>
      <td>0.3290</td>
      <td>...</td>
      <td>-10.895</td>
      <td>0</td>
      <td>0.0420</td>
      <td>0.8670</td>
      <td>0.000000</td>
      <td>0.0839</td>
      <td>0.7430</td>
      <td>132.378</td>
      <td>4</td>
      <td>world-music</td>
    </tr>
    <tr>
      <th>113998</th>
      <td>113998</td>
      <td>2e6sXL2bYv4bSz6VTdnfLs</td>
      <td>Michael W. Smith</td>
      <td>Change Your World</td>
      <td>Friends</td>
      <td>41</td>
      <td>283893</td>
      <td>False</td>
      <td>0.587</td>
      <td>0.5060</td>
      <td>...</td>
      <td>-10.889</td>
      <td>1</td>
      <td>0.0297</td>
      <td>0.3810</td>
      <td>0.000000</td>
      <td>0.2700</td>
      <td>0.4130</td>
      <td>135.960</td>
      <td>4</td>
      <td>world-music</td>
    </tr>
    <tr>
      <th>113999</th>
      <td>113999</td>
      <td>2hETkH7cOfqmz3LqZDHZf5</td>
      <td>Cesária Evora</td>
      <td>Miss Perfumado</td>
      <td>Barbincor</td>
      <td>22</td>
      <td>241826</td>
      <td>False</td>
      <td>0.526</td>
      <td>0.4870</td>
      <td>...</td>
      <td>-10.204</td>
      <td>0</td>
      <td>0.0725</td>
      <td>0.6810</td>
      <td>0.000000</td>
      <td>0.0893</td>
      <td>0.7080</td>
      <td>79.198</td>
      <td>4</td>
      <td>world-music</td>
    </tr>
  </tbody>
</table>
<p>114000 rows × 21 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 114000 entries, 0 to 113999
    Data columns (total 21 columns):
     #   Column            Non-Null Count   Dtype  
    ---  ------            --------------   -----  
     0   Unnamed: 0        114000 non-null  int64  
     1   track_id          114000 non-null  object 
     2   artists           113999 non-null  object 
     3   album_name        113999 non-null  object 
     4   track_name        113999 non-null  object 
     5   popularity        114000 non-null  int64  
     6   duration_ms       114000 non-null  int64  
     7   explicit          114000 non-null  bool   
     8   danceability      114000 non-null  float64
     9   energy            114000 non-null  float64
     10  key               114000 non-null  int64  
     11  loudness          114000 non-null  float64
     12  mode              114000 non-null  int64  
     13  speechiness       114000 non-null  float64
     14  acousticness      114000 non-null  float64
     15  instrumentalness  114000 non-null  float64
     16  liveness          114000 non-null  float64
     17  valence           114000 non-null  float64
     18  tempo             114000 non-null  float64
     19  time_signature    114000 non-null  int64  
     20  track_genre       114000 non-null  object 
    dtypes: bool(1), float64(9), int64(6), object(5)
    memory usage: 17.5+ MB


Unnamed: 0 - повторяет индексы датасета

track_id - id песни, случайный набор цифр и латинских букв

artists - имена / псевдонимы артистов. 

album_name - название альбома песни

track_name - название песни

popularity - популярность песни, таргет переменная. Принимает значения от 0 до 100

duration_ms - длительность песни в миллисекундах

explicit - булевое значение. True - в музыке существует брань. False - отсутствует

danceability - танцевальность песни, значение от 0 до 1

energy - энергичность песни, значение от 0 до 1

key - тональность песни от 0 до 10

loudness - громкость песни, значение десятичная дробь

mode - режим аудиозаписи, значения либо 0 либо 1

speechiness - красноречивость текста песни, значения от 0 до 1

acousticness - качество звука, значение от 0 до 1

instrumentalness - показатель звучания инструментов в песне, значение от 0 до 1

liveness - живость песни, значение от 0 до 1

valence - валентность песни, её привлекательность для человеческого слуха, значение от 0 до 1

tempo - темп песни, указан в ударах в минуту

time_signature - размер такта, указывает, сколько ударов приходится на каждый такт песни. Значение - цифра от 0 до 4

track_genre - жанр песни


```python
data['popularity'].value_counts()
```




    popularity
    0      16020
    22      2354
    21      2344
    44      2288
    1       2140
           ...  
    98         7
    94         7
    95         5
    100        2
    99         1
    Name: count, Length: 101, dtype: int64




```python
data['popularity'].unique()
```




    array([ 73,  55,  57,  71,  82,  58,  74,  80,  56,  69,  52,  62,  54,
            68,  67,  75,  63,  70,   0,   1,  46,  61,  60,  51,  66,  64,
            65,  44,  45,  50,  59,  49,  53,  47,  43,  42,  20,  22,  35,
            19,  24,  18,  23,  40,  38,  41,  30,  37,  39,  48,  36,  34,
            26,  32,  33,  21,  31,  28,  29,  27,  25,  16,   3,  12,   7,
            10,   9,  11,  17,   8,  15,  87,  83,  86,  93,  76,  78,   4,
             2,   5,  85,  81,  84,  72,  79,  77,   6,  13,  14,  89,  96,
           100,  98,  88,  92,  90,  91,  99,  97,  95,  94])



popularity - численное занчение, лежащее от 0 до 100. Большинстов треков имеют низкую оценку или 0 (нет оценки)


```python
data.shape
```




    (114000, 21)




```python
data.isnull().sum()
```




    Unnamed: 0          0
    track_id            0
    artists             1
    album_name          1
    track_name          1
    popularity          0
    duration_ms         0
    explicit            0
    danceability        0
    energy              0
    key                 0
    loudness            0
    mode                0
    speechiness         0
    acousticness        0
    instrumentalness    0
    liveness            0
    valence             0
    tempo               0
    time_signature      0
    track_genre         0
    dtype: int64



Пустых колонок мало, поэтому удалим их.


```python
data = data.dropna()
```


```python
data.isnull().sum()
```




    Unnamed: 0          0
    track_id            0
    artists             0
    album_name          0
    track_name          0
    popularity          0
    duration_ms         0
    explicit            0
    danceability        0
    energy              0
    key                 0
    loudness            0
    mode                0
    speechiness         0
    acousticness        0
    instrumentalness    0
    liveness            0
    valence             0
    tempo               0
    time_signature      0
    track_genre         0
    dtype: int64




```python
data.shape
```




    (113999, 21)



Имеем одну строку с пропуском

Преобразуем категориальные признаки в удобный числовой формат, для этого используем следущие 2 ячейки


```python
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
```


```python
from sklearn.preprocessing import LabelEncoder
for col in categorical_cols:
  le = LabelEncoder()
  data[col] = le.fit_transform(data[col])
  label_encoders[col] = le
```

    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\4029885739.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data[col] = le.fit_transform(data[col])
    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\4029885739.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data[col] = le.fit_transform(data[col])
    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\4029885739.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data[col] = le.fit_transform(data[col])
    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\4029885739.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data[col] = le.fit_transform(data[col])
    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\4029885739.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data[col] = le.fit_transform(data[col])


Анализируем корреляции для отбора ключевых признаков и финальной подготовки данных


```python
corr_matrix = data.corr()
plt.figure(figsize=(21, 21))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
```


    
![png](spotify_files/spotify_21_0.png)
    


Из диаграммы выше можно сделать 3 вывода:

1 - ID записи в таблице (Unnamed: 0) хранят в себе информацию о жанре трека (track_genre), стоящего в данной записи.

2 - С популярностью (popularity) особой корреляции не у одной из характеристик нет. Что говорит нам о многофакторности явления популярности, возможно, мы сможем выявить идеальные прараметры для популярной песни, этим мы и займёмся

3 - Существует корреляция между loudness, energy и acousticness. Причём у energy с loudness прямая, а у energy с acousticness обратная.

Как таковую пользу нам приносит только 3й факт, возможно мы объединим эти столбцы

Сделаем копию оригинального датасета, чтобы работать с категориальными признаками в их оригинальном виде


```python
origin_data = pd.read_csv('dataset.csv')
origin_data = origin_data.dropna()
```

Разделим характеристики по признакам


```python
numerical_features = origin_data.select_dtypes(include=['number']).columns.tolist() #числовые признаки
categorical_features = origin_data.select_dtypes(include=['object']).columns.tolist() #категориальные признаки
```

Уберём ненужное


```python
numerical_features.pop(0)
```




    'Unnamed: 0'




```python
numerical_features.pop(0)
```




    'popularity'




```python
numerical_features
```




    ['duration_ms',
     'danceability',
     'energy',
     'key',
     'loudness',
     'mode',
     'speechiness',
     'acousticness',
     'instrumentalness',
     'liveness',
     'valence',
     'tempo',
     'time_signature']



## Анализ числовых характеристик

Мы решили разбить данные на 3 категории:

1 Самые популярные песни (popularity > 90)

2 Средние (40 <= popularity <= 90 )

3 Непопулярные (popularity < 40)

Целью было выявить типичные для каждой группы признаки и сравнить их с другими группами.

Нужно отметить, что нам стоит сосредоточиться на различиях между типичными значениями самых популярных песен (красная линия) и средних (зелёная линия). Т. к. в непопулярных песнях большое число песен с нулевой популярностью, но как показала практика медианное значение 2 и 3 группы различаются довольно редко.


```python
for numerical_feature in numerical_features:
  popular = data[data['popularity'] > 90]
  unpopular = data[data['popularity'] < 40]
  medium = data[data['popularity'] >= 40]
  medium = medium[medium['popularity'] <= 90]

  plt.figure(figsize=(10, 6))

  plt.scatter(popular[numerical_feature], popular['popularity'], c='orange', label='popular')
  plt.axvline(x=popular[numerical_feature].mean(), color='red', linestyle='--', label='mean popular')

  plt.scatter(medium[numerical_feature], medium['popularity'], c='lightgreen', label='medium')
  plt.axvline(x=medium[numerical_feature].mean(), color='green', linestyle='--', label='mean medium')

  plt.scatter(unpopular[numerical_feature], unpopular['popularity'], c='lightblue', label='unpopular')
  plt.axvline(x=unpopular[numerical_feature].mean(), color='blue', linestyle='--', label='mean unpopular')

  plt.legend()

  plt.xlabel(numerical_feature)
  plt.ylabel('popularity')

  plt.show()
```


    
![png](spotify_files/spotify_33_0.png)
    



    
![png](spotify_files/spotify_33_1.png)
    



    
![png](spotify_files/spotify_33_2.png)
    



    
![png](spotify_files/spotify_33_3.png)
    



    
![png](spotify_files/spotify_33_4.png)
    



    
![png](spotify_files/spotify_33_5.png)
    



    
![png](spotify_files/spotify_33_6.png)
    



    
![png](spotify_files/spotify_33_7.png)
    



    
![png](spotify_files/spotify_33_8.png)
    



    
![png](spotify_files/spotify_33_9.png)
    



    
![png](spotify_files/spotify_33_10.png)
    



    
![png](spotify_files/spotify_33_11.png)
    



    
![png](spotify_files/spotify_33_12.png)
    


## Выводы:

### duration_ms

<p>Про длительность трека ничего особенного не скажешь, разве что долгие треки встречаются редко и популярность их крайне мала.</p>
<p>Поэтому обрежем данные 5 минутами и приведём график из ms в min</p>


```python
popular = data[data['popularity'] > 90]
unpopular = data[data['popularity'] < 40]
medium = data[data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

popular_med_duration_ms = popular['duration_ms'].mean()
medium_med_duration_ms = medium['duration_ms'].mean()
unpopular_med_duration_ms = unpopular['duration_ms'].mean()

popular = popular[popular['duration_ms'] < 5 * 1000 * 60]
medium = medium[medium['duration_ms'] < 5 * 1000 * 60]
unpopular = unpopular[unpopular['duration_ms'] < 5 * 1000 * 60]

plt.figure(figsize=(10, 6))

plt.scatter(popular['duration_ms'] / (1000 * 60), popular['popularity'], c='orange', label='popular')
plt.axvline(x=popular_med_duration_ms / (1000 * 60), color='red', linestyle='--', label='mean popular')

plt.scatter(medium['duration_ms'] / (1000 * 60), medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium_med_duration_ms / (1000 * 60), color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['duration_ms'] / (1000 * 60), unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular_med_duration_ms / (1000 * 60), color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('duration_min')
plt.ylabel('popularity')

plt.show()
```


    
![png](spotify_files/spotify_37_0.png)
    


<p>Теперь можно уверенно сказать, что все группы имеют примерно одинаковую среднюю длительность, однако самые популярные треки находятся в пределе от 2,5 до 4,5 минут</p>

### danceability

<p>Что касается графика danceability, можно сразу сказать, что Самые популярные треки имеют большой danceability (больше 0.5), что сильно выделяет их среди других групп</p>


![image.png](data:image/png;base64,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)

### energy_loudness_acousticness

Для оценки улучшений сохранили первоначальный вид данных.


```python
working_data = data
```

Как было уже замечено loudness, energy и acousticness имеют между собой корреляцию. Из этих соображний делаем следущее:


```python
#Объединяем всё в energy_loudness_acousticness
working_data['energy_loudness_acousticness'] = data['energy'] * data['loudness'] * (1 - data['acousticness'])
#acousticness имеет обратную корреляцию

working_data = working_data.drop(columns=['Unnamed: 0', 'track_id']) # за ненадобностью
working_data = working_data.drop(columns=['energy', 'loudness', 'acousticness']) # убираем дубли (возможно это сделает хуже)
```

    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\387505573.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      working_data['energy_loudness_acousticness'] = data['energy'] * data['loudness'] * (1 - data['acousticness'])



```python
working_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>key</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>energy_loudness_acousticness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10357</td>
      <td>8100</td>
      <td>11741</td>
      <td>73</td>
      <td>230666</td>
      <td>False</td>
      <td>0.676</td>
      <td>1</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.7150</td>
      <td>87.917</td>
      <td>4</td>
      <td>0</td>
      <td>-3.009767</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3287</td>
      <td>14796</td>
      <td>22528</td>
      <td>55</td>
      <td>149610</td>
      <td>False</td>
      <td>0.420</td>
      <td>1</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.2670</td>
      <td>77.489</td>
      <td>4</td>
      <td>0</td>
      <td>-0.217437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12397</td>
      <td>39162</td>
      <td>60774</td>
      <td>57</td>
      <td>210826</td>
      <td>False</td>
      <td>0.438</td>
      <td>0</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.1200</td>
      <td>76.332</td>
      <td>4</td>
      <td>0</td>
      <td>-2.760660</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14839</td>
      <td>8580</td>
      <td>9580</td>
      <td>71</td>
      <td>201933</td>
      <td>False</td>
      <td>0.266</td>
      <td>0</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.1430</td>
      <td>181.740</td>
      <td>3</td>
      <td>0</td>
      <td>-0.104832</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5255</td>
      <td>16899</td>
      <td>25689</td>
      <td>82</td>
      <td>198853</td>
      <td>False</td>
      <td>0.618</td>
      <td>2</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.1670</td>
      <td>119.949</td>
      <td>4</td>
      <td>0</td>
      <td>-2.277291</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113995</th>
      <td>22486</td>
      <td>66</td>
      <td>53329</td>
      <td>21</td>
      <td>384999</td>
      <td>False</td>
      <td>0.172</td>
      <td>5</td>
      <td>1</td>
      <td>0.0422</td>
      <td>0.928000</td>
      <td>0.0863</td>
      <td>0.0339</td>
      <td>125.995</td>
      <td>5</td>
      <td>113</td>
      <td>-1.386848</td>
    </tr>
    <tr>
      <th>113996</th>
      <td>22486</td>
      <td>66</td>
      <td>65090</td>
      <td>22</td>
      <td>385000</td>
      <td>False</td>
      <td>0.174</td>
      <td>0</td>
      <td>0</td>
      <td>0.0401</td>
      <td>0.976000</td>
      <td>0.1050</td>
      <td>0.0350</td>
      <td>85.239</td>
      <td>4</td>
      <td>113</td>
      <td>-0.012859</td>
    </tr>
    <tr>
      <th>113997</th>
      <td>4952</td>
      <td>5028</td>
      <td>38207</td>
      <td>22</td>
      <td>271466</td>
      <td>False</td>
      <td>0.629</td>
      <td>0</td>
      <td>0</td>
      <td>0.0420</td>
      <td>0.000000</td>
      <td>0.0839</td>
      <td>0.7430</td>
      <td>132.378</td>
      <td>4</td>
      <td>113</td>
      <td>-0.476733</td>
    </tr>
    <tr>
      <th>113998</th>
      <td>18534</td>
      <td>7238</td>
      <td>21507</td>
      <td>41</td>
      <td>283893</td>
      <td>False</td>
      <td>0.587</td>
      <td>7</td>
      <td>1</td>
      <td>0.0297</td>
      <td>0.000000</td>
      <td>0.2700</td>
      <td>0.4130</td>
      <td>135.960</td>
      <td>4</td>
      <td>113</td>
      <td>-3.410587</td>
    </tr>
    <tr>
      <th>113999</th>
      <td>4952</td>
      <td>24357</td>
      <td>5999</td>
      <td>22</td>
      <td>241826</td>
      <td>False</td>
      <td>0.526</td>
      <td>1</td>
      <td>0</td>
      <td>0.0725</td>
      <td>0.000000</td>
      <td>0.0893</td>
      <td>0.7080</td>
      <td>79.198</td>
      <td>4</td>
      <td>113</td>
      <td>-1.585222</td>
    </tr>
  </tbody>
</table>
<p>113999 rows × 17 columns</p>
</div>



Теперь проверим новую характеристику - energy_loudness_acousticness



```python
popular = working_data[working_data['popularity'] > 90]
unpopular = working_data[working_data['popularity'] < 40]
medium = working_data[working_data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

plt.figure(figsize=(10, 6))

plt.scatter(popular['energy_loudness_acousticness'], popular['popularity'], c='orange', label='popular')
plt.axvline(x=popular['energy_loudness_acousticness'].mean(), color='red', linestyle='--', label='mean popular')

plt.scatter(medium['energy_loudness_acousticness'], medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium['energy_loudness_acousticness'].mean(), color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['energy_loudness_acousticness'], unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular['energy_loudness_acousticness'].mean(), color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('energy_loudness_acousticness')
plt.ylabel('popularity')

plt.show()
```


    
![png](spotify_files/spotify_49_0.png)
    


Ситуация таже, что и с duration_ms. Обрежем от -5 до 0


```python
tempo_data = working_data[working_data['popularity'] > 90]
popular_med = tempo_data['energy_loudness_acousticness'].mean()

tempo_data = working_data[working_data['popularity'] >= 40]
tempo_data = tempo_data[tempo_data['popularity'] >= 40]
medium_med = tempo_data['energy_loudness_acousticness'].mean()

tempo_data = working_data[working_data['popularity'] < 40]
unpopular_med = tempo_data['energy_loudness_acousticness'].mean()

tempo_data = working_data[working_data['energy_loudness_acousticness'] > -5]
tempo_data = tempo_data[tempo_data['energy_loudness_acousticness'] < 0]

popular = tempo_data[tempo_data['popularity'] > 90]
unpopular = tempo_data[tempo_data['popularity'] < 40]
medium = tempo_data[tempo_data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

plt.figure(figsize=(10, 6))

plt.scatter(popular['energy_loudness_acousticness'], popular['popularity'], c='orange', label='popular')
plt.axvline(x=popular_med, color='red', linestyle='--', label='mean popular')

plt.scatter(medium['energy_loudness_acousticness'], medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium_med, color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['energy_loudness_acousticness'], unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular_med, color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('energy_loudness_acousticness')
plt.ylabel('popularity')

plt.show()
```


    
![png](spotify_files/spotify_51_0.png)
    


Особо средние значения групп не отличаются, но значения Самых лучших песен опять ограничено небольшим промежутком, в данном случае от -4.5 до -0.5

### key

Про key тоже ничего особенного не скажешь, медианы сливаются, да и так анализировать подобный вид графика не имеет смысла

![image.png](data:image/png;base64,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8OAnn3xy586dt9122/Dhw48cOULyhJEPAtddd92CBQt27Njx1ltvTZkypbGxsbOz00dgxHAgsGnTpssvv/xTn/oUxj4Haj5M29rampqavvWtb23cuPGdd9556aWX9u7d6yMwYpAI3HvvvR/72MdWrFixb9++p59+etiwYY888gjJE0ZJCbzwwgs/+MEPli5det7Y98ADDzQ0NCxfvnzr1q1Tp079y7/8y66urqRJMuqHsY9U+Obm5tmzZxdMe3p6xowZc//995M8YeSbwNGjR3O53Nq1a30HRjwbgRMnTowbN+63v/3t5MmTMfbZSAU4u/vuu6+99toAgRGSROD666+fMWOGMr3pppu+8Y1vqC+xCEpAH/t6e3svueSSf//3fy9kbG9vr6urW7RoUVABAy84xr74mnZ3d9fW1uo/Z542bdrUqVPjPWERgEBra2sul9u+fXuA2AhpJDBt2rS5c+dGUYSxz8go2MEVV1wxd+7cm2++efTo0VddddUTTzwRLBUClyFw7733NjU17d69O4qit9566+KLL/7FL35Rxg5bAQjoY9/bb7+dy+W2bNmi8vzt3/7tP/zDP6gvsaAQwNgXT+nAgQO5XG79+vXK9Hvf+15zc7P6EovUCPT09Fx//fXXXHNNahmRKIqiRYsWTZgwofDLFIx96bdEXf/jn//5n3/3u989/vjjQ4YMWbhwYfoyMpuxp6fn7rvvrqmpueCCC2pqmcqaPwAABvBJREFUau67777Mokj/wvWx7/XXX8/lcgcPHlQybrnllq997WvqSywoBDD2xVPC2BfPKC2LmTNnNjU15fP5tBIiT/Tee+9dfPHFW7duLbDA2Jd+TwwaNOjqq69Web/zne9MmjRJfYlFaAKLFi269NJLFy1atG3btp///OcjR47E2B2auYqPsU+h8LXA2BdPEr/kjWeUisXs2bMvvfTSd955J5VsSFIksGzZslwuV/vhI5fL1dTU1NbWnjt3DozSIdDY2HjrrbeqXI899tiYMWPUl1iEJnDppZfOnz9fZfnxj388fvx49SUWQQnoYx9+yesFNcY+Esbm5uY77rijYNrT0zN27Fi8pYMEzpNRb2/v7Nmzx4wZg7freyLqEOaDDz7Yrj0mTpz4zW9+E/dWOhBkm37961/X39Ixd+5c/Yd/7PAIEENg5MiRjz32mDK67777xo0bp77EIigBfewrvKXjP/7jPwoZOzo68JaOBPAx9pGgLV68uK6ubuHChbt27br99tuHDx9++PBhkieMfBD49re/3dDQsGbNmkMfPk6dOuUjMGI4E8AveZ2RsR02bdp0wQUX3Hvvva2trU899VR9fT3eUsCG6hBg+vTpY8eOLfwBl6VLl44aNequu+5y8IepO4ETJ05s6X/kcrmHHnpoy5Yt7777bhRFDzzwwPDhw5999tlt27bdcMMN+AMu7mgjjH1UaI8++mhjY+PgwYObm5vfeOMNqhvsfBDIlTwWLFjgIzBiOBPA2OeMzIfDc889N2HChLq6ur/+67/GO3l9EHWI8cEHH8yZM6exsbHw55p/8IMfdHd3O/jD1J3A6tWrz3vVnz59ehRFhT/X/Bd/8Rd1dXVf+MIXCm+vdg+faQ+MfZkuPy4eBEAABEAABEAgOwQw9mWn1rhSEAABEAABEACBTBPA2Jfp8uPiQQAEQAAEQAAEskMAY192ao0rBQEQAAEQAAEQyDQBjH2ZLj8uHgRAAARAAARAIDsEMPZlp9a4UhAAARAAARAAgUwTwNiX6fLj4kEABEAABEAABLJDAGNfdmqNKwUBEAABEAABEMg0AYx9mS4/Lh4EQIBOAH+qms4KliAAApVJAGNfZdYFqkAABCqOAMa+iisJBIEACDgSwNjnCAzmIAACWSWAsS+rlcd1g8DAIYCxb+DUElcCAiAQlIA+9q1YseKiiy76xS9+8d57791yyy0NDQ0jRoyYOnXqvn37oihau3btBRdccOjQIaVnzpw51157rfoSCxAAARAQIYCxTwQ7koIACFQfATX2PfXUU3/+53/+3HPPnTlz5oorrpgxY8a2bdt27dr1d3/3d+PHj+/u7o6i6K/+6q8efPDBwkWeOXNm1KhRTz75ZPVdMxSDAAgMLAIY+wZWPXE1IAACwQgUxr758+c3NDSsWbMmiqL/+Z//GT9+fG9vbyFnd3f3hRde+NJLL0VR9G//9m9XXHFFYf+ZZ54ZNmxYZ2dnMGkIDAIgAAIkAhj7SJhgBAIgAAKTJ08eO3bsoEGDNm3aVKBx55131tbWDtUeNTU1jz32WBRFR44cGTRo0IYNG6Io+spXvjJjxgwABAEQAAFxAhj7xEsAASAAAtVBYPLkyV/+8pfHjBkzc+bMwk/4Zs6c2dzc3PrRR3t7e+F6brrppttvv/3w4cMXXHDBa6+9Vh0XCZUgAAIDmgDGvgFdXlwcCICAPwKFX/Lu3r374x//+OzZs6MoeuKJJ0aMGNHR0VE2yQsvvNDQ0PCjH/1o/PjxZQ2wCQIgAAIpE8DYlzJwpAMBEKhWAuotHS0tLZdccsmcOXNOnjw5bty4z33uc6+++uo777yzevXq73znO/l8vnCFPT09l1122eDBgx944IFqvWboBgEQGFgEMPYNrHriakAABIIRUGNfFEW7du26+OKL//Ef//HQoUPTpk0bNWpUXV3dJz7xidtuu03/4d8999xTW1t78ODBYKIQGARAAAQcCGDsc4AFUxAAARBwIjBjxoyvfOUrTi4wBgEQAIFwBDD2hWOLyCAAAtkl0N7evm7duiFDhrz88svZpYArBwEQqDACGPsqrCCQAwIgMCAITJ48+cILL5w7d+6AuBpcBAiAwAAhgLFvgBQSlwECIAACIAACIAACdgIY++x8cAoCIAACIAACIAACA4QAxr4BUkhcBgiAAAiAAAiAAAjYCWDss/PBKQiAAAiAAAiAAAgMEAIY+wZIIXEZIAACIAACIAACIGAngLHPzgenIAACIAACIAACIDBACGDsGyCFxGWAAAiAAAiAAAiAgJ0Axj47H5yCAAiAAAiAAAiAwAAhgLFvgBQSlwECIAACIAACIAACdgIY++x8cAoCIAACIAACIAACA4TA/wfJyt4A6vk51gAAAABJRU5ErkJggg==)

Поэтому просто выведем вероятность стать популярным для каждого key


```python
best_d = data[data['popularity'] > 90]
key_list = []
ver_list = []
for key in data['key'].unique():
  ver = best_d[ best_d['key'] == key ]['key'].count() / data[ data['key'] == key ]['key'].count()
  key_list.append(key)
  ver_list.append(ver)


fig, ax = plt.subplots()


ax.bar(key_list, ver_list)

ax.set_ylabel('Вероятность')
ax.set_title('Вероятности по ключам')

plt.show()
```


    
![png](spotify_files/spotify_57_0.png)
    


Вероятности скачут в пределе 0.001, что довольно мало, поэтому мы решили удалить эту колонку


```python
working_data = working_data.drop(columns=['key'])
working_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artists</th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>danceability</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>track_genre</th>
      <th>energy_loudness_acousticness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10357</td>
      <td>8100</td>
      <td>11741</td>
      <td>73</td>
      <td>230666</td>
      <td>False</td>
      <td>0.676</td>
      <td>0</td>
      <td>0.1430</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.7150</td>
      <td>87.917</td>
      <td>4</td>
      <td>0</td>
      <td>-3.009767</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3287</td>
      <td>14796</td>
      <td>22528</td>
      <td>55</td>
      <td>149610</td>
      <td>False</td>
      <td>0.420</td>
      <td>1</td>
      <td>0.0763</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.2670</td>
      <td>77.489</td>
      <td>4</td>
      <td>0</td>
      <td>-0.217437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12397</td>
      <td>39162</td>
      <td>60774</td>
      <td>57</td>
      <td>210826</td>
      <td>False</td>
      <td>0.438</td>
      <td>1</td>
      <td>0.0557</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.1200</td>
      <td>76.332</td>
      <td>4</td>
      <td>0</td>
      <td>-2.760660</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14839</td>
      <td>8580</td>
      <td>9580</td>
      <td>71</td>
      <td>201933</td>
      <td>False</td>
      <td>0.266</td>
      <td>1</td>
      <td>0.0363</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.1430</td>
      <td>181.740</td>
      <td>3</td>
      <td>0</td>
      <td>-0.104832</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5255</td>
      <td>16899</td>
      <td>25689</td>
      <td>82</td>
      <td>198853</td>
      <td>False</td>
      <td>0.618</td>
      <td>1</td>
      <td>0.0526</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.1670</td>
      <td>119.949</td>
      <td>4</td>
      <td>0</td>
      <td>-2.277291</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113995</th>
      <td>22486</td>
      <td>66</td>
      <td>53329</td>
      <td>21</td>
      <td>384999</td>
      <td>False</td>
      <td>0.172</td>
      <td>1</td>
      <td>0.0422</td>
      <td>0.928000</td>
      <td>0.0863</td>
      <td>0.0339</td>
      <td>125.995</td>
      <td>5</td>
      <td>113</td>
      <td>-1.386848</td>
    </tr>
    <tr>
      <th>113996</th>
      <td>22486</td>
      <td>66</td>
      <td>65090</td>
      <td>22</td>
      <td>385000</td>
      <td>False</td>
      <td>0.174</td>
      <td>0</td>
      <td>0.0401</td>
      <td>0.976000</td>
      <td>0.1050</td>
      <td>0.0350</td>
      <td>85.239</td>
      <td>4</td>
      <td>113</td>
      <td>-0.012859</td>
    </tr>
    <tr>
      <th>113997</th>
      <td>4952</td>
      <td>5028</td>
      <td>38207</td>
      <td>22</td>
      <td>271466</td>
      <td>False</td>
      <td>0.629</td>
      <td>0</td>
      <td>0.0420</td>
      <td>0.000000</td>
      <td>0.0839</td>
      <td>0.7430</td>
      <td>132.378</td>
      <td>4</td>
      <td>113</td>
      <td>-0.476733</td>
    </tr>
    <tr>
      <th>113998</th>
      <td>18534</td>
      <td>7238</td>
      <td>21507</td>
      <td>41</td>
      <td>283893</td>
      <td>False</td>
      <td>0.587</td>
      <td>1</td>
      <td>0.0297</td>
      <td>0.000000</td>
      <td>0.2700</td>
      <td>0.4130</td>
      <td>135.960</td>
      <td>4</td>
      <td>113</td>
      <td>-3.410587</td>
    </tr>
    <tr>
      <th>113999</th>
      <td>4952</td>
      <td>24357</td>
      <td>5999</td>
      <td>22</td>
      <td>241826</td>
      <td>False</td>
      <td>0.526</td>
      <td>0</td>
      <td>0.0725</td>
      <td>0.000000</td>
      <td>0.0893</td>
      <td>0.7080</td>
      <td>79.198</td>
      <td>4</td>
      <td>113</td>
      <td>-1.585222</td>
    </tr>
  </tbody>
</table>
<p>113999 rows × 16 columns</p>
</div>



### mode

Про mode можно однозначно сказать, что вероятность стать популярным выше пре mode = 0

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1IAAAINCAIAAAC769nGAAAgAElEQVR4AeydC3hU1bm/d7iFXCDcQY8Q8FIEUeEvIqJ1tBwBwYJ6Yr1UG2rbU0Qr+NR71VCpN/AInofaHtQGawFFEKgRkYpBvBIo1xByAVMJiAZIiEzuM7P+DVsWY2bWmsvKnm/Nmt8889g1a+9vre97v63nPTOzMxbDAwRAAARAAARAAARAIAEIWAlQI0oEARAAARAAARAAARBg0D5cBCAAAiAAAiAAAiCQEASgfQnRZhQJAiAAAiAAAiAAAtA+XAMgAAIgAAIgAAIgkBAEoH0J0WYUCQIgAAIgAAIgAALQPlwDIAACIAACIAACIJAQBKB9CdFmFAkCIAACIAACIAAC0D5cAyAAAiAAAiAAAiCQEARM0D6v11tRUXHs2LEaPEAABEAABEAABEAggQkcO3asoqLC6/UG1VgTtK+iosLCAwRAAARAAARAAARA4ASBiooKY7Xv2LFjlmVVVFQksNyjdBDQg8CRIzVz57Y8jxzRIyFkQUPgSNWRuevnzl0/90hV8CsBVwpNY7BrAhCw3ws7duyYsdpXU1NjWVZNTU3QCjEJAiAQOwJuN7OslqfbHbtNsZN+BNyNbmuWZc2y3I3BrwRcKfo1DRkZQkAuRSZ8yCuv0JA2ogwQiAsC+D/mcdEm55OE9jnPGDuAQHACcimC9gWnhlkQAIFoCED7oqFmYAy0z8CmoqQ4IQDti5NGIU0QMIAAtM+AJrZFCdC+tqCINUAgGgLQvmioIQYEQCAaAtC+aKgZGAPtM7CpKClOCED74qRRSBMEDCAA7TOgiW1RArSvLShiDRCIhgC0LxpqiAEBEIiGALQvGmoGxkD7DGwqSooTAtC+OGkU0gQBAwg0N7O8vJZnc7MB1aCEqAk0e5vzSvLySvKavcGvBFwpUbNFIAjICUD75HxwFARAAARAAARAAAQMIQDtM6SRKAMEQAAEQAAEQAAE5ASgfXI+OAoCINB2BJqaWG5uy7Opqe0WxUrxR6DJ05S7LTd3W26TJ/iVgCsl/pqKjOOEALQvThqFNEHAAAK4pcOAJrZFCbiloy0oYg0QiIYAtC8aaogBARCIhgC0LxpqBsZA+wxsKkqKEwLQvjhpFNIEAQMIQPsMaGJblADtawuKWAMEoiEA7YuGGmJAAASiIQDti4aagTHQPgObipKiJtDcyPbMYwV3t/yzuTHqZcIMpNS+Dz/88Nprrz3ttNMsy1q5ciXP2OfzPfbYY/369evcufPYsWNLS0v5oaNHj956661dunTJyMi44447jh8/zg+JBvIKRVGYBwEQaHsC0L62ZxqXK0L74rJtSNoJAlvvZ0vas8XWd88l7dnW+53Yh68plyKLn+fEYM2aNb/73e/eeuutVtr3zDPPZGRkrFq1aseOHZMnTx40aFB9fb2dwIQJEy688MLPP//8o48+Ovvss2+55ZaQickrDBmOE0AABNqMALSvzVDG90LQvvjuH7JvKwJb7z8lfNz8FluOmp9cipzVPs7NX/t8Pl+/fv3mzp1rHz127FhycvLSpUsZY0VFRZZlbd682T707rvvJiUlHTx4kK8TdCCvMGgIJkEABBwhAO1zBGv8LQrti7+eIeM2J9Dc+L33+fy1b0l75z7tlUsRgfbt27fPsqxt27ZxwldcccU999zDGHvllVe6devG55ubm9u3b//WW2/xGT5oaGioOfmoqKiwLKumpoYfxQAEQICGQHMzW7as5YkfZ6NpgC67NnublxUuW1a4TPLjbLhSdOkW8nCIwJ55wd/qs/1vzzyHttVO+z755BPLsr766ite8I033viTn/yEMfbkk0/+4Ac/4POMsd69e7/44ov+M/Y4JyfH+v4D2hdICTMgAAIgAAIgAAI0BArulmlfwd0OZWWm9uHdPocuFywLAiAAAiAAAiDQBgTwbp8NsU0+5PXvh1xs/c/EGARAwFkC+JDXWb5xszo+5I2bViFR5wg01sne7Wusc2hnuRQRfLfPvqXjueeeswuuqalpdUvHli1b7EPvvfeeRrd01FaylQPZ62kt/6ytdKhbWBYE4psAbumI7/61Wfa4paPNUGKh+CXwdb5M+77Od6gySu07fvz4thMPy7Kef/75bdu2ffnll4yxZ555plu3bqtXr965c+eUKVNa/QGXESNGbNq06eOPPz7nnHN0+QMuyzJaN29ZhkMNw7IgEMcEoH1x3Ly2TB3a15Y0sVacEihf0toc/G/mLV/iUFmU2pefn//9+y6s7Oxsxpj955r79u2bnJw8duzYkpISXvzRo0dvueWW9PT0rl27/vznP9fizzUHOp/dOZgfbxsGIGATgPbhSjhBANqHCwEEWAK+2xebrsvFVjWH2kqZrePT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YRIE4omASBv4vEPFQPuUwPL2iAZKqyMYBEwkIFIBE2vVqCbLslauXBnjhNDrGAPHdnFEQKQNfN6hWnTUvqeeemrkyJHp6em9e/eeMmVKcXExL76+vn769Ok9evRIS0u74YYbvv76a35INJBXKIoKc563RzQIcx2cBgKJQwAqQNJraB8JdmwKAiICIm3g86JAxXm5FFmKq0cXPn78+Nzc3MLCwu3bt0+cOHHAgAFut9teatq0af3791+/fv2WLVtGjx49ZsyYkFvIKwwZLj+Bt0c0kIfjKAgkFgGfj1VW1h88WFRUVF9fn1i1U1cbqfY1NjaqpyxSfJ/PV+murHRX+ny+oLucuFJYZSUTHA8ahEkQiCcCIm3g8w4VI5ciGu3zL7WystKyrA8//JAxduzYsY4dO7755pv2CXv27LEs67PPPvM/P3AsrzDw/IhmeHtEg4hWw8kgYDgBzW7pcLlcd99994wZM7p169anT5+FCxe63e6pU6emp6efddZZa9as4e3YtWvXhAkT0tLS+vTpc9tttx0+fNg+9O6771522WUZGRk9evSYNGnS3r177fny8nLLslasWHHllVempKRccMEFn376KV/Nf2BZ1osvvjhhwoTOnTsPGjSI//eNMbZz586rrrqqc+fOPXr0+NWvfnX8+HE7MDs7e8qUKbNmzerVq1eXLl1+/etfc0XLzMycN28eX//CCy/MycmxX/pr3wMPPHDOOeekpKQMGjTo0UcfbWpqss/Jycm58MILX3rppYEDByYlJfF1oh6ItA8/zhY1UgQaQ0CkDXzeoUrlUkSvfWVlZZZl7dq1izG2fv16y7Kqq6s5iwEDBjz//PP8JR80NDTUnHxUVFRYllVTU8OPtuGAt0c0aMO9sBQIxD0Bkfa53Szw6f92YOBRt5vV1Z0CEnjCqWPCkcvl6tKly+zZs0tLS2fPnt2+fftrrrlm4cKFpaWld955Z8+ePWtraxlj1dXVvXv3fvjhh/fs2bN169arr776qquushddvnz5ihUrysrKtm3b9uMf//j888/3er2MMVv7zj333Ly8vJKSkqysrMzMzObm5sBULMvq2bPnSy+9VFJS8uijj7Zv376oqIgx5na7TzvttBtuuGHXrl3r168fNGhQdna2HZ6dnZ2enn7TTTcVFhbm5eX17t37kUcesQ+FqX2zZ8/+5JNPysvL//73v/ft2/fZZ5+1w3NyctLS0iZMmLB169YdO3YEZhvpDLQvUmI4P3EIiLSBzzuEQmvt83q9kyZNuuyyy+ziFy9e3KlTJ38QF1988QMPPOA/Y49zcnKs7z+gfYGUMAMCsSYg0j7LYoHPiRNPpZeaGuQEl+vUCb16tT7h1DHhyOVyXX755fZhj8eTlpZ2++232y8PHTrEP0yYPXv2uHHj+Cr2/ydZUlLCZ+zB4cOH+f+Pamvfyy+/bB/avXu3ZVl79uxpFcIYsyxr2rRpfP6SSy658847GWMLFy7s3r07/37LO++8065dO/vbzNnZ2T169LCVlDH2pz/9KT093dbNMLWPb8cYmzt37kUXXWTP5OTkdOzYsbKy0v8ElTG0T4UeYs0mwPVONHCofK21b9q0aZmZmRUVFXbx4Wsf3u1z6HLBsiCgREA/7Zs+fTqvaMCAAXPmzLFf+nw+y7JWr17NGMvKyurYsWOa38OyLPsj4NLS0ptvvnnQoEFdunRJS0uzLOudd97h7/YVFBTYq1VVVfEvq/Dt7IFlWa+++iqfnDlz5pVXXskYu/fee+2BfejYsWN8hezsbP52I2Ns+/btlmX961//YoyFqX2vv/76mDFj+vbtm5aWlpyc3Lt3b3uXnJycs88+myejPoD2qTPECqYSENken3eocH2176677jrjjDO++OILXnn4H/LykH//11Neof+ZUYx5e0SDKNZECAgYS0CkfYEf0brdLCYf8s6YMYPTbuVM/MtwEyZMuOGGG8q+/7Dfhxs8ePC4cePef//9oqKiwsJCHmK/27dt2zZ78erqasuy8vPz+V580LbaN2jQIP/vvQwdOjTwu32ffvpp+/bt//CHP2zevLm0tPSJJ57IyMiw87G/28dzUx9A+9QZYgVTCYi0gc87VLhcimi+2+fz+e66667TTz+9tLTUv2z7lo7ly5fbk8XFxfxTGP/TWo3lFbY6OdKXvD2iQaQL4nwQMJmASPuIana5XOFo3yOPPDJ48ODAb+YdOXLEsqyNGzfa6X/00UfRaZ/9qa69yOjRo8P8kLfu5Fcb//znP/MPeUeNGnX//ffbS9XU1KSkpARq33PPPXfmmWdy5L/4xS+gfZwGBiAQMwIibeDzDmUilyIa7bvzzjszMjI2bNhw6OSD/wdu2rRpAwYM+OCDD7Zs2XLpiUdILvIKQ4bLT+DtEQ3k4TgKAolFID617+DBg717987KyiooKNi7d+/atWunTp3q8Xi8Xm/Pnj1vu+22srKy9evXX3zxxdFpX69evV555ZWSkpLHH3+8Xbt2u3fvZozV1taedtpp//Vf/7Vr164PPvjgzDPPbHVLxy233LJ79+533nmnb9++Dz30kH0hPfTQQ/369du4cePOnTuvu+669PT0QO1bvXp1hw4dli5dunfv3hdeeKFHjx7QvsT61xDV6kFApA183qE05VJEo33fvxmj5VVubq5dv/3nmrt3756amnr99dcfOnQoJBd5hSHD5Sfw9ogG8nAcBYHEItDQwLKz62fO1OTH2cJ8t48xVlpaev3113fr1i0lJeXcc8+dOXOm/Qfn/vGPfwwZMiQ5OfmCCy7YsGFDdNr3xz/+8eqrr05OTh44cOAbb7zBLwn5H3B5/PHHe/bsmZ6e/qtf/arh5E+Y1dTU3HTTTV27du3fv/+iRYtEf8Dl/vvvt2NvuummefPmxV77GpobsldmZ6/Mlvw4W3Y2y87Gj7PxywED0wiItIHPO1SwXIpotK9tS5VXqLgXb49ooLg+wkHAPAKi73uZV2k4FXFTDOdk+xz77/aFfz7hmeg1IXxsrTkBkTbweYfyl0sRtC8Edt4e0SBEPA6DQOIRgAr49xza508DYxBIHAIibeDzDqGA9imB5e0RDZRWRzAIGEbA52Nud/3Ro/hxNt7YxNQ+n8/nbnS7G92SH2ezb+/Gj7PxSwUDwwiItIHPO1QvtE8JLG+PaKC0OoJBwDACmt3SYRhdDcsRvbOLH2fTsFlIKcYERNrA5x3KB9qnBJa3RzRQWh3BIGAYAWifYQ0NVQ60LxQhHE9cAiJt4PMOoYH2KYHl7RENlFZHMAgYRgDaZ1hDQ5UD7QtFCMcTl4BIG/i8Q2igfUpgeXtEA6XVEQwChhGA9hnW0FDlQPtCEcLxxCUg0gY+7xAaaJ8SWN4e0UBpdQSDgGEEoH2GNTRUOdC+UIRwPHEJiLSBzzuEBtqnBJa3RzRQWh3BIGAYAWifYQ0NVQ60LxQhHE9cAiJt4PMOoYH2KYHl7RENlFZHMAgYRgDaZ1hDQ5UD7QtFCMcTl4BIG/i8Q2igfUpgeXtEA6XVEQwChhGor2dZWfX//d+a/Dhbm9P1/7W3zMzMefPmtfkW8bWgSPvqm+uzlmVlLcuqb64PWtGJK4VlZbH64MeDBmESBOKJgEgb+LxDxUD7lMDy9ogGSqsjGARMJCBSAQNq9de+ysrK2tpaA4pSKcHgXqtgQSwIMMZE2sDnHaIE7VMCy9sjGiitjmAQMJFAdCrg9XkrmiqKG4srmiq8Pq+eYPy1T88MY5xVdL2OcZLYDgRICIi0gc87lBW0Twksb49ooLQ6gkHARAJRqEBZY9nL1S/Pr5pvP1+ufrmssaxN2LhcrrvvvnvGjBndunXr06fPwoUL3W731KlT09PTzzrrrDVr1ti77Nq1a8KECWlpaX369LntttsOHz5sz7vd7ttvvz0tLa1fv37PPfecv/bxD3nLy8sty9q2bZsdUl1dbVlWfn4+Yyw/P9+yrLVr1w4fPrxz585XXXXVN998s2bNmnPPPbdLly633HJLvL9fGEWv26StWAQE9Ccg0gY+71AJ0D4lsLw9ooHS6ggGAcMIRHVLR1ljGRc+/0GbmJ/L5erSpcvs2bNLS0tnz57dvn37a665ZuHChaWlpXfeeWfPnj1ra2urq6t79+798MMP79mzZ+vWrVdfffVVV11ld+bOO+8cMGDA+++/v3PnzmuvvbZLly4zZsywD4WvfaNHj/7444+3bt169tlnu1yucePGbd26dePGjT179nzmmWfi+hIQaR9+nC2u24rk24SASBv4fJvsErgItC+QSQQzvD2iQQRr4VQQMJ5A5Nrn9Xn93+fz176Xq19W/7TX5XJdfvnlNniPx5OWlnb77bfbLw8dOmRZ1meffTZ79uxx48bx5lRUVFiWVVJScvz48U6dOi1btsw+dPTo0ZSUlCi07/3337dXePrppy3L2rdvn/3y17/+9fjx4/m+8TiA9sVj15BzbAiItIHPO5QGtE8JLG+PaKC0OoJBwDACkWtfRVOFv+q1Glc0VSgScrlc06dP54sMGDBgzpw59kufz2dZ1urVq7Oysjp27Jjm97Asa82aNdu3b7cs68svv+Thw4cPj0L7Kisr7RX+8pe/pKam8tUef/zxESNG8JfxOID2xWPXkHNsCIi0gc87lAa0Twksb49ooLQ6gkHAMAKRa19xY3Er1fN/WdxYrEjI/9t4jDH+yay9rGVZK1eunDBhwg033FD2/Yfb7Q5T+7788kvLsrZu3WqvWVlZ2eq7fdXV1fah3NzcjIwMXlFOTs6FF17IX8bjANoXj11DzrEhINIGPu9QGtA+JbC8PaKB0uoIBgHDCESufTF4t4+/PyfSvkceeWTw4MHNzc2tunH8+PGOHTvyD3mrqqpSU1P5atwg6+rqLMt655137PB169ZB+/DdvlbXEl4mIAGRNvB5h5hA+5TA8vaIBkqrIxgEDCMQufbF4Lt9XNRE2nfw4MHevXtnZWUVFBTs3bt37dq1U6dO9Xg8jLFp06ZlZmauX79+165dkydPTk9P56tx7WOMjR49+oc//GFRUdGGDRtGjRoF7YP2GfZvNsqJgoBIG/h8FGuGEwLtC4eS8BzeHtFAGIkDIJCABCLXPsaY03fyclETaR9jrLS09Prrr+/WrVtKSsq55547c+ZMn8/HGDt+/Phtt92Wmprat2/fOXPm+H9k7K99RUVFl156aUpKyvDhw/FuH2MM2peA//aj5FYERNrA51ud31YvoX1KJHl7RAOl1REMAoYRqK9nEyfW/+xnkf44m3N/t88wwLqVI/puX31z/cTFEycunij5cbaJE9nEifhxNt1ainzajIBIG/h8m+30/YWgfd/nEeEr3h7RIML1cDoImE9ApALyyuPiVzrkJSTg0eh6nYCgUHICEhBpA593iAm0Twksb49ooLQ6gkHARAJQARO7Grwm9Do4F8yCAH6T17lrQC62ivuKbI/PK66PcBAwjwBUwLyeiipCr0VkMA8C3BNEA4cQyaXIcmjXWC4rr1AxE1G3+Lzi+ggHAaMIuN0sNbX+3HOLCgvr6+uNKg3FBCMg0j53ozv1ydTUJ1Pdje5gcezElcJSU1sGeICAkQS4J4gGDlUtlyJoXwjsom7x+RDxOAwCCUUgqjt5E4qQYcVKtM+aZVmzLIn2WRazLGifYVcEyjlFgHuCaHDq1DYdQfuUcIq6xeeVVkcwCBhGANpnWENDlQPtC0UIxxOXAPcE0cAhNNA+JbCibvF5pdURDAKGEYD2GdbQUOVA+0IRwvHEJcA9QTRwCA20TwmsqFt8Xml1BIOAYQSgfYY1NFQ50L5QhHA8cQlwTxANHEID7VMCK+oWn1daHcEgYBgBaJ9hDQ1VDrQvFCEcT1wC3BNEA4fQQPuUwIq6xeeVVkcwCBhGANrnfEMty1q5ciVjrLy83LKsbdu2Ob+ncAdonxANDiQ8Ae4JooFDhKB9SmBF3eLzSqsjGAQMI1BXx1yu+htvjPTH2QzD4Gg5XPs8Hs+hQ4eam5sd3U6+uEj76prqXLkuV66rrqku6AonrhTmcrG64MeDBmESBOKJAPcE0cChYqB9SmBF3eLzSqsjGARMJCBSARNrJaiJax/B3gFbotcBSDABAt8R4J4gGjhECtqnBFbULT6vtDqCQcBEAoEq4G50Bz7rm0/9PefAo+5Gt/8bRYEnhEPO5XLdfffdM2bM6NatW58+fRYuXOh2u6dOnZqenn7WWWetWbOGL7Jr164JEyakpaX16dPntttuO3z4sH3o3XffveyyyzIyMnr06DFp0qS9e/fa8/YHrCtWrLjyyitTUlIuuOCCTz/9lK/mP7As689//vOkSZNSUlLOPffcTz/9tKyszOVypaamXnrppXxBxtiqVatGjBiRnJw8aNCgWbNm8bfxSktLf/jDHyYnJw8ZMmTdunVc+/w/5M3Nzc3IyOD7rly50rK++7OsOTk5F1544SuvvNK/f/+0tLQ777zT4/E8++yzffv27d279x/+8AceFcUgsNdRLIIQEDCSAPcE0cChqqF9SmBF3eLzSqsjGARMJBCoAvZf7m31z4mLJ/LqU59MbXXUmmW5cl38hF5zerU6gR+SDFwuV5cuXWbPnl1aWjp79uz27dtfc801CxcuLC0tvfPOO3v27FlbW8sYq66u7t2798MPP7xnz56tW7deffXVV111lb3s8uXLV6xYUVZWtm3bth//+Mfnn3++1+vl36s799xz8/LySkpKsrKyMjMzuaj5p2RZ1n/8x3+88cYbJSUl11133cCBA3/0ox+tXbu2qKho9OjREyZMsE/euHFj165dFy1atG/fvnXr1g0cOHDWrFmMMa/XO2zYsLFjx27fvv3DDz8cMWJEFNqXnp6elZW1e/fuv//97506dRo/fvxvfvOb4uLiv/zlL5Zlff755/4JRzQO7HVE4TgZBAwmwD1BNHCodmifElhRt/i80uoIBgHDCLjdrFev+uHDW/04Wytjs1/GRvsuv/xym7HH40lLS7v99tvtl4cOHbIs67PPPmOMzZ49e9y4cbwVFRUVlmWVlJTwGXtw+PBhy7J27drFte/ll1+2D+3evduyrD179rQKYYxZlvXoo4/a85999pllWa+88or9cunSpZ07d7bHY8eOfeqpp3j4a6+9dtpppzHG3nvvvQ4dOhw8eNA+9O6770ahfampqd9++629wvjx4wcO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UxNjbMuWLe3atXviiSdKSkpyc3NTUlJyc3Pt7mVmZnbp0uXpp58uKSn53//93/bt269bt44xFqb2HThwYO7cudu2bdu3b58dvmnTJntll8uVnp5+//33F594aHixRNprDUtASiDgEAH//5oFHTu0L7RPCWzQVvlPKq2OYBAwkUBEKuDz+fzf5/P/l2vN3q8VP+2VaN+jjz5qs3e73ZZlvfvuu1z7Xn/9dfvQ0aNHU1JS3njjDcbYrbfeevXVV/N23X///UOHDrVfZmZmTpgwgR+66aabrrnmmvC1jwfag0mTJv32t7+1xy6Xa8SIEa1O0OplRL3WKnMkAwJOE/D/r1nQsUMJQPuUwAZtlf+k0uoIBgHDCET+3b7K2gb/f6FajStrG1QISbRv2bJlfOWuXbu++uqrXPu+/PJLfmj48OGzZs1ijI0YMcIe2IdWrVrVsWNHj8fDGMvMzPz973/PQ+bPnz9w4MDwtc/j8TzxxBPDhg3r3r17Wlpahw4dbrzxRns1l8v1y1/+kq+s4UCkffhun4bNQkoxJtDqv2aBLx3KB9qnBDawT61mlFZHMAgYRiDyO3n31wT/hNf+F21/jdI3/AYNGvT8889zxv9+f45/t2/lypV8PiMjw/7E1v7KXVtp35dffmlZ1tatW+2NKisrg3637+mnn+7Zs+drr722ffv2srKySZMmTZkyxQ5xuVwzZszgeWo4EGkffpxNw2YhpRgTaGULgS8dygfapwQ2sE+tZpRWRzAIGEYgcu1z9N2+UaNG3X///TbjmpqalJSUcLTP/lSXMVZVVZWamir6kPe8886zV87MzLQ/1bVf3nzzzfbLuro6y7Leeecde37dunVBte/aa6+944477HO8Xu8555wD7bNp4J8gENcEWtlC4EuHqoP2KYEN7FOrGaXVEQwChhGIXPsc/W7fQw891K9fv40bN+7cufO6665LT08PR/vOO++8999/f9euXZMnTx4wYID9l1P++c9/8ls6Fi1a1OqWjq5duz777LMlJSULFixo37792rVr7caOHj36hz/8YVFR0YYNG0aNGhVU++69997+/ft/8sknRUVFv/zlL7t27QrtM+xfC5STmARa2ULgS4ewQPuUwAb2qdWM0uoIBgHDCESufYwx5+7krampuemmm7p27dq/f/9Fixb5/wEXyYe8b7/99nnnndepU6dR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)

### speechiness

<p>Наибольшая концентрация популярных треков наблюдается вблизи нулевых значений speechiness. Это указывает на слабую зависимость между "разговорностью" трека и его популярностью, хотя есть редкие исключения около отметки 0.4.</p>
<p>Однако, сердние значения наших групп особо не отличаются, что говорит нам о слабой разнице по этому параметру</p>

![image.png](data:image/png;base64,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)

### instrumentalness

instrumentalness: треки с instrumentalness > 0.05 не становились самыми популярными, разница типичных представителей наших групп обльше 0.1, что в наших условиях довлольно много

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1IAAAINCAIAAAC769nGAAAgAElEQVR4Aex9CXgURdp/IUIIOUDCpXIqyqGC7CofKDq6CuKioH64oh9+sLvutwvrX/TZ9cIjuOhqgJVBESGEQxCQIwYihEMIIQoiQkAgNyzCgIQQ5khmkkzmqP+GwqJS3V1d3T0zmaN48mh1ddVb7/t7q3t+XfVWFYDin0BAICAQEAgIBAQCAgGBQAwgAGLARmGiQEAgIBAQCAgEBAICAYEAFLRPdAKBgEBAICAQEAgIBAQCMYGAoH0x4WZhpEBAICAQEAgIBAQCAgFB+0QfEAgIBAQCAgGBgEBAIBATCAjaFxNuFkYKBAQCAgGBgEBAICAQELRP9AGBgEBAICAQEAgIBAQCMYGAoH0x4WZhpEBAICAQEAgIBAQCAgFB+0QfEAgIBAQCAgGBgEBAIBATCEQD7fP5fBaLxW63O8Q/gYBAQCAgEBAICAQEAjGMgN1ut1gsPp9PlsZGA+2zWCxA/BMICAQEAgIBgYBAQCAgELiEgMViiVraZ7fbAQAWiyUU5L6qyjFrVuNfVVUomguzNqqsVbN2zpq1c1aVNaTmxzbqYdYJhDoCAYGAQEAgEMYIoLEwu90etbTP4XAAABwOh6yFAc50OiEAjX9OZ4AlR4I4p9sJpgMwHTjdITU/tlGPhJ4hdBQICAQEAgKB8ECATYqiYZKXbWGAvRDbBETQvgB3JyFOICAQEAgIBAQCAUWATYoE7dMItqB9YrRPY5cRxQUCAgGBgEBAIBAyBATtCyjUgvYJ2hfQDiWECQQEAgIBgYBAIIAICNoXQDBhY0ifiO0TsX0B7VPRKszv9zc0NNSJf1GNgNfrjdYOLOwSCEQoAoL2BdRxgvaJ0b6AdqhoFeZ2u3/66aci8S/aESguLq6pqYnWbizsEghEIgKC9gXUa4L2CdoX0A4VlcJ8Pl9JSUl5ebndbq+trY3q0a6YNq62tvbUqVPFxcVizC8qH2RhVIQiIGhfQB3n8cBNmxr/PJ6Ayo0MYR6fZ1Pppk2lmzy+kJof26hHRt8gtayrqysqKnK5XGSmSEclArW1tUVFRXV1dVFpnTBKIBCJCAjaF4leEzoLBCIYAUT7BBWIYBdyqy58zQ2VKCgQCBECgvaFCGjRjEBAIIAQEFQgdnqC8HXs+FpYGikICNoXUE81NMClSxv/GhoCKjcyhDV4G5YeWrr00NIGb0jNj23UI6NvkFrGIBUwmUxTp04lQYiRdAz6OkY8K8yMXAQE7Quo78SSDrGkI6AdKiqFxSAVELQvKnuyMEogEIkICNoXUK8J2idoX0A7VFQKE7RP1a1+v98TFcvCYtDXqs4VBQQCzYuAoH0BxV/QPkH7AtqholKYTirg88KKXfDkqsb/+gK2CbDJZPrrpX/JyckpKSlvvvmm3++HEFqt1meffbZ9+/bx8fGjRo0qKytDvli6dGm7du2ysrL69OkTFxc3cuTI06dPo1sTJ04cO3YsdtnUqVNNJhO6JEf7li9f/utf/zoxMbFLly5PP/30+fPnUZldu3YBAHJycn71q1+1atVq165dWFTkJnT6OnINFpoLBMIeAUH7AuQijxsWz4G7/yxO6QDilI4A9aloFaOHCpzOhFnd4Epw+S+rGzydGRB8TCZTYmLi1KlTS0pKPv/887Zt26anp0MIx4wZ079///z8/MOHDz/00EN9+vRpuBSwu3Tp0latWt1xxx179+49cODAkCFD7rrrLqQJJ+1bvHhxTk7OiRMnvvvuu2HDhj388MOoOqJ9AwcO3L59+/Hjxy9evBgQA5tXiB5fN6/GonWBQLQj0Jy0b/fu3Y888si1114LAMjKysJQ+/3+t956q2vXrm3atHnggQfwdzaE8OLFi88880xSUlK7du3+8Ic/8Oz/zrYQN2ooUfAyXNWy8QdpMbhM+/bEYvi20+0EYrTPUE+KicqaqcDpTLiyxRXO10j+WjT+BYL5mUym/v37oxE+COGrr77av3//srIyAMCePXuQP6qqquLj49euXQshXLp0KQBg37596FZxcTEA4Pvvv4cQctI+0sc//PADAAC9xxDt27BhA1kg0tOafR3pBgv9BQJhjwCbFIGg6p+Tk/PGG298+eWXFO374IMP2rVrt2HDhh9//HHMmDG9e/fGW3yNGjVq0KBB+/bt++abb/r06fP000+rasi2ULW6eoGCl6/8IGHatxjAgpfV60ZXCUH7osufwbJGGxXweZuM8+EBv5UtYFZ347O9JpPp97//PTZ1w4YNV199NfovebbE7bff/s477yDad/XVV/t8Plylffv2y5Yt46d9Bw4ceOSRR7p3756YmNi2bVsAQGFhIYQQ0b4zZ85gyVGQ0ObrKDBYmCAQCHsE2KQouLQPg0PSPr/f37Vr11mzZqG7drs9Li5u9erVEMKioiIAwA8//IBubdmypUWLFmfPnsVyZBNsC2WraMj0uC+P86FfI5L2rWoJPW4NoiK/qKB9ke/DUFigjQpU7LryWXWF8/0y21thNAAugLTv97///ZgxYzCCU6ZMkcb2OZ3OlJSUZ555Jj8/v7i4eNu2bQCAQ4cOYdpns9mwhChIaPN1FBgsTBAIhD0CbFLUDLTvxIkT+D2I0Lv33ntfeOEFCOHixYvbt2+PIfV4PC1btvzyyy9xDk7U19c7fvlnsVgAAA6HA98NZKJ4TpPfpOUAvnDpb/mln6XiOdraCk7QujYdDJT2+Dxrj61de2xt6A9nW7sWrl0bm0fiGXBYM1XVRgVOrmryiFHM7+Qqg0aYTKYBAwZgIa+99prSJO+6devwJC+a1YUQlpSU4EneV1555c4778Si7rrrLintO3DgAAAArwJZsWIFft2h0T5B+zCAIiEQEAgEA4Gwo3179uwBAPz888/Y2ieffPJ3v/sdhPC99967+eabcT6EsFOnTvPnzydzUDo1NRU0/Rcs2rf/edZv0v7npbop5gQtaF2xRXFDINAcCGijfcEf7UtMTHzppZdKSkpWrVqVkJCwYMECCOHYsWMHDBjwzTffHD58eNSoUdSSjiFDhuzbt+/AgQNDL/1DKG7durVFixafffZZWVnZ22+/nZycLKV9lZWVrVu3fvnll0+cOLFx48abb75Z0L7m6IOiTYFA7CIQnbSv2Ub7qKEI/tG+YAatx27vFpaHJQLaaN/l2D5qScelVR0Biu2bMmXKX/7yl+Tk5GuuuWbatGnkBi7t2rWLj49/6KGH8MIytIFLZmbmDTfcEBcX9+CDD546dQrD/Pbbb3fp0qVdu3YvvfTS888/L6V9EMJVq1b16tUrLi5u2LBh2dnZgvZh9ERCICAQCAECYUf7AjLJSwLHtpAsqSdNxfaRk7z8sX1BDlrXY5euOmKSVxdsMVdJG+2DsHHFLlq6e+WzKpAreTUdm4ZoX8z5TK/Bmn2ttyFRTyAgEOBEgE2KmiG2Dy3pmD17NjLA4XBQSzoOHDiAbm3btq35l3T8Z4Mv4yt5gzyNxdkVjBcTSzqMYxgLEvRQAToEontAdm/5z0IKciNlHvAF7eNBCZfR42tcWSQEAgKBICDQnLSvpqbm0KV/AIAPP/zw0KFDaLrkgw8+aN++/caNG48cOTJ27FhqA5fBgwd///3333777U033RQWG7hACFe3vhzhh1fyLm2lwVlBDlrXoImxooL2GcMvVmrrpALBWfAkaF9Qu51OXwdVJyFcIBDbCDQn7UMr18ilFxMnToQQou2au3TpEhcX98ADD5SWlmIfXbx48emnn05MTExOTv79738fFts1r213ZVUHpn2LAVzbDqutkhCjfSoAqdyO7SPxVMAJw9uCCoShU4KkkvB1kIAVYgUCuhFoTtqnW2lNFdkWahIlU9hVeYXzkad0LL60gYurUqaKNCvIQevSBoOUI0b7ggRslIkVVCDKHMowR/iaAY64JRBoFgTYpChEsX1BtZxtodGms3qxaF9WL175wQxa59XBcDlB+wxDGBMCBBWICTdfMlL4OnZ8LSyNFATYpEjQPjU/fpHAon1fJKjVJ+4HLWidaCO4SUH7gotvtEgXVCBaPKluh/C1OkaihEAgtAgI2mcM70CN9iEtghO0bsxCDbUF7dMAVgwXFVQgdpwvfB07vhaWRgoCgvYZ89TFI01G+z4D8P8u/X12Kbbv4hFj0iOsdoO3YemhpUsPLW3wNoRS9YYGuHRp419DSJsNpYlR1ZagAlHlTqYxwtdMeMRNgUAzICBonzHQ13dpQvuubCd7ifat72JMuqgtEIhCBAQViEKnKpgkfK0AjMgWCDQbAoL2GYN+VRyL9q2KMyZd1BYIRCECggo0i1MBAFlZWSFuWvg6xICL5gQCqggI2qcKEbMANdq3HMC/X/pbHoujfR6fZ1Pppk2lmzw+DxO1AN/0eOCmTY1/npA2G2ArYkecoALN4mtB+5oFdtGoQCDcEBC0z5hHnOeajPaR2zWvBNB5zpj0CKstlnREmMOaSV1B+5oFeK20z+12G9dT+No4hkKCQCCwCAjaZwzPgGzXbEyF8KktaF/4+CKcNQkfKmAymZ5//vmpU6e2b9++c+fO6enpTqdz0qRJiYmJN954Y05ODobx6NGjo0aNSkhI6Ny584QJEy5cuIBubdmy5e67727Xrl2HDh1Gjx59/PhxlH/y5EkAQGZm5n333RcfHz9w4MC9e/diaWQCADB//vxRo0a1adOmd+/e69atw3ePHDly//33t2nTpkOHDn/605/woUQTJ04cO3bs9OnTO3bsmJSU9Oc//xlTtJ49e86ZMwdLGDRoUGpqKrokad8rr7xy0003xcfH9+7d+80332z4ZTFUamrqoEGDFi1a1KtXrxYtWmA5uhPh42vdJoiKAoEoQ0DQPmMODewGLsZ0afbagvY1uwsiQgEZKuB0QulfXd0Vc6R3nU5YW8sqcOWeYspkMiUlJc2YMaOsrGzGjBktW7Z8+OGH09PTy8rKJk+enJKS4nK5IIQ2m61Tp06vv/56cXFxQUHBiBEj7r//fiR0/fr1mZmZ5eXlhw4devTRR2+77TafzwchRLSvX79+mzZtKi0tHTduXM+ePT1yUQgAgJSUlEWLFpWWlr755pstW7YsKiqCEDqdzmuvvfaJJ544evTozp07e/fujc6u/I/wiRMnJiYmPvXUU8eOHdu0aVOnTp2mTZuG9OGkfTNmzNizZ8/Jkyezs7O7dOmSlpaGqqempiYkJIwaNaqgoODHH39UBI77hoyvueuKggIBgUAwEBC0zxiqAdyu2Zgi4VBb0L5w8EL46yBDBQCA0r/f/vaKLW3byhQwma4U6NiRLnDlnmLKZDINHz4c3fZ6vQkJCc8++yy6PHfuHADgu+++gxDOmDFj5MiRWIrFYgEAkGeFo1sXLlwAABw9ehTTvoyMDHSrsLAQAFBcXIyF4AQA4C9/+Qu+/K//+q/JkydDCNPT06+55hqn04lubd68+aqrrqqoqEC0r0OHDoiSQgg//fTTxMRERDc5aR9uDkI4a9asX//61ygnNTW1VatWlZV8p0qSUhTSMr5WKCmyBQICgdAgIGifMZzZo33rOhmTHmG1Be2LMIc1k7oyVEDK+QCAIaF9U6ZMwTD06NFj5syZ6NLv9wMANm7cCCEcN25cq1atEoh/AAA0BVxWVjZ+/PjevXsnJSUlJCQAADZv3oxp3/79+5E0q9UKANi9ezduCycAAJ999hm+fPHFF++77z4I4UsvvYQS6JbdbscSJk6ciIcbIYSHDx8GAPz0008QQk7a98UXX9x1111dunRJSEiIi4vr1Onymyo1NbVPnz5YGeMJGV8bFyokCAQEAgYQELTPAHj/eTc7/s1a0rG2M/R5jTUQSbUF7YskbzWfrjJUQHYONySTvFOnTsVIUJwJB8ONGjXqiSeeKG/6D43D9e3bd+TIkTt27CgqKjp27BiugiZ5Dx06hITbbDYAwK5du3BbOBFY2te7d+8PP/wQCx8wYIA0tm/v3r0tW7Z89913f/jhh7Kysn/84x/t2rVDVVBsH65uPCHja+NChQSBgEDAAAKC9hkAD0KY9xiL9q0EsELmRW+syfCtLWhf+PomnDQLHypgMpl4aN+0adP69u0rjcyrqqoCAOTn5yN0v/nmG320D83qIiFDhw7lnOSt/SW0ccGCBXiSd8iQIS+//DIS5XA44uPjpbRv9uzZN9xwA+4Rf/zjHwXtw2iIhEAg6hEQtM+Yizfd1oT2fQbgxEt/6HC2lQCeXGWsgUiq3eBtmPf9vHnfzwv94Wzz5sF588ThbJHRWyKO9p09e7ZTp07jxo3bv3//8ePHt27dOmnSJK/X6/P5UlJSJkyYUF5evnPnzjvvvFMf7evYsePixYtLS0vffvvtq666qrCwEELocrmuvfba//7v/z569Ghubu4NN9xALel4+umnCwsLN2/e3KVLl9deew35/rXXXuvatWt+fv6RI0cee+yxxMREKe3buHHj1VdfvXr16uPHj8+dO7dDhw6C9kXGkyO0FAgEAgFB+4yhSI32UYezxdhonzEoRe1YQSDiaB+EsKys7PHHH2/fvn18fHy/fv1efPFFv98PIfz666/79+8fFxc3cODAvLw8fbTvk08+GTFiRFxcXK9evdasWYP7AXsDl7fffjslJSUxMfFPf/pTfX09quVwOJ566qnk5OTu3bsvW7ZMaQOXl19+GdV96qmn5syZI2gfxlwkBAJRj4CgfcZcTMX2UbRvdceYiu0zBqWoHSsIhA/tCwfEMVPkVwbt28dfvhlLCl83I/iiaYGALAKC9snCwp1JreRdAeAbl/5WXDqcbXUC3P88LJ4DPZf2u6+vaYwF3HRb43/ra7jbiJiCXp9318ldu07u8oZ2IYvXC3ftavzzxtD6mYjpFVJFBRUgMRG0j0RDpAUCAoFgIyBonzGE2fv24cG/VS1hZpcmUYArAdxyp7G2w662WNIRdi4JS4UE7SPdImgfiYZICwQEAsFGQNA+YwhTo33UmbyY9ikloov5CdpnrDPFSm1B+2LF0xAKX8eOr4WlkYKAoH3GPGUraTKGp5X2rQTRNNsraJ+xzhQrtQUViBVPC9oXO54WlkYOAoL2GfPVV7capX25jxjTIIxqC9oXRs4IY1UE7Qtj5wRYNeHrAAMqxAkEDCMgaJ8xCNd2MEr7VgKYN9aYEuFSW9C+cPFEeOshqEB4+yeQ2glfBxJNIUsgEAgEBO0zhqLx0T4U9hcVzE/QPmOdKVZqCyoQK54Wk7yx42lhaeQgIGifMV9R+/bpiO3Dqz3ctcZUaf7agvY1vw8iQQNB+yLBS4HRUfg6MDgKKQKBwCEgaJ8xLLfd3WSS9zMAn770hw9nw6xONbH/r8ZUaf7abq975rczZ3470+29tE9hqDRyu+HMmY1/7pA2Gyrzoq4dQQWizqWKBglfK0IjbggEmgkBQfuMAf9l9ya0T4nbye7bRxXeOdKYKqK2QCAyEIhiKmAymaZOnYrc0LNnzzlz5kSGS4KmZRT7OmiYCcECgeAiIGifMXyp0T6Kya0EcPdjV07p2PsciyNG/mifMShF7VhBIIqpAEn7KisrXS5XrDhVwc4o9rWCxSJbIBDuCAjaZ8xDdfYmTG4FgP+49IcOZ/vy+iZn8rprmxSmOGLkx/Z5fd79Z/bvP7M/9Iez7d8P9+8Xh7MZ68yhqq2PCvj8PkuDpcRdYmmw+Py+UCmrrR2S9mmrGaWl9fk6SsEQZgkEwgIBQfuMuaH6VBMmRy3pKFlAS88b26Q8Zn5iJS+NlIZrpxMC0PjndGqoJYo2FwI6qEC5uzzDlmG2mtFfhi2j3F0eEP1NJtPzzz8/derU9u3bd+7cOT093el0Tpo0KTEx8cYbb8zJyUGtHD16dNSoUQkJCZ07d54wYcKFCxdQvtPpfPbZZxMSErp27Tp79myS9uFJ3pMnTwIADh06hKrYbDYAwK5duyCEu3btAgBs3br19ttvb9Omzf3333/+/PmcnJx+/folJSU9/fTTkT5eqMPXAXGrECIQEAgoISBonxIyfPlfJCIaV76yi/nCv+adeh8RkFVFU2tWtvZ9kSQzPiFlflHB+SCEYiUvX6eJ9VJaqUC5uxwTPjIREOZnMpmSkpJmzJhRVlY2Y8aMli1bPvzww+np6WVlZZMnT05JSXG5XDabrVOnTq+//npxcXFBQcGIESPuv/9+5MXJkyf36NFjx44dR44ceeSRR5KSkqSxfaq0b+jQod9++21BQUGfPn1MJtPIkSMLCgry8/NTUlI++OCDiO4uWn0d0cYK5QUCEYGAoH3G3LTyKrgSmC/8y3xxjtlqnmdJQ7RvniWtMefCv/CvVJPxCXct3P9XuHNk438jf24XIyhoH4ZCJBgIaKICPr+PHOfDD5TZas6wZRif7TWZTMOHD0faer3ehISEZ599Fl2eO3cOAPDdd9/NmDFj5MgrK64sFgsAoLS0tKampnXr1mvXrkXlL168GB8fr4P27dixA0l4//33AQAnTpxAl3/+858feughlI7Q/2rydYTaKNQWCEQWAoL2GfPXF4mXOZ8s7buUSf5QBWR8wpjGQawtaF8QwY0i0ZqogKXBQj5BVNrSYDEIjMlkmjJlChbSo0ePmTNnoku/3w8A2Lhx47hx41q1apVA/AMA5OTkHD58GABw6tQpXP3222/XQfsqKyuRhCVLlrRt2xZLe/vttwcPHowvIzGhydeRaKDQWSAQcQgI2mfIZeXW/Y2jer/Quyajfb/EIZE/VAEZnzCkcTArC9oXTHSjR7YmKlDiLiGfICpd4i4xiAsZjQchxAF5SCwAICsra9SoUU888UR5039Op5OT9p06dQoAUFBQgGRWVlZSsX02mw3dWrp0abt27bBFqampgwYNwpeRmNDk60g0UOgsEIg4BATtM+Qy6kdIlfaZrWbj4xOGNA5mZUH7golu9MjWRAVCMNqHx+eUaN+0adP69u3r8XgoH9TU1LRq1QpP8lqt1rZt22JpmEHW1tYCADZv3oyqb9++XdA+CklxKRAQCIQMAUH7DEGtg/YZH58wpHEwKwvaF0x0o0e2JtoXgtg+TNSUaN/Zs2c7deo0bty4/fv3Hz9+fOvWrZMmTfJ6vRDCv/zlLz179ty5c+fRo0fHjBmTmJiIpWHaByEcOnToPffcU1RUlJeXN2TIEEH7oqc3C0sEApGGgKB9hjxG0b6PKmZ/98pD373y0EcVs6lb+DKKR/v+cyZb6q7U1F2poT+cLTUVpqaKw9kMdeaQVdZE+yCEwV7Ji4maEu2DEJaVlT3++OPt27ePj4/v16/fiy++6Pf7IYQ1NTUTJkxo27Ztly5dZs6cSU4Zk7SvqKho2LBh8fHxt99+uxjtC1lPEw0JBAQCUgQE7ZNioiEnz5qH+RxPIrpj+zQAJ4rGMAJaaR9ifuR63ibr4mMYyfA3XYevw98ooaFAIKIRELTPkPt4qB5ZJrpX8hqCUlSOGQT0UYGIOKUjZnzIa6g+X/NKF+UEAgIB7QgI2qcdM6IGSeka01UfLt/z6vI9r5qrPqRuxcL4hM/vO3b+2LHzx4zvpkZgrJ70+eCxY41/vjA9skvdhJgqIahA7Lhb+Dp2fC0sjRQEBO0z5CmK2ymt5N1Xuy/ETMiQVXoriyUdepGLrXqCCsSOv4WvY8fXwtJIQUDQPkOeyrZmk8xPifbFSEifoH2GOlPMVBZUIGZcDYWvY8fXwtJIQUDQPkOeIjkffThb0+2aF1gXrLGvcTW4DLUX3pX10T63151bk5tZnZlbk6tvCbDTiY7Eg05neAMktLuEgKACsdMRhK9jx9fC0khBQNA+Q57ip3245BLbEkNNhnFlHbRvY/VGjAxKbKzeqNVEQfu0Ita85QUVaF78Q9m68HUo0RZtCQR4EBC0jwclxTIUZVGa5KWKRSvz00r7pJxPH/MTtE+xg4blDUEFwtItQVFK+DoosAqhAgEDCAjaZwA8CFdbV5OUjpP2ma3mqJzt1UT73F43CR2V1jTbK2ifoU4c8sqCCoQc8mZrUPi62aAXDQsEFBAQtE8BGL5siqzw07419jV8LURSKU20L7cml0KPvMytyeW3XNA+fqzCoaSgAuHghdDoIHwdGpxFKwIBfgQE7ePHSqYkyVTMVvNHFbMPPH//gefvZxzOhqossi2SERfhWW6v++/b/v73bX/nGavLrM6k0CMvM6sz+cFwu+Hf/97453bzVxIlmw0BQQWCCj0AICsrC0J48uRJAMChQ4eC2hxbuKqvdazoEht3szEXdwUCbAQE7WPjo3KXZCqa0lE52qcCVtPbARztaypYXIU7AqpUINwNCG/9MO3zer3nzp3zeDzNqC/b19LoXtUVXeXucnFMXzM6VDQdBQgI2mfIiQesBzSxPVw4KmP7NEEZwNg+Te2Kws2OAJsKNLt6ka4Apn3hYAjD11LOh16PDOZX7i7Hr1AyIQ69DAdfCx0iBQFB+wx5Kt+aT759zFUfLj781uLDb0kPZyOLRetKXp/fd9J28qTtJOeRJDre+7Le8vngyZONf+JwNll8wi2TQQVCrKrJZHr++eenTp3avn37zp07p6enO53OSZMmJSYm3njjjTk5OVifo0ePjho1KiEhoXPnzhMmTLhw4QK6tWXLlrvvvrtdu3YdOnQYPXr08ePHUT6aYM3MzLzvvvvi4+MHDhy4d+9eLI1MAAAWLFgwevTo+Pj4fv367d27t7y83GQytW3bdtiwYVgghHDDhg2DBw+Oi4vr3bv39OnT8TBeWVnZPffcExcX179//+3bt2PaR07yLl26tF27drjdrKwsAAC6TE1NHTRo0OLFi7t3756QkDB58mSv15uWltalS5dOnTq9++67uJaOhJKvdXz1+fw+cpyPfKPGyH74OvAXVQQCUgQE7ZNioiGHfPWwt2vGJVU5X+RGrsgu6WCbI2V+jG99JceIJR1KyIRnvpQKON1O6V+dpw7rL73rdDtrG5Xnq2IAACAASURBVGoZBfAtRsJkMiUlJc2YMaOsrGzGjBktW7Z8+OGH09PTy8rKJk+enJKS4nI17q9us9k6der0+uuvFxcXFxQUjBgx4v7770di169fn5mZWV5efujQoUcfffS2227zXfr4QJSrX79+mzZtKi0tHTduXM+ePTFRI1UCAFx//fVr1qwpLS197LHHevXq9Zvf/Gbr1q1FRUVDhw4dNWoUKpyfn5+cnLxs2bITJ05s3769V69e06dPhxD6fL5bb731gQceOHz48O7duwcPHqyD9iUmJo4bN66wsDA7O7t169YPPfTQ//t//6+kpGTJkiUAgH379pEKa0pLfY2q64jxsDRY8FtUmrA0WDQpJgoLBMIHAfavZMD1FLTPEKTU24e9kneRdZHq3G5ER65IaR+POTpiuimfCdpHARLml1IqAKYD6d9vV/4WG9L2vbbSAqalJlyg48yOVAF8i5EwmUzDhw9HBbxeb0JCwrPPPosuz507BwD47rvvIIQzZswYOXIklmOxWAAApaWlOAclLly4AAA4evQoXk6RkZGBbhUWFgIAiouLqSoQQgDAm2++ifK/++47AMDixYvR5erVq9u0aYPSDzzwwD//+U9cfcWKFddeey2EcNu2bVdfffXZs2fRrS1btuigfW3btq2urkYSHnrooV69eiHyCiHs27fv+++/j9vVmpD6GknQsaKrxF1CvW/JyxJ3iVbdRHmBQDggwPMrGVg9Be0zhCf53uEZ7ct35TPai/TIFYr2hcwcQfsYnSoMb0mpAMXY0GVoaN+UKVMwRD169Jg5cya69Pv9AICNGxvPjBk3blyrVq0SiH8AADQFXFZWNn78+N69eyclJSUkJAAANm/ejGnf/v37kTSr1QoA2L17N24LJwAAa9euRZf//ve/AQC4Vm5uLgDA4XBACDt27NimTRusQps2bQAALpfLbDb37t0bS7Pb7Tpo34ABA7CE//3f//3tb68Q7nvvvfell17Cd7UmpL5GEsRon1YkRfmo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)

### liveness

liveness нам мало о чём говорит, медианы слишком близко друг к другу, а группа самых популярных разбилась на маленькие группы вдоль всей линии, уменьшаясь по мере роста liveness. Можно сделать вывод, что студийное исполнение популярнее живого

![image.png](data:image/png;base64,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pYyxEYC1zCCAtM8MelgXEUAEEAFEwEkECiC2j9CaUCh0YOEvFApRouN6wlO0T5KkbcltTo4ebKs0UYq0D4cBIoAIIAKIQKEgUAAreSVJKisru+2224i377bbbtu6davrhI8o4DXah8zP+YmHtM95zLFFRAARQAQQAREE2POXC2bfvocffnj58uW//OUvibfvl7/85bvf/e6HH37YC8zPg7QvNBMSGQpYxioEkPZZhSTKQQQQAUQAEbAWgVOZU5FLEf+MP3Ipksvn7CZOMCkSXdLxgQ98oKKigtW1oqLi/e9/P5vjVhq20HmtcvlcWbLM2kGD0mAEkPbB+OBdRAARQAQQAbcQmJmbcZKKwKRIlPZdd911wWCQ1TsQCFx33XVsjltp2ELntYpcirg1tpZsu+tG1u6tW7O3bo3i4Wxrzq5Zc3YNezjbmrq9a+r24uFsS3bAoOGIACKACDiGgH2LdhUZDkyKRGnfj3/84//8z/9kG/jFL37xox/9iM1xKw1b6LxW/hm/Y4MJG0IEEAFEABFABBABLyNg3xZ9igwHJkU6aN973/ve22677YcLfytXrnzve99LuOB/Lvwptu1MJmyhMzqwrbRn2708/lA3RAARQAQQAa8hsDu5uyZd80ziGa8phvqYR6BhooElCXanYVIkSvs+D/7de++9dpsByIctBCracQsXc5ifIQYkPDP0+8YnvtH4xDeeGfo9V/33I7//xuvf+Mbr3/j9yOKt3w89840nGr/xROPvh/AJ6+bZKlxP4SUigAggAsWKQEumxQ7KoSgTJkWitE9RtEcyYQudVBIXc7g1Y3FJh1vIY7uIACKACCACmgisS6ybzc06Q0hgUoS0z8pewMUcmkPfpgJI+2wCFsUiAogAIoAIWIKAL+uzknCoy/I67fu///u/kpKSn/3sZ8SEqampH/3oR9dff/3y5cu//vWvx+NxddMW78AWala3sAAu5rBkbhgQgrTPAGhYBRGwHIGtia2Wy0SBiEBxINCYabSQbwCiYFLksrfv+PHjH/7wh//8z/+c0r7777//5ptvPnLkyMmTJ++8887PfvazgG3kFmyhZnULC6C3z63JibTPLeSxXUQAEUAEEAERBNDbJ01MTHz0ox997bXXPve5zxHal0ql3vWud+3du5dQse7u7pKSkqNHj8LMzDu0D2P7RIa+HWWQ9tmBKspE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)

### valence_tempo

<p>Два следующих графика не дают нам той информации, которую мы ищем, тк медианы очень близко друг к другу, и точки почти симметрично распределились по осям ( медианы очень близки к центру картинки)</p>

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1IAAAINCAIAAAC769nGAAAgAElEQVR4AexdCXgURdou5Ai5gAU5VE4vBBVkFRcVHVnl8AJ1cdVdXXQvV5fd6OOtHK7o/nKsJBwSQrgiBAMEksgtSSAKxIABOXKCHMMxhKRnJjOTYdIz079jQaWo7q7p7um5ksozD1TX8dVXb1XXvPPVV1VAYH8MAYYAQ4AhwBBgCDAEGAItAAHQAtrImsgQYAgwBBgCDAGGAEOAISAw2scGAUOAIcAQYAgwBBgCDIEWgQCjfS2im1kjGQIMAYYAQ4AhwBBgCDDax8YAQ4AhwBBgCDAEGAIMgRaBAKN9LaKbWSMZAgwBhgBDgCHAEGAIMNrHxgBDgCHAEGAIMAQYAgyBFoEAo30toptZIxkCDAGGAEOAIcAQYAgw2sfGAEOAIcAQYAgwBBgCDIEWgUBzoH0ej8doNFosFiv7YwgwBBgCDAGGAEOAIdCCEbBYLEaj0ePxSNLY5kD7jEYjYH8MAYYAQ4AhwBBgCDAEGAK/IGA0Gpst7bNYLAAAo9HYgsk9azpDgCEghUBtrXXmTN+nthYl13K1M/NnzsyfWcs1RaJUekBKHr0ES2UIMAQYAiFFANrCLBZLs6V9VqsVAGC1WiVbyCIZAgyBlouA3S4A4PvY7QgEu8sOPgLgI2B3NUWiVHpASh69BEtlCDAEGAIhRYBOiprDIi+9hSEFm1XGEGAIRBQCUjSN0b6I6iKmDEOAIaAvAnRSxGifvmgzaQwBhkAkIcBoXyT1BtOFIcAQCAECjPaFAGRWBUOAIRCRCDDaF5HdwpRiCDAEgodAi6Z9brfbyf6aNQKNjY1erzd47w+THN0IMNoX3f3HtGcIMARUI9ByaZ/NZisvLy9jf80dgRMnTrhcLtVvBivQEhBgtK8l9DJrI0OAIYAh0EJpn9vtLi8vP3nyZENDQ7O2drXoxjU0NFgslurq6oqKCrmjKbF3gQVbHgKM9rW8PmctZgi0cARaKO1zOp1lZWUNDQ0tvPtbQvMdDkdZWZnT6WwJjWVtVIcAzwsbNvg+PI8K8h5+Q+WGDZUbeE9TJEqlB6Tk0UuwVIYAQ4AhEFIEWjTtY1QgpGMtTJVBis/6Okzws2oZAgwBhgBDIIIQYLQvgjqDqRIMBBjtCwaqTCZDgCHAEGAIRCMCjPZFY6+p1tlgMCQlJaku1iwKMNrXLLoxOI1obBSWLvV9GhtRBY3uxqX7ly7dv7TR3RSJUukBKXn0EiyVIcAQYAiEFAFG+0IKd7gqY7SPLfKGa+xFdL1sS0dEdw9TjiHAENAfAUb79Mc0AiWqpX1er5fHnNwjsEXKVWLWPuVYtbicjPa1uC5nDWYItHQEGO1TMwI8bsFUKBzP9P3rcaspKZvXYDD885e/Dh06dOnSZdKkSfB4YY7jXnzxxU6dOsXGxo4ZM6aqqgqKWLp0aceOHdevX3/jjTfGxMSMGjXq1KlTMGnChAnjxo1DNSUlJRkMBviI076MjIw777wzISGhe/fuzz///Pnz52GewsJCAMCmTZt+/etft23btrCwEImK6gCjfVHdfcFVntG+4OLLpDMEGAIRhwCjfYq75FS2sL6nsBJc+qzvKZzKVlxYNqPBYEhISEhKSqqoqFixYkVcXFxaWpogCGPHjh0wYEBRUdGBAwdGjx594403Nv7ifrR06dK2bdveddddu3fv3rdv3913333vvfdC6Qpp3+LFizdt2nTs2LE9e/bcc889jzzyCCwOad+gQYO2bdt29OjRuro6WaWjKiG4tI93CeWzhZKJvn95dih0VI0MQRAY7Yu2HmP6MgQYAgEiEE7at3Pnzscff/yaa64BAKxfvx61xOv1Tp48uUePHu3bt3/ooYeQoUsQhLq6uj/84Q+JiYkdO3b885//bLPZUCm5gGQLVVOBU9nCylZNnM9H/lr5PgEzP4PBMGDAAHSB2LvvvjtgwICqqioAwK5du2CjamtrY2NjV69eLQjC0qVLAQDFxcUwqby8HADw/fffC4KgkPbhQO3duxcAAGGEtC8nJwfP0AzCqvtaeZtL3xYyWzeNiszWQunbykuznOFHgNG+8PcB04AhwBAIKQKSpAhpAFAoGIFNmzZ9+OGH69atI2jfZ5991rFjx5ycnB9//HHs2LH9+vVD/vhjxowZPHhwcXHxt99+e+ONNz7//PN+FZNsoToq4HFfYedDBr+VrYT1vQJc7TUYDC+//DJqRU5OTps2beC/bnfTOvIdd9zxn//8B9K+Nm3a4HdOdOrUadmyZcpp3759+x5//PFevXolJCTExcUBAI4cOSIIAqR9p0+fRso0j4C6vlbe5tK3mwhf05AAjPkphzD8ORntC38fMA0YAgyBkCIgSYqQBsGlfU3VYNY+r9fbo0ePmTNnwlSLxRITE7Nq1SpBEMrKygAAe/fuhUmbN29u1arVmTNnkBzJgGQL1VEBU6H0Fzz8sjcF5AOnI+17+eWXx44di0B47bXXxL59dru9S5cuf/jDH4qKisrLy7du3QoA2L9/P6J9ZrMZSWgeAXV9rbDNvOsKOx9O+zJbs9VehSiGPxujfeHvA6YBQ4AhEFIEJEkR0iAMtO/YsWOIiEA9HnjggX//+9+CICxevLhTp05IOZ7nW7duvW7dOhSDAhcvXrRe/jMajQAAq9WKUgVBUEcFjmfSaN/xTFyy2rDBYBg4cCAq9d5778kt8q5ZswYt8sJVXUEQKioq0CLvO++8M3ToUCTq3nvvFdO+ffv2AQDQLpAvv/wSoQ2tfYz2IQBpgfLZtCFRPptWlqVFDgI8L6xe7ftg+9Z5D7/68OrVh1dru5xNJC8crQ3C5rNwNIPVyRBgCOiPQMTRvl27dgEAzp49i9r6zDPP/P73vxcE4dNPP7355ptRvCAIXbt2/eKLL/AYGJ46dSq48i8g2hdka19CQsIbb7xRUVGRmZkZHx+fmpoqCMK4ceMGDhz47bffHjhwYMyYMcSWjrvvvru4uHjfvn3DfvmDrd6yZUurVq2WL19eVVU1ZcqUDh06iGlfTU1Nu3bt3n777WPHjuXm5t58882M9onHj/+Ykok02lcy0b8EloMhECQEgrP5LEjKMrEMAYZAiBFonrRPZ2vfJd8+YkvHL7s69PDte+211/7xj3906NDhV7/61QcffIAf4NKxY8fY2NjRo0ejfS3wAJfs7Ozrr78+Jibm4YcfPnnyJBo0U6ZM6d69e8eOHd94442JEyeKaZ8gCJmZmX379o2Jibnnnnvy8vIY7UPoqQgwa58KsFjWECIQtM1nIWwDq4ohwBAIIgIRR/t0WeTFAZNsobpFXkHw7diFW3ebvLh028mr6to0SPvwBrIwHQHVfU0XB1OZb58SlCI/TzNb5A3m5rPI70ymIUOAIaAEAUlShAqGwbcPbumYNWsWVMJqtRJbOvbt2weTtm7dGqItHbA+cumkV+Cnt/y8kQI/SBnhTgkw2kcBRzIpKLTv55MVI3Anr6tBKPmnkD/K96+rQRINFnkFAs1sS0cw3VGuwI09MAQoCFy0CTueFDbc7vv3ov9D1iiSWFIwEAgn7bPZbPt/+QMAfP755/v374frlZ999lmnTp1yc3MPHjw4btw44gCXIUOGfP/99999991NN90UogNcEPBBcJRmtA+hG6RAsGgfZH6Rc27fjnGku+GOpitbgoRt1IttZrQvmJvPor6vWQNCg8DmoeREtLlpo2FoVGC10BEIJ+2DW0fxrRcTJkwQBAEe19y9e/eYmJiHHnqosrIStaGuru75559PSEjo0KHDyy+/HLrjmpEGLBBtCASR9gmC76yWSLilQ8z5oDcCY3704drMaB+z9tG7m6UGGwEx54MTEWN+wUZejfxw0j41emrPK9nC4FIB7cqykvoj0Pz72tVA/rxu8kAFbLWXNqSaGe0L5uYzGowsjSEgCL71XHzmIcJstTdiBokkKULahci3D9UXjIBkC5s/FQgGlNEi0+MW6qsF82Hfvx538+/rkn/SZtuSf0ZLv+mgp1qnomZG+4K5+UyH3mEimjcCO56kTUQ7nmzerY+i1kmSIqQ/o30IChaIEgQsZULtXvzjrDlcVlaG7veLkmaoUTN/FG22zR+lRlY05xUvMPldWmp+tA8yv/U9m4bEen02n0XzyGC6hwSBDbc3jTrC1LcS+HZ4sL/IQIDRvsjoB6aFLgiIOJ9Qu9d5dm/ZD/nNmfYxa58gCGLOB7946MyvWdI+QfBdFG4qFI5n+v71NF3trctLxoQwBKQRYNY+aVwiLpbRvojrEqaQRgQ8btzIh8I+2rd3s9Naq1Fs5Bdjvn2anYoaG4WlS32fxkbUz43uxqX7ly7dv7TR3RSJUukBKXn0EiyVIdBcEND8GjYXAKKlHYz2RUtPMT39IVBfjageHrhE+3b+xV/5aE5v4Tt5mZkhmgcv0735IKDN6N582h8dLWG0Lzr6qXloCQBYv359sNpiPoyzPRS+RPs2j9S53gg5ugW1Ssz8vr5NKJnoO1+Gd6FczTPAnIqaZ7+yVoUbAQ0nwIuZH93RItxNbIH1M9rXAjs9bE0OLu0LpbWv9G0hcg5qRv2J5uiNd/xylyC45GGd2dp3p0gz/tNs7eN5YcMG34fnETy8h99QuWFD5Qbe0xSJUukBKXn0EiyVIRCpCIh/SSo8B1TthvpIBaC56sVoX3Pt2Uhsl1ra53KpMVOFzLcvAq9lw3s7wtXDVdUrrNmpqLlu6dALWCanxSIg5nxwj5RC5tdicYuGhjPaF+ZeMhgMEydOTEpK6tSpU7du3dLS0ux2+0svvZSQkHDDDTds2rQJ6Xfo0KExY8bEx8d369bthRdeuHDhAkzavHnzfffd17Fjx86dOz/22GNHjx6F8cePHwcAZGdnP/jgg7GxsYMGDdq9ezeShgcAAF988cWYMWPat2/fr1+/NWvWoNSDBw+OGDGiffv2nTt3/tvf/oauRZkwYcK4ceM++uijq6++OjEx8ZVXXkEUrU+fPrNnz0YSBg8ePHXqVPiI07533nnnpptuio2N7dev36RJkxov+9RPnTp18ODBixYt6tu3b6tWrZAcRYEQ7OTlXVfY+fBzCjJbh385NcLVU9SLmjKJl5Za8k5eTRCyQgyBSwiwXWLNeigw2ne5e+12QfxxOi8nCxKpdrvQgF14Ly7eVFg2ZDAYEhMTp02bVlVVNW3atNatWz/yyCNpaWlVVVWvvvpqly5dHA6HIAhms7lr167vv/9+eXl5aWnpyJEjR4wYAYWuXbs2Ozu7urp6//79TzzxxO233+7xeARBgLTvlltu2bBhQ2Vl5fjx4/v06cNji1lIJwBAly5dFi1aVFlZOWnSpNatW5eVlQm+Ftuvueaap59++tChQ/n5+f369YO35/0sfMKECQkJCc8+++zhw4c3bNjQtWvXDz74AApUSPumTZu2a9eu48eP5+Xlde/effr06bD41KlT4+Pjx4wZU1pa+uOPPyIllQZEzE/nc/vKZ9POpipv4rtKFdbXR1B39ZQ2IwLyiZmfX6ciZu2LgH7TQQV2YI0OIGIi2JlQGBjNL8ho3+U+BUAQfx599HKyIMTFSWQwGJoyXH01maEpTTZkMBiGDx8Ok91ud3x8/Isvvggfz507BwDYs2ePIAjTpk0bNarp3F2j0QgAwG8rhkUuXLgAADh06BCifenp6TDpyJEjAIDy8nL4iP8LAPjHP/6BYn7zm9+8+uqrgiCkpaX96le/stvtMGnjxo1XXXWVyWSCtK9z586QkgqCsGDBgoSEBEg3FdI+VJ0gCDNnzrzzzjthzNSpU9u2bVtTU4NnUBcO6i0dJRNptK9kojpVdfcR1Fc9dY2JgNxqnYoY7YuATgtUhVPZwhXHU/cUTmUHKrOFl2cnwDfrAcBo3+XuFXM+AISQ0L7XXnvtshJC7969Z8yYAR+9Xi8AIDc3VxCE8ePHt23bNh77AwDAJeCqqqrnnnuuX79+iYmJ8fHxAICNGzci2ldSUgKlcRwHANi5cyeqCwUAAMuXL0ePr7/++oMPPigIwhtvvAEDMMlisSAJEyZMQOZGQRAOHDgAADhx4oQgCApp31dffXXvvfd27949Pj4+Jiama9eusJapU6feeOONSJnAAzpfzqajOS0YTng6qhc49JEvgdG+yO8juoansq/YveRb2W/l+zDmR8eNnsqsfXR8ojyV0b7LHSheorXbhZAs8iYlJV1WguRMyBluzJgxTz/9dPWVf9AO179//1GjRm3fvr2srOzw4cOoCFzk3b9/PxRuNpsBAIWFhaguFNCX9vXr1+/zzz9HwgcOHCj27du9e3fr1q0/+eSTvXv3VlVVffzxxx07doRFoG8fKh54QGfap3n3ANGSIDnhBUksoXyzeWS0L6q70uO+ws7X5GjbSljfi11Por1vmW+fduyioCSjfWHuJIPBoIT2ffDBB/379xd75tXW1gIAioqKYDO+/fZbbbQPrupCIcOGDVO4yNtw2bUxNTUVLfLefffdb7996bgQq9UaGxsrpn2zZs26/vrrEfR/+ctfoob26WVOC0QOfSkzGEZE1FXNLMBoX1R3qKmQ5nFhkviJG9XNDanybCdvSOEOaWWM9oUUbnFlCmnfmTNnunbtOn78+JKSkqNHj27ZsuWll15yu90ej6dLly4vvPBCdXV1fn7+0KFDtdG+q6++evHixZWVlVOmTLnqqquOHDkiCILD4bjmmmt+97vfHTp0qKCg4Prrrye2dDz//PNHjhzZuHFj9+7d33vvPdi69957r0ePHkVFRQcPHnzyyScTEhLEtC83N7dNmzarVq06evRoSkpK586do4b26eU8p1mOko0LursMigdu84hpbBTmzfN9Lm8kFwSh0d047/t5876fp+1yNpG85oFURLbieCaN9h3PjEilo0cpMfNjp7dET+9RNGW0jwJOKJIU0j5BEKqqqp566qlOnTrFxsbecsstr7/+utfrFQThm2++GTBgQExMzKBBg3bs2KGN9s2fP3/kyJExMTF9+/bNyspCLacf4DJlypQuXbokJCT87W9/u3jxIixltVqfffbZDh069OrVa9myZXIHuLz99tuw7LPPPjt79uyooX2BWOkQrILguzyjaU3q8rnKKEZuR7CY88Ei4i2r+m4QxjVnYYZAhCDArH3B7gh0AnzJPwUXdmxFsOtl8oOJAKN9wUQ3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vmjxFHZCAAIQgAAEIAABTQSQfb7xLVy4cOjQoYmJiWPHjt23b5/PTJmZmSNGjEhMTBw1atRHH33kMw+RuhII6KZ33nlnwoQJl7o+d9xxhz9X6moklUuSFNBTCqX09HSLxXLXXXcpMQSMJKDGU/X19U888cSgQYMSEhKuvfZafv2MdJDclho3zZkz5zvf+U6vXr2uueaaZ555pq2tzXg747zFnTt3/vznP7/qqqssFsvGjRv90dixY8eNN96YkJAwfPjw9957z1+2cMUj+3yQzMjISEhIWL58+eHDhx9//PFLL720pqbGI9/u3bt79Ogxc+bMwsLCv/zlLz179szPz/fIw1ddCahx0/33379o0aKDBw8eOXLkkUceGTBgQEVFha5WUbk3ATWekkuVlZVdffXVP/zhD5F93hgNiFHjqY6OjptuuunOO+/84osvysrKPvvss9zcXANsowmFgBo3paamJiYmpqamlpWVbdu27aqrrvrjH/+o1EDAGAJbt279f//v/23YsEEg+0pLS/v06fPss88WFhYuWLCgR48e//jHP3Q1D9nnA+/YsWOnTJkiJzgcjsGDB7/++use+e67776f/exnSuTNN9/8u9/9TvlKwAACatzkbsa5c+f69ev3/vvvu0cSNoCASk+dO3fuBz/4QUpKysMPP4zsM8Av3k2o8dTixYuHDRt29uxZ7+LEGENAjZumTJny4x//WLHn2WefHT9+vPKVgMEEBLJv6tSpI0eOVOz59a9/PXHiROWrHgFknyfVjo6OHj16uM/HPvTQQ5MmTfLIl5ycPGfOHCXyf//3f7/3ve8pXwnoTUClm9zNaGpq6tWr15YtW9wjCetNQL2n/vd///fuu++WJAnZp7dTfNav0lM//elPH3jggccff/zKK68cOXLkq6++eu7cOZ8VEqkHAZVuSk1NHTBggLys5ejRo9ddd92rr76qhz3UqYaAQPb98Ic/fPrpp5VKli9f3r9/f+WrHgFknyfVEydOWCyWPXv2KAnPP//82LFjla9yoGfPnmlpaUrkokWLrrzySuUrAb0JqHSTuxm///3vhw0bxgIXdyYGhFV66vPPP7/66qtPnjyJ7DPAKT6bUOkpeUHzo48+un///oyMjIEDB06fPt1nhUTqQUClmyRJmjdvXs+ePS+55BKLxTJ58mQ9jKFOlQQEsu/aa6997bXXlHo++ugji8XS2tqqxIQ9gOzzRKryoUL2eYIz9rtKNylGvf7660lJSYcOHVJiCBhDQI2nmpqavvWtb23dulU2idk+Y1zj0YoaT0mSdO211yYnJyszfG+99dagQYM8quKrfgRUumnHjh3f/OY3ly1blpeXt2HDhuTk5L/+9a/6WUXNYgLIPjGfCKeqnELnJW9k/aTSTbKRb7755oABA7KzsyNrc3y2rsZTBw8etFgsPc5/urk+PXr0KCkpiU9oEem1Gk9JkvSjH/3ojjvuUCzcunWrxWLp6OhQYgjoSkClmyZMmPCnP/1JsWTVqlW9e/d2OBxKDAEjCQhkHy95jXSE37bGjh375JNPyskOh+Pqq6/2uaXj5z//uVLFuHHj2NKh0DAmoMZNkiTNmDGjf//+X375pTFW0Yo3gYCeamtry3f73HXXXT/+8Y/z8/MRE94wdY0J6ClJkv785z8PHTpUERBz58696qqrdLWKyj0IqHHT97///alTpyoF09LSevfurczRKvEEjCEgkH1Tp04dNWqUYsZvfvMbtnQoNIwLZGRkJCYmrlixorCw8Le//e2ll15aXV0tSdKDDz744osvynbs3r37kksumTVr1pEjR6ZNm8YBLsa553xLatz0xhtvJCQkrFu3rur8p7m5+XwF/NcgAmo85W4KL3ndaRgZVuOp8vLyfv36Pfnkk0VFRR9++OGVV175yiuvGGkkbalx07Rp0/r165eenl5aWvrxxx8PHz78vvvuA53BBJqbmw+6PhaLZfbs2QcPHjx+/LgkSS+++OKDDz4oGyMf4PL8888fOXJk0aJFHOBisI8uNLdgwYIhQ4YkJCSMHTt27969csKtt9768MMPK5kyMzO/853vJCQkjBw5kgNLFSxGBgK6aejQoZaLP9OmTTPSQtqSCQT0lDsoZJ87DYPDajy1Z8+em2++OTExcdiwYezkNdhBcnMB3dTZ2Tl9+vThw4f36tUrOTn5iSeeqK+vj4ip8dzojh07Lv77Y5ElxMMPP3zrrbcqZHbs2HHDDTckJCQMGzaM45oVLAQgAAEIQAACEIAABDQRYCevJnwUhgAEIAABCEAAAtFCANkXLZ7CTghAAAIQgAAEIKCJALJPEz4KQwACEIAABCAAgWghgOyLFk9hJwQgAAEIQAACENBEANmnCR+FIQABCEAAAhCAQLQQQPZFi6ewEwIQgAAEIAABCGgigOzThI/CEIAABCAAAQhAIFoIIPuixVPYCQEImILA0KFD58yZYwpTMAICEIBAkASQfUECIzsEIBDfBJB98e1/eg+B6CaA7Itu/2E9BCBgMAFkn8HAaQ4CEAgjAWRfGGFSFQQgEGUEli5detVVVzkcDsXuSZMm/fd//3dJScmkSZOuvPLKvn373nTTTZ988omSwV321dfXP/bYY5dffnm/fv1uv/323NxcOdu0adNGjx69cuXKoUOH9u/f/9e//nVTU5Oc5HA4ZsyYMXz48ISEhOTk5FdeeUWOLy8v/9WvfjVgwICkpKRJkyaVlZUpLRKAAAQgEC4CyL5wkaQeCEAg+gjU1dUlJCRs375dNv306dPy19zc3CVLluTn59tstr/85S+9evU6fvy4nMdd9v3bv/3bL37xi+zsbJvN9txzz1122WWnT5/+Z7Zp06Z94xvfuPfee/Pz83ft2jVo0KCXXnpJLj516tSkpKQVK1aUlJR8/vnny5YtkyTp7Nmz3/3udx999NG8vLzCwsL7779/xIgRHR0dchH+DQEIQCBcBJB94SJJPRCAQFQSuOuuux599FHZ9KVLlw4ePNh98k+OHzly5IIFC+SwIvs+//zz/v37t7e3K90ePnz40qVL//l12rRpffr0UWb4nn/++ZtvvlmSpKampsTERFnqKaUkSVq1atWIESOcTqcc2dHR0bt3723btrnnIQwBCEBAOwFkn3aG1AABCEQxgczMzAEDBsjq7Uc/+tGzzz4rSVJzc/Nzzz133XXXDRgwoG/fvt27d3/++eflTiqyb+HChd27d+/r9unevfvUqVP/mW3atGnXX3+9AmX27Nnf/va3JUnat2+fxWIpLS1VkuTAn/70px49erjV1Ldbt25vv/22Rza+QgACENBIANmnESDFIQCB6CbQ1tbWv3//9evXl5eXd+vW7cCBA5Ik/e53vxs2bNiGDRvy8vKKi4tHjx799NNPy/1UZN8bb7xx9dVXF1/8OXny5D+zyWv7FC5z5swZOnSoJEl5eXk+Zd/kyZPHjh17cU3FDQ0NSg0EIAABCISFALIvLBipBAIQiGICjzzyyL333jtjxozrrrtO7saoUaP++te/yuHm5uYBAwZ4y76PP/64R48ePvde+JN9bW1tvXv39n7J+8477yQlJTU2NkYxREyHAASigQCyLxq8hI0QgICeBD755JPExMQRI0b87W9/k9u55557brjhhoMHD+bm5v7iF7/o16+ft+xzOp0TJkwYPXr0tm3bysrKdu/e/dJLL2VnZ/+zBn+yT5Kk6dOnJyUlvf/++yUlJV9++WVKSookSS0tLddee+1tt922a9eu0tLSHTt2PPXUU3a7Xc9OUzcEIBCPBJB98eh1+gwBCLgTcDgcV111lcViOXr0qBxfVlZ2++239+7dOzk5eeHChbfeequ37JO3aDz11FODBw/u2bNncnLyAw88UF5e/s8aBLLP4XC88sorQ4cO7dmz55AhQ1577TW5xaqqqoceeujyyy9PTEwcNmzY448/zuSfu48IQwACYSGA7AsLRiqBAAQgAAEIQAACZieA7DO7h7APAhCAAAQgAAEIhIUAsi8sGKkEAhCAAAQgAAEImJ0Ass/sHsI+CEAAAhCAAAQgEBYCyL6wYKQSCEAAAhCAAAQgYHYCyD6zewj7IAABCEAAAhCAQFgIIPvCgpFKIAABCEAAAhCAgNkJIPvM7iHsgwAEIAABCEAAAmEhgOwLC0YqgQAEIAABCEAAAmYngOwzu4ewDwIQgAAEIAABCISFALIvLBipBAIQgAAEIAABCJidALLP7B7CPghAAAIQgAAEIBAWAsi+sGCkEghAAAIQgAAEIGB2Av8fgBCiMQGLFlwAAAAASUVORK5CYII=)

![image.png](data:image/png;base64,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dNFF//Iv/4LtmhmZDBwo2isej999990TJ04cPnz4pEmTZs+ezV8Gw14ZsAvRheT35fP5li9fLtYnnGFNTc0999wzatSoCRMmPPnkk9iumSBs7Sk1e9XV1d1xxx0XX3zxyJEjv/GNbzz11FNs3z5BEMzYC2GftRaENBAAARAAARAAARCwKQGEfTY1DNQCARAAARAAARAAAWsJIOyzliekgQAIgAAIgAAIgIBNCSDss6lhoBYIgAAIgAAIgAAIWEsAYZ+1PCENBEAABEAABEAABGxKAGGfTQ0DtUAABEAABEAABEDAWgII+6zlCWkgAAIgAAIgAAIgYFMCCPtsahioBQIgAAIgAAIgAALWEkDYZy1PSAMBEAABEAABEAABmxJA2GdTw0AtEACBzBOYNm3aE088kfl+0SMIgAAIZIYAwr7McEYvIAACDiCAsM8BRoKKIAACJggg7DMBD01BAARcRODhhx/m349ZXV1dWlo6Y8aMMWPGfOlLX/rRj37U3NwsDnfatGnz5s174oknxo0b96UvfenNN9/s6up65JFHLrjggq9//euFhYViNfGFtu+///5f/dVfjRw58qabbiotLWXANm7c+K1vfWvEiBGTJk166aWXWDkOQAAEQCB9BBD2pY8tJIMACDiJQHt7+y233DJ79uzw4F9LS8vEiRN/9atfnThxoqSk5K677rrzzjvF8UybNu3CCy98/vnnA4HA888/P2zYsHvuuefNN98MBAKPPfbYJZdcEovFBEEQw75rrrnmo48+Onbs2P333//Vr371zJkzgiAUFxcPHTr0N7/5TUVFxfLly0eNGrV8+XInwYKuIAACziSAsM+ZdoPWIAACaSDA3+R9/vnn7777btZJMBj0+XwVFRX/G89Nmzbt9ttvF0/19/ePGTPmxz/+sfgxHA77fL59+/axsG/t2rXiqdbW1lGjRq1bt04QhIceeuiuu+5iwp966qlvfetb7CMOQAAEQCBNBBD2pQksxIIACDiPAB/2fe973xs+fPgY7s/n84k3cKdNm/b444+z4eXk5PzP//yP+DGZTPp8voKCAhb21dbWsprXX3/9c889JwjCDTfcIB6Ip/Lz84cPH97f389q4gAEQAAE0kEAYV86qEImCICAIwnwYd+MGTMefPDByvP/urq6xNU+/oHfSZMmvfzyy2zAPp8vLy8PYR8DggMQAAH7EEDYZx9bQBMQAIEsE7jrrrvmzZsnKvFf//VfV199dV9fn1wnPjoUBIEO+8S7uoIgRCKR0aNHq93knTx5srwjlIAACICAtQQQ9lnLE9JAAAQcTGD27NlTpkyprq5ubm4+ffr0xIkTv/e973322WdVVVXbtm175JFHxPuwusK+yZMnb9++vbS0dObMmTk5Ob29vYIgHDp0iD3SsWLFCjzS4eAvDVQHAUcRQNjnKHNBWRAAgXQSqKiouPnmm0eNGuXz+aqrqwOBwKxZs8aNGzdq1Ki//Mu//PnPf55MJvXe5N2yZcvkyZNHjBgxderUo0ePMvXFDVyGDx+ek5Pzhz/8gZXjAARAAATSRwBhX/rYQjIIgICnCYgbuLS1tXmaAgYPAiBgJwII++xkDegCAiDgIgII+1xkTAwFBFxCAGGfSwyJYYAACNiNAMI+u1kE+oAACCDsw3cABEAABEAABEAABDxBAGGfJ8yMQYIACIAACIAACIAAwj58B0AABEAABEAABEDAEwQQ9nnCzBgkCIAACIAACIAACCDsw3cABEAABEAABEAABDxBAGGfJ8yMQYIACIAACIAACIAAwj58B0AABEAABEAABEDAEwQQ9nnCzBgkCIAACIAACIAACCDsw3cABEAABEAABEAABDxBAGGfJ8yMQYIACIAACIAACIDA/w/OPSZoQ3+RaAAAAABJRU5ErkJggg==)

### time_signature

time_signature: Самые популярные треки расположенны на 3й и 4й позиции, больше всего шанс у треков с 5ой позицией.

![image.png](data:image/png;base64,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wPYpLzESBAEFCVhtn3rpZW374vG4ejSQEQiAQIAEYPsChImhQAAEWBBYsmSJ9dm+5eXlmqYZX/KWl5dPmjRpw4YN3/ve98aNG/fzn/+8t7f3rbfeKiwszM/Pf+SRRxKJhBFif3//Y489Nn369Ly8vOLi4u3btxNCP3HixK233pqfn5+Xlzdr1qxNmzbpum77kvf111+fOXPmuHHjbr/99hdeeGHSpEnGgGVlZVddddU//vGPwsLCiRMnLlq0qLu729j073//+4Ybbpg0adLkyZPnz59vPqzTmt2cOXP+N5fN5i5YsMB83nRhYeGf//zn+++//4ILLjAad+3adeONN55//vkzZ8585JFHotEoITVsAgEQCBUB2L5QyY1kQUAFAp2dndddd93SpUsjZ/62bt1qtX1jx46dN2/evn37du7cedFFF918881333334cOHN2zYkJOTs3btWgPBAw88cP3113/yySfHjh17/vnnc3Nza2pq0tGZP3/+vHnzDh48ePz48Q0bNuzcudNm+z799NNzzz33+eefr66u/utf/zp58mSr7ZswYcIdd9xx6NChTz755JJLLnnyySeNid5999333nuvtra2srLyZz/72ezZs5PJpK7re/bs0TRt69atkUikvb09o+2bOHHiX/7yl2PDf+PHj1+9enVNTc1nn332k5/85Je//GW6vNAOAiAQNgKwfWFTHPmCgAoErGe/rGfdjDN/5mmzBx98MC8vr6enx8j5lltuefDBB3Vd//rrr8eMGXPy5EmTxdy5c3//+9+bq7aF2bNn//GPf7Q1WuddtGjR/PnzzR3uvfdeq+3Ly8szz/A98cQT11xzjbmnuXD69GlN0w4dOqTruvNLXmu+uq7bzvbdfvvt5ji//vWvly1bZq7u2rXr3HPP7evrM1uwAAIgEGYCsH1hVh+5g4CsBKw2yGq/ysvL8/LyzKyeeuqpWbNmmauLFy9euHChrusbN27UNG285e+88867++67zT1tC2+88cZ55513/fXXP/XUUwcOHDC2Wuf98Y9//Kc//cns9dJLL1ltnzWGF1988fLLLzf2rKmp+cUvfnH55ZdfcMEF48eP1zTN+Pp4tLbv6aefNqcuKirKyckxM8vLy9M07ciRI+YOWAABEAgzAdi+MKuP3EFAVgIE22f6rf/lZlxXZya5ZMmSBQsW6Lq+du3aMWPGHD16tNbyF4lEzD2dCw0NDa+++urChQvHjh378ssv277kJdu+q666yhxw9erVhYWFxuoVV1xx8803b9269ciRI1VVVZqmVVRUuJ7tu+mmm37zm9+Yg5SUlFiv7Vu9erW56fvf//4jjzxiSSu1iFs9TD5YAIGQE4DtC/kbAOmDgJQE5s2bt2LFCiN061k345YOM6V0tq+6ulrTtE8++cTc0/vCypUrZ8+ebbN9ixYtuvXWW81B7rvvPtN92mIwbV9bW5s1hl27dpm27+TJk5qm7d271xzwf5cn3nXXXcZqIpEoKChIZ/vuueeeuXPnmh2xAAIgAAJWArB9VhpYBgEQkIPA0qVLr7766vr6+tOnT2/bts16S4fpt/6Xic1ymWf7dF2/9957L7vssvfee6+urm737t3PPvvsxo0b0yX/6KOPfvjhh3V1df/973+vueYa4+tgq900bul44YUXampqXnvttYsuuig/P98YzRaDafuSyeRFF11033331dbWbtu27eqrrzZt3+Dg4Lhx455++umWlpbOzk5d11977bW8vLyNGzd+9dVXS5cunThxYjrbd+DAgXHjxpWWllZWVtbU1Kxfv760tDRdXmgHARAIGwHYvrApjnxBQAUC1dXV11577bhx4zRNc/6Ai5mhzXJZbd/AwMBTTz112WWXjR07dtq0aQsXLjx48KDZ0bawYsWK73znO7m5uVOnTr3//vvb2tpsZ/t0XX/99ddnzJhh/IDL008/fckllxiD2GIwbZ+u61u2bPnBD36Qm5t75ZVX7tixw7R9uq6/8cYbl1566bnnnmv8gMvAwMBDDz00efLkiy++eNWqVbZbOqxf8ho3As+bN2/ChAnjx4+/8sorn3nmGVs6WAUBEAgtAdi+0EqPxEEABGgReOCBB2688UZao2NcEAABEMiWAGxftuTQDwRAAAQsBJ5//vn9+/fX1ta+/PLLY8eOfeONNywbsQgCIAACQhCA7RNCBgQBAiAgAoGf/vSn5k+fmAsevyS96667pk6dev7558+aNevVV18VIR3EAAIgAAI2ArB9NiBYBQEQCC+BpqYm20+f1NbWGs/JCC8UZA4CIKAQAdg+hcREKiAAAiAAAiAAAiCQngBsX3o22AICIAACIAACIAACChGA7VNITKQCAiAAAiAAAiAAAukJwPalZ4MtIAACIAACIAACIKAQAdg+hcREKiAAAiAAAiAAAiCQngBsX3o22AICIAACIAACIAACChGA7VNITKQCAiAAAiAAAiAAAukJwPalZ4MtIAACIAACIAACIKAQAdg+hcREKiAAAiAAAiAAAiCQnsD/A6nyOkaYxR7zAAAAAElFTkSuQmCC)

## Анализ категориальных признаков


```python
categorical_features
```




    ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']



Уберём track_id


```python
categorical_features.pop(0)
```




    'track_id'



### Топ исполнители

Выведем топ исполнителей с самыми лучшими треками


```python
popular_artists = origin_data.groupby('artists')['popularity'].mean()
best = pd.DataFrame(popular_artists.sort_values().tail(20))

plt.bar(list(best.index), list(best.popularity))
plt.xlabel('artist')
plt.ylabel('popularity')
plt.xticks(rotation=90)
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
     [Text(0, 0, 'Future;Drake;Tems'),
      Text(1, 0, 'The Weeknd;Gesaffelstein'),
      Text(2, 0, 'Beach Weather'),
      Text(3, 0, 'Elley Duhé'),
      Text(4, 0, 'Shakira;Rauw Alejandro'),
      Text(5, 0, 'Bad Bunny;Rauw Alejandro'),
      Text(6, 0, 'Ruth B.'),
      Text(7, 0, 'Bad Bunny;Tony Dize'),
      Text(8, 0, 'Luar La L'),
      Text(9, 0, 'Rauw Alejandro;Lyanno;Brray'),
      Text(10, 0, 'Drake;21 Savage'),
      Text(11, 0, 'Rema;Selena Gomez'),
      Text(12, 0, 'Harry Styles'),
      Text(13, 0, 'Beyoncé'),
      Text(14, 0, 'Joji'),
      Text(15, 0, 'Bad Bunny;Bomba Estéreo'),
      Text(16, 0, 'Bad Bunny;Chencho Corleone'),
      Text(17, 0, 'Manuel Turizo'),
      Text(18, 0, 'Bizarrap;Quevedo'),
      Text(19, 0, 'Sam Smith;Kim Petras')])




    
![png](spotify_files/spotify_85_1.png)
    


В некоторых треках присутствует несколько исполнителей, это нужно отразить в датасете. Вдруг колаборации имеют большую популярность, чем обычные треки. Или дуэты более популярны, чем соло исполнения

### Новая числовая характеристика - количество исполнителей


```python
working_data['artists_cnt'] = 0
for i in origin_data.index:
  working_data.loc[i, "artists_cnt"] = origin_data['artists'][i].count(';') + 1
working_data['artists_cnt']
```




    0         1
    1         1
    2         2
    3         1
    4         1
             ..
    113995    1
    113996    1
    113997    1
    113998    1
    113999    1
    Name: artists_cnt, Length: 113999, dtype: int64



Сделаем график для этого показателя


```python
popular = working_data[working_data['popularity'] > 90]
unpopular = working_data[working_data['popularity'] < 40]
medium = working_data[working_data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

perfect_artists_cnt = popular['artists_cnt'].mean()

plt.figure(figsize=(10, 6))

plt.scatter(popular['artists_cnt'], popular['popularity'], c='orange', label='popular')
plt.axvline(x=perfect_artists_cnt, color='red', linestyle='--', label='mean popular')

plt.scatter(medium['artists_cnt'], medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium['artists_cnt'].mean(), color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['artists_cnt'], unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular['artists_cnt'].mean(), color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('artists_cnt')
plt.ylabel('popularity')

plt.show()

print(f'Идеальное число для коллаборации - {perfect_artists_cnt}')
```


    
![png](spotify_files/spotify_90_0.png)
    


    Идеальное число для коллаборации - 1.411764705882353


<p>Медианы сливаются - значит, что наш критерий особо не выделяет популярные песни, но можно сказать, что популярные треки имеют от 1 до 3 исполнителей</p>


### Преобразуем категориальную характеристику artists в числовую, через среднее арифметическое популярности их песен

<p>Стоит выделить популярных и не популярных певцов, через среднее арифмитическое их песен</p>


```python
artists = {}
for i in origin_data.index:
  list_artists = origin_data['artists'][i].split(';')
  for artist in list_artists:
    if artist in artists.keys():
      artists[artist] = [artists[artist][0] + origin_data['popularity'][i], artists[artist][1] + 1]
    else:
      artists.update( {artist : [origin_data['popularity'][i], 1]} )

artists_avg = { a: s[0] / s[1] for a, s in artists.items() }
artists_avg
```




    {'Gen Hoshino': np.float64(58.0),
     'Ben Woodward': np.float64(42.92307692307692),
     'Ingrid Michaelson': np.float64(49.083333333333336),
     'ZAYN': np.float64(66.66666666666667),
     'Kina Grannis': np.float64(51.09090909090909),
     'Chord Overstreet': np.float64(42.916666666666664),
     'Tyrone Wells': np.float64(48.25),
     'A Great Big World': np.float64(54.875),
     'Christina Aguilera': np.float64(50.4),
     'Jason Mraz': np.float64(30.035714285714285),
     'Colbie Caillat': np.float64(71.0),
     'Ross Copperman': np.float64(50.75),
     'Zack Tabudlo': np.float64(60.578947368421055),
     'Dan Berk': np.float64(52.333333333333336),
     'Anna Hamilton': np.float64(53.333333333333336),
     'Deepend': np.float64(54.166666666666664),
     'Landon Pigg': np.float64(56.5),
     'Andrew Foy': np.float64(34.65853658536585),
     'Renee Foy': np.float64(38.17857142857143),
     'Boyce Avenue': np.float64(54.08888888888889),
     'Bea Miller': np.float64(66.88888888888889),
     'Jennel Garcia': np.float64(57.5),
     'Brandi Carlile': np.float64(12.17142857142857),
     'Sam Smith': np.float64(15.46875),
     'KT Tunstall': np.float64(13.925925925925926),
     'Eddie Vedder': np.float64(50.27777777777778),
     'Lucius': np.float64(4.2),
     'Highland Peak': np.float64(45.666666666666664),
     'Motohiro Hata': np.float64(56.875),
     'Andrew Belle': np.float64(57.421052631578945),
     'Ron Pope': np.float64(59.333333333333336),
     'Adam Christopher': np.float64(48.875),
     'Aron Wright': np.float64(44.6),
     'Sara Bareilles': np.float64(63.22222222222222),
     'Kurt Cobain': np.float64(53.57142857142857),
     'Tim Halperin': np.float64(48.5),
     'Canyon City': np.float64(51.45454545454545),
     'Aaron Espe': np.float64(62.0),
     'Tyler Ward': np.float64(48.25),
     'Five For Fighting': np.float64(63.25),
     'Bailey Jehl': np.float64(52.42857142857143),
     'Nusrat Fateh Ali Khan': np.float64(42.21590909090909),
     'Drew Holcomb & The Neighbors': np.float64(60.333333333333336),
     'Gabrielle Aplin': np.float64(27.9375),
     'The Civil Wars': np.float64(15.15),
     'Callum J Wright': np.float64(52.0),
     'Fifth Harmony': np.float64(61.75),
     'JP Cooper': np.float64(36.07142857142857),
     'Mone Kamishiraishi': np.float64(32.25),
     'John Adams': np.float64(51.875),
     'Kitri': np.float64(47.0),
     'Augustana': np.float64(63.0),
     'Matthew Perryman Jones': np.float64(53.0),
     'Ray LaMontagne': np.float64(57.75),
     'Sierra Ferrell': np.float64(67.0),
     'Meg Birch': np.float64(43.0),
     'Catherine Feeny': np.float64(54.0),
     'Joshua Hyslop': np.float64(56.285714285714285),
     'JJ Heller': np.float64(49.05882352941177),
     'Howie Day': np.float64(58.0),
     'Ben Rector': np.float64(56.666666666666664),
     'Matt Nathanson': np.float64(57.333333333333336),
     'Rachael Yamagata': np.float64(44.5),
     'Parachute': np.float64(55.75),
     'Susie Suh': np.float64(55.0),
     'Robot Koch': np.float64(32.5),
     'Lindsey Stirling': np.float64(47.0),
     'Eden Elf': np.float64(51.0),
     'Ren Avel': np.float64(51.0),
     'Rachel Grae': np.float64(51.0),
     'Cary Brothers': np.float64(51.0),
     'The Mayries': np.float64(51.1),
     'Joseph Sullinger': np.float64(38.46153846153846),
     'Yonnyboii': np.float64(50.0),
     'Tobey Rosen': np.float64(52.0),
     'Aqualung': np.float64(37.0),
     'Karis': np.float64(51.0),
     'Ray Lorraine': np.float64(51.0),
     'Agustín Amigó': np.float64(38.0),
     'Nylonwings': np.float64(37.25),
     'Amy Stroup': np.float64(46.0),
     'Connie Talbot': np.float64(63.0),
     'Caleb Santos': np.float64(68.0),
     'Viva Music Publishing Inc.': np.float64(68.0),
     'Priscilla Ahn': np.float64(50.666666666666664),
     'Jon Bryant': np.float64(53.0),
     'Donovan Woods': np.float64(49.0),
     'Angelina Cruz': np.float64(38.0),
     'Stephen Speaks': np.float64(64.75),
     'The Weepies': np.float64(57.25),
     'Deb Talan': np.float64(57.25),
     'Steve Tannen': np.float64(57.25),
     'Freddie King': np.float64(33.8),
     'Taj Mahal': np.float64(5.8),
     "Keb' Mo'": np.float64(0.0),
     'Gerald Albright': np.float64(0.0),
     'Thomas Daniel': np.float64(53.75),
     'Masayoshi Yamazaki': np.float64(44.8),
     'Todd Carey': np.float64(38.5),
     'Ariza': np.float64(30.0),
     'Chuck Leavell': np.float64(30.0),
     'YUZU': np.float64(49.5),
     'Jonah Baker': np.float64(44.46666666666667),
     'Billkin': np.float64(58.0),
     'Anna Nalick': np.float64(65.0),
     'Us The Duo': np.float64(49.44444444444444),
     'Tenille Townes': np.float64(57.0),
     'Megan Davies': np.float64(57.0),
     'Megan Nicole': np.float64(61.0),
     'Amos Lee': np.float64(62.0),
     'Meghan Trainor': np.float64(36.25),
     'Days N Daze': np.float64(29.625),
     'Mindy Gledhill': np.float64(53.0),
     'Dave Barnes': np.float64(24.5),
     'Get Dead': np.float64(30.5),
     'Albert King': np.float64(19.333333333333332),
     'Aoi Teshima': np.float64(48.0),
     'Emma Heesters': np.float64(54.0),
     'Jae Hall': np.float64(54.0),
     'A Fine Frenzy': np.float64(57.0),
     'Roses & Frey': np.float64(53.5),
     'Aidan Hawken': np.float64(48.0),
     'The Rescues': np.float64(46.0),
     'Ivan & Alyosha': np.float64(55.0),
     'Austin Plaine': np.float64(42.5),
     'Plamina': np.float64(55.0),
     'Imaginary Future': np.float64(55.0),
     'Phillip LaRue': np.float64(49.5),
     'krissy & ericka': np.float64(64.0),
     'Takehara Pistol': np.float64(30.0),
     'Allman Brown': np.float64(48.333333333333336),
     'Liz Lawrence': np.float64(50.0),
     'Eddy Tyler': np.float64(39.81818181818182),
     'Joy Williams': np.float64(62.0),
     'Lusaint': np.float64(54.5),
     'Sarah Hyland': np.float64(54.0),
     "Mother's Daughter": np.float64(47.5),
     'Beck Pete': np.float64(47.5),
     'The Perishers': np.float64(48.25),
     'Madysyn': np.float64(52.0),
     'Matt Johnson': np.float64(50.333333333333336),
     'Jaclyn Davies': np.float64(54.0),
     'Greg Laswell': np.float64(51.666666666666664),
     'Sara Phillips': np.float64(26.0),
     'Joe Brooks': np.float64(43.5),
     'Masaharu Fukuyama': np.float64(32.108108108108105),
     'James TW': np.float64(54.0),
     'Ciaran Lavery': np.float64(51.5),
     'John Frusciante': np.float64(48.0),
     'Sara Farell': np.float64(44.55555555555556),
     'AJR': np.float64(29.5625),
     'Rodrigo y Gabriela': np.float64(50.0),
     'Front Porch Step': np.float64(27.4),
     'Eric Hutchinson': np.float64(43.5),
     'Brett Dennen': np.float64(51.0),
     'Rail Yard Ghosts': np.float64(30.5),
     'The Bridge City Sinners': np.float64(29.666666666666668),
     'Frank Turner': np.float64(27.772727272727273),
     'Matt Costa': np.float64(0.0),
     'Ichiko Aoba': np.float64(55.25),
     'Meiko': np.float64(43.5),
     'Erin McCarley': np.float64(40.0),
     'Isabella Celander': np.float64(56.5),
     'Amber Leigh Irish': np.float64(51.75),
     'Jess Penner': np.float64(33.0),
     'Daniel Robinson': np.float64(55.5),
     'Yosui Inoue': np.float64(27.833333333333332),
     'Tamio Okuda': np.float64(34.5),
     'Celine': np.float64(35.0),
     'Blame Jones': np.float64(48.5),
     'Matt White': np.float64(60.0),
     'Madi Diaz': np.float64(37.0),
     'Tristan Prettyman': np.float64(49.5),
     'Mark Oblea': np.float64(32.0),
     'Hanare Gumi': np.float64(29.214285714285715),
     'Benedict Cumberbatch': np.float64(45.0),
     'RENT STRIKE': np.float64(28.666666666666668),
     'Mischief Brew': np.float64(31.333333333333332),
     'Grace George': np.float64(54.0),
     'Jack Savoretti': np.float64(50.0),
     'Richard Walters': np.float64(48.0),
     'Kaiak': np.float64(49.5),
     'Viktoria Tolstoy': np.float64(50.0),
     'Bailey Rushlow': np.float64(52.0),
     'Wayne Mack': np.float64(57.0),
     'Aimyon': np.float64(54.0),
     'Sungha Jung': np.float64(38.75),
     'Zee Avi': np.float64(36.5),
     'Joshua Radin': np.float64(48.8),
     'Bailey Pelkman': np.float64(50.5),
     'Randy Rektor': np.float64(50.5),
     'Tres Hermanas': np.float64(32.0),
     'Dave Van Ronk': np.float64(54.0),
     'Josh Klinghoffer': np.float64(31.0),
     'Freddie Boatright': np.float64(32.0),
     'AcousticTrench': np.float64(37.2),
     'Lee DeWyze': np.float64(47.0),
     'Clementine Duo': np.float64(55.0),
     'Brandon Chase': np.float64(47.666666666666664),
     'Austin Basham': np.float64(60.0),
     'Eastmountainsouth': np.float64(52.0),
     'Peter Bradley Adams': np.float64(53.5),
     'Suy Galvez': np.float64(28.0),
     "Howlin' Wolf": np.float64(60.666666666666664),
     'Buddy Guy': np.float64(47.666666666666664),
     'Bonnie Raitt': np.float64(59.666666666666664),
     'Mance Lipscomb': np.float64(31.0),
     'Violent Femmes': np.float64(39.8),
     'Drei Rana': np.float64(29.5),
     'Selena Marie': np.float64(30.0),
     'Harley Poe': np.float64(27.333333333333332),
     'Trevor Hall': np.float64(57.5),
     'Marieme': np.float64(54.0),
     'Colin & Caroline': np.float64(44.5),
     'Ysabelle Cuevas': np.float64(52.5),
     'Ethan Loukas': np.float64(34.0),
     'Janine Teñoso': np.float64(29.0),
     'Penny and Sparrow': np.float64(56.75),
     'Sonny Boy Williamson II': np.float64(45.0),
     'Shannon & Keast': np.float64(45.0),
     'Mia Rollo': np.float64(29.0),
     'AJJ': np.float64(31.75),
     'SANNA NORTH': np.float64(29.0),
     'Pigeon Pit': np.float64(27.25),
     'Kris Allen': np.float64(50.0),
     'Lúc': np.float64(50.5),
     'Edy Hafler': np.float64(43.6),
     'Mat Kearney': np.float64(51.5),
     'Simon Samaeng': np.float64(46.0),
     'Sakura Fujiwara': np.float64(27.666666666666668),
     'Eddie van der Meer': np.float64(32.0),
     'Maris Racal': np.float64(27.5),
     'Apes of the State': np.float64(28.5),
     'Gail Blanco': np.float64(26.75),
     'Emilie Mover': np.float64(41.0),
     'William Fitzsimmons': np.float64(55.0),
     'Vivid Color': np.float64(52.0),
     'Albert Collins': np.float64(54.0),
     'Jon McLaughlin': np.float64(52.0),
     'EJ De Perio': np.float64(28.0),
     'Beyond The Guitar': np.float64(31.5),
     "Hannah's Yard": np.float64(50.5),
     'Vendredi': np.float64(26.5),
     'Son&Dad': np.float64(33.0),
     'Chris Cresswell': np.float64(25.333333333333332),
     'Chuck Ragan': np.float64(26.666666666666668),
     'Tim Vantol': np.float64(28.0),
     'Chad Hates George': np.float64(27.0),
     'Acoustic Guitar Collective': np.float64(41.0),
     'Jules Larson': np.float64(31.0),
     'Rosi Golan': np.float64(42.0),
     'Cat Power': np.float64(44.0),
     'Will Moore': np.float64(38.25),
     'C.W. Stoneking': np.float64(47.0),
     'Johnnyswim': np.float64(27.6),
     'Drew Holcomb': np.float64(56.0),
     'Ellie Holcomb': np.float64(56.0),
     'Marc Broussard': np.float64(27.0),
     'Juicy Karkass': np.float64(29.0),
     'Mindy Smith': np.float64(0.0),
     'Anya Marina': np.float64(32.0),
     'SafetySuit': np.float64(52.0),
     'Hannah Trigwell': np.float64(48.0),
     'Futuristic': np.float64(47.0),
     'Filip Nordin': np.float64(38.25),
     'John Legend': np.float64(14.088888888888889),
     'Melvin Taylor': np.float64(53.5),
     'Lucky Peterson': np.float64(53.5),
     'Titus Williams': np.float64(53.5),
     'Ray "Killer" Allison': np.float64(53.5),
     'Luther Allison': np.float64(46.5),
     'We The Heathens': np.float64(27.0),
     'Ramshackle Glory': np.float64(27.666666666666668),
     'Eric Lumiere': np.float64(25.466666666666665),
     'Carly Lyman': np.float64(48.0),
     'Jason Lux': np.float64(48.0),
     'Michelle Featherstone': np.float64(35.0),
     'Big Mama Thornton': np.float64(57.0),
     'Neulore': np.float64(40.0),
     'Nathan Angelo': np.float64(45.0),
     'Andrew Ripp': np.float64(43.333333333333336),
     'Covers Culture': np.float64(26.0),
     'Acoustic Covers Culture': np.float64(26.0),
     'Lounge Covers Culture Of Popular Songs': np.float64(26.0),
     'Keisha White': np.float64(26.5),
     'Beans on Toast': np.float64(26.0),
     'Not Half Bad': np.float64(27.0),
     'Defiance, Ohio': np.float64(29.0),
     'NORMANDY': np.float64(27.0),
     'Lynn Miller': np.float64(27.0),
     'Parker Jenkins': np.float64(43.0),
     'Sunset & Highland': np.float64(54.0),
     'Shikao Suga': np.float64(27.0),
     'Guus Dielissen': np.float64(39.857142857142854),
     'Naotaro Moriyama': np.float64(26.5),
     'Postcards & Polaroids': np.float64(45.0),
     'Sad Boy': np.float64(45.0),
     'Yu Takahashi': np.float64(27.0),
     'Unkle Bob': np.float64(41.0),
     'The Cameron Collective': np.float64(36.0),
     'Kurei': np.float64(26.0),
     'Little Walter': np.float64(59.0),
     'Matt Pless': np.float64(26.0),
     'Jay Filson': np.float64(26.0),
     'Ankle Grease': np.float64(26.0),
     'John Elliott': np.float64(33.0),
     'Taylor Manns': np.float64(57.0),
     'Zak Manley': np.float64(58.0),
     'Linn Brikell': np.float64(45.0),
     'Rosie Thomas': np.float64(51.0),
     'The Shins': np.float64(3.923076923076923),
     'Sufjan Stevens': np.float64(62.26315789473684),
     'Josh Ottum': np.float64(51.0),
     'Celestial Conscience': np.float64(39.8),
     'Erato': np.float64(54.0),
     'Charlotte Almgren': np.float64(54.0),
     'Caj Morgan': np.float64(46.0),
     'Kazuyoshi Saito': np.float64(26.5),
     'Brendan James': np.float64(51.0),
     'Calibretto 13': np.float64(26.0),
     'Laura Jane Grace': np.float64(26.0),
     'Derek Trucks': np.float64(29.0),
     'Grace Petrie': np.float64(26.0),
     'Salty Snacks': np.float64(38.0),
     'Alex Goot': np.float64(43.0),
     'Michael Logen': np.float64(33.0),
     'Ice Seguerra': np.float64(25.0),
     'Jimmy Rogers': np.float64(50.0),
     'Steve Pulvers': np.float64(24.25),
     'Stephan Baulig': np.float64(37.0),
     'Rory Block': np.float64(25.5),
     'Bobbylene': np.float64(25.0),
     'Joey Cape': np.float64(25.0),
     "Cannon's Jug Stompers": np.float64(28.0),
     'ELLE': np.float64(25.0),
     'J.J. Grey': np.float64(26.0),
     'Chrissy Dave': np.float64(25.0),
     'J-Que Beenz': np.float64(45.0),
     'Roberto Diana': np.float64(47.0),
     'Muniesa': np.float64(47.0),
     'Bootstraps': np.float64(46.0),
     'Graham Colton': np.float64(54.0),
     'Stephanie Briggs': np.float64(50.0),
     'Correatown': np.float64(38.0),
     'Aberola': np.float64(26.0),
     'She/Her/hers': np.float64(25.0),
     'Corporate Hearts': np.float64(26.0),
     'Dan Andriano in the Emergency Room': np.float64(25.0),
     'Limoblaze': np.float64(34.416666666666664),
     'Lecrae': np.float64(48.2),
     'Happi': np.float64(44.5),
     'Criolo': np.float64(35.01162790697674),
     'Rael': np.float64(45.0),
     'BaianaSystem': np.float64(30.816666666666666),
     'Plastilina Mosh': np.float64(25.18421052631579),
     'BNegão': np.float64(31.375),
     'MC Hariel': np.float64(45.84615384615385),
     'Liniker': np.float64(39.09090909090909),
     'Maria Vilani': np.float64(42.0),
     'Jaques Morelenbaum': np.float64(52.666666666666664),
     'Jorge Drexler': np.float64(26.327272727272728),
     'Sebastián Prada': np.float64(38.0),
     'Emicida': np.float64(36.94444444444444),
     'Mano Brown': np.float64(45.666666666666664),
     'Liz Reis': np.float64(36.0),
     'Evandro Fióti': np.float64(37.0),
     'Juanafé': np.float64(27.42105263157895),
     'Milton Nascimento': np.float64(34.27272727272727),
     'Samuca e a Selva': np.float64(22.869565217391305),
     'Jackie Mittoo': np.float64(26.842105263157894),
     'Pompi': np.float64(25.0),
     'Menahan Street Band': np.float64(45.142857142857146),
     'Los Pirañas': np.float64(22.333333333333332),
     'Orquestra Afrosinfônica': np.float64(29.166666666666668),
     'Makavelli': np.float64(32.0),
     'Jaymitta': np.float64(35.0),
     'Buguinha Dub': np.float64(25.75),
     'Curumin': np.float64(34.0),
     'Edgar': np.float64(25.5),
     'Los Amigos Invisibles': np.float64(25.6),
     'Gal Costa': np.float64(40.125),
     'Metá Metá': np.float64(23.333333333333332),
     'Cássia Eller': np.float64(42.2),
     'Samba de Lata de Tijuaçú': np.float64(34.0),
     'VANDAL': np.float64(34.0),
     'Mandrill': np.float64(21.2),
     'Luedji Luna': np.float64(31.0),
     'As Ganhadeiras de Itapuã': np.float64(33.0),
     'Hugh Masekela': np.float64(5.5625),
     'Manu Dibango': np.float64(23.615384615384617),
     'Ernest Ranglin': np.float64(9.4),
     'Calle 13': np.float64(31.074074074074073),
     'El Michels Affair': np.float64(33.714285714285715),
     'Ali Farka Touré': np.float64(11.25),
     'Toumani Diabaté': np.float64(10.875),
     'Antonio Carlos & Jocafi': np.float64(30.25),
     'Ferraz': np.float64(34.0),
     'Afrocidade': np.float64(22.09090909090909),
     'OQuadro': np.float64(23.333333333333332),
     'Dimak': np.float64(30.0),
     'Nomade Orquestra': np.float64(20.166666666666668),
     'Russo Passapusso': np.float64(30.0),
     'Nação Zumbi': np.float64(40.35294117647059),
     'Romperayo': np.float64(20.5),
     'William Onyeabor': np.float64(23.833333333333332),
     'Carolaine': np.float64(31.0),
     'Bixiga 70': np.float64(25.0),
     'Orquestra Brasileira de Música Jamaicana': np.float64(22.285714285714285),
     'Tony Allen': np.float64(24.272727272727273),
     'Gugu Shezi': np.float64(0.0),
     'CalledOut Music': np.float64(41.666666666666664),
     'Bang Data': np.float64(42.0),
     'QUITAPENAS': np.float64(22.0),
     'Morbo y Mambo': np.float64(20.75),
     'Letta Mbulu': np.float64(32.333333333333336),
     'Damon Albarn': np.float64(37.5),
     'Subhira': np.float64(27.0),
     'La Misa Negra': np.float64(20.833333333333332),
     'Demian Rodríguez': np.float64(28.0),
     'Newen Afrobeat': np.float64(20.307692307692307),
     'C-Funk': np.float64(30.0),
     'Mcklopedia': np.float64(30.0),
     'El Hijo De La Cumbia': np.float64(20.666666666666668),
     'Bule Bule': np.float64(25.0),
     'Rapadura': np.float64(27.0),
     'The Budos Band': np.float64(25.333333333333332),
     'Roberto Mendes': np.float64(30.0),
     'Claudia Manzo': np.float64(26.8),
     'The New Mastersounds': np.float64(24.636363636363637),
     'Spellbinder': np.float64(30.0),
     'Moses Bliss': np.float64(45.0),
     'Emandiong': np.float64(49.0),
     'Mulatu Astatke': np.float64(30.0),
     'TJ Cream': np.float64(20.0),
     'Caê': np.float64(22.0),
     "Jor'dan Armstrong": np.float64(20.40909090909091),
     'Sarah Téibo': np.float64(21.833333333333332),
     'Sigag Lauren': np.float64(26.0),
     'Paola Carla': np.float64(33.0),
     'Timoneki': np.float64(20.25),
     'Silvana Estrada': np.float64(25.0),
     'Becca Folkes': np.float64(19.333333333333332),
     'Thiago França': np.float64(22.2),
     'Money Chicha': np.float64(21.5),
     'Breakestra': np.float64(22.2),
     'Africania': np.float64(21.0),
     'Superthriller': np.float64(27.0),
     'Abayomy Afrobeat Orquestra': np.float64(20.0),
     'Tonho Matéria': np.float64(26.0),
     'Céu': np.float64(30.0),
     'Prinx Emmanuel': np.float64(44.0),
     'Grace Lokwa': np.float64(44.0),
     'Cymande': np.float64(37.4),
     'Monophonics': np.float64(49.333333333333336),
     'Kelly Finnigan': np.float64(50.5),
     'Amplexos': np.float64(23.0),
     'Digitaldubs': np.float64(26.25),
     'Leão Etíope do Méier': np.float64(23.0),
     'Lenine': np.float64(39.44444444444444),
     'Choklate': np.float64(25.0),
     'The Bongo Hop': np.float64(22.25),
     'Dafuniks': np.float64(24.0),
     'La Cumbia Chicharra': np.float64(19.333333333333332),
     'Emilie Rambaud': np.float64(23.0),
     'Fela Kuti': np.float64(31.8),
     'Orgone': np.float64(19.428571428571427),
     'Grupo Fantasma': np.float64(18.666666666666668),
     'Maikcel': np.float64(21.5),
     'La Flor del Recuerdo': np.float64(20.0),
     'Carmen Lienqueo': np.float64(24.0),
     'Lack Of Afro': np.float64(22.333333333333332),
     'Wax': np.float64(21.333333333333332),
     'Herbal T': np.float64(21.0),
     'Illy': np.float64(38.2),
     'Eddie Roberts': np.float64(23.0),
     'Lamar Williams Jr.': np.float64(22.333333333333332),
     'Jay Mitta': np.float64(26.0),
     'Kinky': np.float64(46.90909090909091),
     'Los Auténticos Decadentes': np.float64(37.09375),
     'Tropkillaz': np.float64(50.333333333333336),
     'The Bamboos': np.float64(25.625),
     'Kutiman': np.float64(34.0),
     'Doctor Flake': np.float64(44.8),
     'Saundra Williams': np.float64(36.0),
     'Ozoro': np.float64(20.0),
     'Nathanael': np.float64(23.0),
     'Voilaaa': np.float64(20.6),
     'Pat Kalla': np.float64(19.285714285714285),
     'Big Ty': np.float64(22.0),
     'Markis': np.float64(22.0),
     'The Poets Of Rhythm': np.float64(23.857142857142858),
     'Fred Hammond': np.float64(23.0),
     'Andres Nusser': np.float64(22.0),
     'Hawa': np.float64(21.0),
     'Fouley Badiaga': np.float64(21.0),
     'Son Palenque': np.float64(24.0),
     'Cerrero': np.float64(24.0),
     'El León Pardo': np.float64(24.0),
     'Washington': np.float64(35.0),
     'Ikebe Shakedown': np.float64(25.333333333333332),
     'Nosa': np.float64(0.0),
     'The Mighty Imperials': np.float64(20.0),
     'Nidia Gongora': np.float64(19.8),
     'Otto': np.float64(34.25),
     'Wganda Kenya': np.float64(19.0),
     'Ada Ehi': np.float64(25.666666666666668),
     'Osibisa': np.float64(18.0),
     'Nero X': np.float64(38.0),
     'Marko': np.float64(19.0),
     'Angeloh': np.float64(19.0),
     'S.O': np.float64(21.0),
     'Lijadu Sisters': np.float64(45.0),
     'Brownout': np.float64(18.8),
     'Orlando Julius': np.float64(20.0),
     'Speedometer': np.float64(18.666666666666668),
     'Eddie Neblett': np.float64(18.5),
     'Jorge Du Peixe': np.float64(21.0),
     'São Paulo Ska Jazz': np.float64(19.5),
     'Ken Stewart': np.float64(22.0),
     'Lafayette Afro Rock Band': np.float64(18.25),
     'Femi Kuti': np.float64(19.5),
     'Michael Masser': np.float64(21.0),
     "Oscar D'León": np.float64(19.61111111111111),
     'Marizu': np.float64(22.4),
     'ATR': np.float64(21.0),
     'Orchestra Baobab': np.float64(16.8),
     'ST-Saint': np.float64(19.0),
     'Mark Asari': np.float64(20.0),
     'Reblah': np.float64(20.0),
     '678NATH': np.float64(20.0),
     'Raptist & Marko': np.float64(20.0),
     'Leslie Skye': np.float64(20.0),
     'Zelijah & Konola': np.float64(20.0),
     'Gtay': np.float64(20.0),
     'Nizzy Nath': np.float64(20.0),
     'JL Poleon': np.float64(20.0),
     'A Mose': np.float64(17.75),
     'TBabz': np.float64(20.0),
     'Sir Jean': np.float64(33.0),
     'Les Mamans du Congo': np.float64(20.0),
     'RROBIN': np.float64(20.0),
     'João Selva': np.float64(20.0),
     'Ocote Soul Sounds': np.float64(17.5),
     'Oba Reengy': np.float64(18.2),
     'Yoski': np.float64(20.0),
     'Orquestra Voadora': np.float64(19.25),
     'Ada Betsabe': np.float64(20.0),
     'Principal': np.float64(20.0),
     'Tneek': np.float64(21.0),
     'Star Feminine Band': np.float64(21.0),
     'Rehmahz': np.float64(18.0),
     'Asha Elia': np.float64(20.0),
     'Victor Rice': np.float64(19.666666666666668),
     'The Souljazz Orchestra': np.float64(18.0),
     'Ondatrópica': np.float64(20.0),
     'Thee Commons': np.float64(19.0),
     'Kiko Dinucci e Bando Afro Macarrônico': np.float64(19.333333333333332),
     'Gyedu-Blay Ambolley': np.float64(18.666666666666668),
     'Onã': np.float64(20.0),
     'Roxie Ray': np.float64(22.0),
     'Ebo Taylor': np.float64(26.75),
     'Greyboy': np.float64(32.4),
     'Nino Moschella': np.float64(53.0),
     'Rex Williams': np.float64(46.0),
     'Akalé Wubé': np.float64(39.0),
     'Ilover': np.float64(20.0),
     'Dj Horphuray': np.float64(42.0),
     'Jason Nicholson-Porter': np.float64(20.0),
     'Tehillah Daniels': np.float64(20.0),
     'Naffymar': np.float64(20.0),
     'Jaron Nurse': np.float64(17.333333333333332),
     'Bantunagojeje': np.float64(19.0),
     'The Sound Stylistics': np.float64(21.0),
     'The Soul Snatchers': np.float64(21.0),
     'Vaudou Game': np.float64(17.0),
     'Greatman Takit': np.float64(20.0),
     'Euforquestra': np.float64(18.5),
     'Kyle Hollingsworth': np.float64(19.0),
     'Eben': np.float64(40.666666666666664),
     'The Dynamics': np.float64(31.4),
     'Piya Malik': np.float64(38.666666666666664),
     'Matata': np.float64(20.0),
     'Jungle Fire': np.float64(20.857142857142858),
     'Buyepongo': np.float64(18.0),
     'Lass': np.float64(17.5),
     'Wayne Gidden': np.float64(21.0),
     'Ricardo Lemvo': np.float64(18.666666666666668),
     'Foli Griô Orquestra': np.float64(18.0),
     'DJ Vinimax': np.float64(19.0),
     'Cebolinha do Passinho': np.float64(19.0),
     'Makina Loca': np.float64(19.0),
     'Brown Sabbath': np.float64(18.0),
     'Aaron Behrens': np.float64(19.0),
     'The Heliocentrics': np.float64(18.25),
     'Lowell Pye': np.float64(20.0),
     'Mulú': np.float64(21.0),
     'Le Super Mojo': np.float64(18.0),
     'Bosq': np.float64(33.333333333333336),
     'Rubén Blades': np.float64(30.11111111111111),
     'La Chilinga': np.float64(20.0),
     'Julian Assange': np.float64(21.0),
     'Tom Morello': np.float64(32.666666666666664),
     'Kamilya Jubran': np.float64(21.0),
     'Edz': np.float64(19.0),
     'Iconili': np.float64(18.0),
     'Hektombe': np.float64(18.0),
     'Sharyn': np.float64(17.5),
     'Qyubi': np.float64(19.0),
     'Totó La Momposina': np.float64(64.0),
     'Susana Baca': np.float64(64.0),
     'Maria Rita': np.float64(41.88235294117647),
     'BANTU': np.float64(19.0),
     'Megaloh': np.float64(34.75),
     'Ghanaian Stallion': np.float64(19.0),
     'Soothsayers': np.float64(18.0),
     'Renaud Bilombo': np.float64(19.0),
     'Patchworks': np.float64(18.0),
     'Nildes Bomfim': np.float64(19.0),
     'Alan Evans': np.float64(18.0),
     'Cuarteto Patria': np.float64(17.666666666666668),
     'La Dame Blanche': np.float64(19.0),
     'Warsaw Afrobeat Orchestra': np.float64(18.0),
     'Höröyá': np.float64(16.5),
     'Lee Fields': np.float64(49.0),
     'The Shacks': np.float64(49.0),
     'Hot Chip': np.float64(44.5),
     'Lefties Soul Connection': np.float64(19.5),
     'Sambabook': np.float64(18.0),
     'J Prince': np.float64(17.0),
     'Alogte Oho & His Sounds of Joy': np.float64(18.0),
     'Bacao Rhythm & Steel Band': np.float64(20.77777777777778),
     'Village Cuts': np.float64(18.0),
     'Felipe Gordon': np.float64(18.0),
     'Pacific Express': np.float64(18.0),
     'Jonathan Bulter': np.float64(18.0),
     'Nolly': np.float64(18.0),
     'The Vaccines': np.float64(18.0),
     'John Hill': np.float64(18.0),
     'Rich Costey': np.float64(18.0),
     'The Sorcerers': np.float64(17.0),
     'Cyril Neville': np.float64(20.0),
     'King Sunny Ade': np.float64(19.0),
     'Silvio Rodríguez': np.float64(68.0),
     'La BOA': np.float64(18.0),
     'IFÁ': np.float64(18.0),
     'Okwei V. Odili': np.float64(18.0),
     'Whitefield Brothers': np.float64(25.0),
     'Mateus Aleluia': np.float64(29.5),
     'Black Coffee': np.float64(35.357142857142854),
     'Mahal Pita': np.float64(20.0),
     'Tim Bowman Jr.': np.float64(19.0),
     'The Shaolin Afronauts': np.float64(25.333333333333332),
     'Manoel Cordeiro': np.float64(18.0),
     'Jeff Franca': np.float64(20.0),
     'Dekel': np.float64(41.0),
     'Johnny Drille': np.float64(38.0),
     'First Fire': np.float64(17.0),
     'Elikem Kofi': np.float64(17.0),
     'Dos Santos': np.float64(17.0),
     'Zé Bigode Orquestra': np.float64(17.333333333333332),
     'Carlos Malta': np.float64(17.0),
     'Dofono de Omulu': np.float64(17.0),
     'Ekedi Nicinha': np.float64(17.0),
     'Chief Checker': np.float64(17.0),
     "Lo'Jo": np.float64(17.0),
     'Vincent Ségal': np.float64(17.0),
     'Robert Wyatt': np.float64(17.0),
     'The Mauskovic Dance Band': np.float64(17.5),
     'ÀTTØØXXÁ': np.float64(19.0),
     'Nathaniel Bassey': np.float64(40.0),
     'Chico Mann': np.float64(17.0),
     'Quantic': np.float64(28.904761904761905),
     'Beloved Music': np.float64(18.0),
     'Triple O': np.float64(17.0),
     'Maga Bo': np.float64(17.5),
     'The Sugarman 3': np.float64(18.0),
     'Joseph Henry': np.float64(17.0),
     'Emma Noble': np.float64(19.0),
     'Guinu': np.float64(17.0),
     'Uhuru-Yenzu': np.float64(18.0),
     'TB1': np.float64(39.0),
     'Louis Pascal': np.float64(34.0),
     'Oneness Of Juju': np.float64(17.0),
     'Mestre Camaleão': np.float64(17.0),
     'Jlyricz': np.float64(15.5),
     'Monomono': np.float64(16.0),
     'Tony Allen With Africa 70': np.float64(17.0),
     'Girma Bèyènè': np.float64(31.5),
     'Niña Dioz': np.float64(19.0),
     'Shigeto': np.float64(19.0),
     'Geraldo Pino': np.float64(17.0),
     'The Heartbeats': np.float64(17.0),
     'Ty': np.float64(31.5),
     'El Rego': np.float64(16.0),
     'Kill Emil': np.float64(17.0),
     'Antibalas': np.float64(16.0),
     'Lara George': np.float64(39.0),
     'Nomo': np.float64(16.0),
     'Assagai': np.float64(17.0),
     'Rama Traore': np.float64(17.0),
     'JKriv': np.float64(16.0),
     'The Funkees': np.float64(39.0),
     'Pat Thomas': np.float64(49.0),
     'Henrik Schwarz': np.float64(42.666666666666664),
     'Buena Onda Reggae Club': np.float64(16.0),
     'The Daktaris': np.float64(16.0),
     'Alex Maas': np.float64(17.0),
     'Camila Recchio': np.float64(18.0),
     'DIXSON': np.float64(18.0),
     'Bola Johnson': np.float64(44.0),
     'Afrika 70': np.float64(36.0),
     'Azarel': np.float64(16.0),
     'Leor': np.float64(16.0),
     'Edem Evangelist': np.float64(16.0),
     'Nkumba System': np.float64(16.0),
     'Mamani Keïta': np.float64(16.0),
     'Captain Planet': np.float64(16.0),
     "cacique'97": np.float64(16.0),
     'Joshua Ali': np.float64(18.0),
     'Zem Audu': np.float64(17.0),
     'Dele Ojo': np.float64(16.0),
     'The Polyversal Souls': np.float64(16.0),
     'Sir Frank Karikari': np.float64(16.0),
     'T$unami811': np.float64(17.0),
     'Los Celestinos': np.float64(17.0),
     'Kylie Auldist': np.float64(17.0),
     'Gil Joe': np.float64(17.0),
     'Elif Çağlar': np.float64(16.0),
     'Bing Ji Ling': np.float64(18.0),
     'Elliott Cole': np.float64(18.0),
     'Ceci Bastida': np.float64(19.0),
     'Lido Pimienta': np.float64(19.0),
     "Gilles Peterson's Havana Cultura Band": np.float64(16.0),
     'Ria Currie': np.float64(17.0),
     'Meridian Brothers': np.float64(16.0),
     'The Neighbourhood': np.float64(75.58333333333333),
     'MGMT': np.float64(70.63636363636364),
     'Sam Tinnesz': np.float64(58.11764705882353),
     'Yacht Money': np.float64(63.4),
     'Nirvana': np.float64(65.24705882352941),
     'KALEO': np.float64(18.06896551724138),
     'The Score': np.float64(62.68888888888889),
     'grandson': np.float64(32.23529411764706),
     'WALK THE MOON': np.float64(76.6),
     'Red Hot Chili Peppers': np.float64(41.132075471698116),
     'The Killers': np.float64(7.402777777777778),
     'Phoebe Bridgers': np.float64(51.46666666666667),
     'Nickelback': np.float64(56.285714285714285),
     'Weezer': np.float64(16.308333333333334),
     'Armin van Buuren': np.float64(38.01546391752577),
     'Benno De Goeij': np.float64(1.0),
     'Deftones': np.float64(31.11764705882353),
     'Toni Halliday': np.float64(0.0),
     'Ryan Pardey': np.float64(0.0),
     'Julieta Venegas': np.float64(37.0),
     'The Offspring': np.float64(51.93939393939394),
     'Syd': np.float64(60.5),
     'The 1975': np.float64(70.83333333333333),
     'Radiohead': np.float64(76.65),
     'Hoobastank': np.float64(38.958333333333336),
     'The White Stripes': np.float64(66.6896551724138),
     'Thousand Foot Krutch': np.float64(67.63636363636364),
     'Dominic Fike': np.float64(43.625),
     'Zendaya': np.float64(53.714285714285715),
     'R.E.M.': np.float64(23.19047619047619),
     'Zoé': np.float64(3.6),
     'Skillet': np.float64(40.96),
     'The Smashing Pumpkins': np.float64(26.928571428571427),
     'Los Enanitos Verdes': np.float64(11.758620689655173),
     'Korn': np.float64(17.23076923076923),
     'Fitz and The Tantrums': np.float64(14.86111111111111),
     'Ashes Remain': np.float64(67.5),
     '3 Doors Down': np.float64(35.666666666666664),
     'X Ambassadors': np.float64(51.904761904761905),
     'Jensen McRae': np.float64(0.0),
     'Counting Crows': np.float64(0.2926829268292683),
     'Andrés Calamaro': np.float64(4.3125),
     'Alejandro Sanz': np.float64(17.0),
     'Sebastian Yatra': np.float64(6.666666666666667),
     'Leiva': np.float64(43.8125),
     'Ivan Ferreiro': np.float64(20.0),
     'Vanessa Carlton': np.float64(15.222222222222221),
     'Megadeth': np.float64(27.790697674418606),
     'Volbeat': np.float64(0.16666666666666666),
     'Julio Iglesias': np.float64(11.4),
     'Vicentico': np.float64(10.333333333333334),
     'The Black Keys': np.float64(44.13333333333333),
     'Queens of the Stone Age': np.float64(34.583333333333336),
     'Juanes': np.float64(6.03921568627451),
     'Niño Josele': np.float64(0.0),
     'Carlos Vives': np.float64(0.0),
     'Hillsong UNITED': np.float64(67.66666666666667),
     'TAYA': np.float64(52.666666666666664),
     'Yeah Yeah Yeahs': np.float64(12.033333333333333),
     'A-Trak': np.float64(7.318181818181818),
     'Duman': np.float64(21.53846153846154),
     'Marilyn Manson': np.float64(17.839622641509433),
     'Blur': np.float64(8.1),
     'Babasónicos': np.float64(29.92),
     'Grouplove': np.float64(39.5),
     'Shinsei Kamattechan': np.float64(60.125),
     'Neon Trees': np.float64(13.448275862068966),
     'The Seige': np.float64(58.4),
     'The Strokes': np.float64(63.44230769230769),
     'Creed': np.float64(52.96551724137931),
     'Matt Redman': np.float64(46.43333333333333),
     'The EverLove': np.float64(65.0),
     'Charlie Brown Jr.': np.float64(42.34594594594594),
     'Young the Giant': np.float64(51.857142857142854),
     'No Doubt': np.float64(3.6956521739130435),
     'Jimmy Eat World': np.float64(16.675324675324674),
     'Pantera': np.float64(46.891891891891895),
     'Dave Audé': np.float64(9.166666666666666),
     'Bounty Killer': np.float64(17.095238095238095),
     'Soundgarden': np.float64(32.55172413793103),
     'Stone Temple Pilots': np.float64(39.1),
     'Poets of the Fall': np.float64(48.714285714285715),
     'Welshly Arms': np.float64(43.333333333333336),
     'Willyecho': np.float64(52.0),
     'King Kavalier': np.float64(64.0),
     'The Afters': np.float64(56.0),
     'Smash Mouth': np.float64(8.4),
     'Godsmack': np.float64(33.85),
     'Detonautas Roque Clube': np.float64(47.46153846153846),
     'Rob Zombie': np.float64(9.801801801801801),
     'Ozzy Osbourne': np.float64(49.73684210526316),
     'The Verve': np.float64(16.4),
     'Pixies': np.float64(78.0),
     'Mazzy Star': np.float64(20.5),
     'Franz Ferdinand': np.float64(77.0),
     'Adam Jensen': np.float64(58.2),
     'Bullet For My Valentine': np.float64(45.45),
     'Royal Blood': np.float64(13.466666666666667),
     'Chris Tomlin': np.float64(40.574803149606296),
     'Lauren Daigle': np.float64(43.888888888888886),
     'Audrey Assad': np.float64(30.75),
     'Los Prisioneros': np.float64(20.80952380952381),
     'Aleks Syntek': np.float64(20.375),
     'La Ley': np.float64(0.0),
     'Fito Paez': np.float64(12.6),
     'Molotov': np.float64(11.25),
     'You Me At Six': np.float64(40.0),
     'lovelytheband': np.float64(52.875),
     'Papa Roach': np.float64(67.5),
     'Porcupine Tree': np.float64(52.333333333333336),
     'Layto': np.float64(62.0),
     'Neoni': np.float64(59.25),
     'Rage Against The Machine': np.float64(77.28571428571429),
     'Foo Fighters': np.float64(69.05714285714286),
     'Arcane': np.float64(64.66666666666667),
     'League of Legends': np.float64(69.66666666666667),
     'The Phantoms': np.float64(45.333333333333336),
     'Bishop Briggs': np.float64(21.333333333333332),
     'Lifehouse': np.float64(47.4),
     'Staind': np.float64(30.11111111111111),
     'Third Eye Blind': np.float64(33.42857142857143),
     'Los Caligaris': np.float64(42.43421052631579),
     'Pato Fu': np.float64(47.27272727272727),
     'Bush': np.float64(19.428571428571427),
     'Barns Courtney': np.float64(25.875),
     'Lennon Stella': np.float64(46.53846153846154),
     'Joy Division': np.float64(27.466666666666665),
     'TobyMac': np.float64(0.0),
     'Charly García': np.float64(30.256756756756758),
     'Terrian': np.float64(0.0),
     'Leigh Nash': np.float64(0.0),
     'O Rappa': np.float64(40.86440677966102),
     'Declan McKenna': np.float64(80.0),
     'Bohnes': np.float64(67.0),
     'Arrested Youth': np.float64(56.333333333333336),
     'Leeland': np.float64(56.833333333333336),
     'Casting Crowns': np.float64(66.0),
     'Sanctus Real': np.float64(60.666666666666664),
     'Os Paralamas Do Sucesso': np.float64(45.0),
     'Las Pastillas del Abuelo': np.float64(36.23529411764706),
     'I DONT KNOW HOW BUT THEY FOUND ME': np.float64(14.6),
     'Moderatto': np.float64(0.0),
     'Faith No More': np.float64(9.28),
     'Jet': np.float64(38.5),
     'Blind Melon': np.float64(12.483870967741936),
     'Live': np.float64(27.0),
     'The Velvet Underground': np.float64(46.333333333333336),
     'Ella Es Tan Cargosa': np.float64(51.0),
     'Danna Paola': np.float64(0.3684210526315789),
     'KAROL G': np.float64(10.412280701754385),
     'Los Bunkers': np.float64(5.75),
     'Manuel García': np.float64(0.0),
     'Empire of the Sun': np.float64(23.11111111111111),
     'BLUE ENCOUNT': np.float64(65.0),
     'Survive Said The Prophet': np.float64(56.6),
     'Bryce Fox': np.float64(59.166666666666664),
     'Matthew West': np.float64(22.333333333333332),
     'Flört': np.float64(35.625),
     'Los Tipitos': np.float64(48.2),
     'Esteman': np.float64(0.0),
     'Tan Bionica': np.float64(47.6),
     'Highly Suspect': np.float64(37.42857142857143),
     'Wolfmother': np.float64(36.65384615384615),
     'Hole': np.float64(41.44444444444444),
     'Los Abuelos De La Nada': np.float64(2.6875),
     'Creep Hyp': np.float64(33.28888888888889),
     'Paula Pera y el fin de los Tiempos': np.float64(0.0),
     'Bersuit Vergarabat': np.float64(15.5),
     'La Vela Puerca': np.float64(14.833333333333334),
     'La Mosca Tse-Tse': np.float64(14.2),
     'La Renga': np.float64(0.0),
     'The Technicolors': np.float64(66.0),
     'Cidergirl': np.float64(58.55555555555556),
     'Alter Bridge': np.float64(53.02325581395349),
     'Weathers': np.float64(71.0),
     'MISSIO': np.float64(57.2),
     'Matt Maher': np.float64(64.0),
     'Los Piojos': np.float64(47.5),
     'Catupecu Machu': np.float64(30.884615384615383),
     'Bandalos Chinos': np.float64(6.285714285714286),
     'Adan Jodorowsky': np.float64(0.0),
     'Panteon Rococo': np.float64(42.24074074074074),
     'P.O.D.': np.float64(7.888888888888889),
     'Sumo': np.float64(41.05),
     'Enjambre': np.float64(46.27272727272727),
     'Everclear': np.float64(20.0),
     'Gondwana': np.float64(15.11111111111111),
     'Tex Tex': np.float64(50.0),
     'Saint Motel': np.float64(69.33333333333333),
     'Living Colour': np.float64(68.0),
     'The Servant': np.float64(39.8),
     'Drowning Pool': np.float64(39.42857142857143),
     'Patricio Rey y sus Redonditos de Ricota': np.float64(42.857142857142854),
     'Marcelo Nova': np.float64(44.714285714285715),
     'Caifanes': np.float64(46.333333333333336),
     'Divididos': np.float64(32.10344827586207),
     'Pitty': np.float64(40.111111111111114),
     'Guasones': np.float64(43.75),
     'The Adams': np.float64(47.0),
     'NX Zero': np.float64(40.57142857142857),
     'Fountains Of Wayne': np.float64(18.0),
     'Pulp': np.float64(0.0),
     'Los Rodriguez': np.float64(2.125),
     'Interpuesto': np.float64(48.0),
     'Crazy Town': np.float64(73.0),
     'RAC': np.float64(68.2),
     'Louis The Child': np.float64(38.285714285714285),
     'Hinder': np.float64(62.77777777777778),
     'Goodnight Fellows': np.float64(36.0),
     'Watt White': np.float64(61.2),
     'Audioslave': np.float64(57.608695652173914),
     'Phil Wickham': np.float64(43.638297872340424),
     'Kari Jobe': np.float64(38.36),
     'Ne Jupiter': np.float64(47.0),
     'Mancha De Rolando': np.float64(47.0),
     'RZO': np.float64(44.714285714285715),
     'Kapanga': np.float64(41.5),
     'Atreyu': np.float64(22.2),
     'Comisario Pantera': np.float64(35.5),
     'Luis Humberto Navejas': np.float64(47.0),
     'Callejeros': np.float64(47.0),
     'Jeremy Camp': np.float64(0.0),
     'Semisonic': np.float64(0.0),
     'LCD Soundsystem': np.float64(8.714285714285714),
     'Soulwax': np.float64(5.833333333333333),
     'La 25': np.float64(45.4),
     'Alejandro Vazquez': np.float64(47.0),
     'Phosphorescent': np.float64(62.0),
     'Pearl Jam': np.float64(60.90909090909091),
     'Zayde Wølf': np.float64(53.68421052631579),
     'Slank': np.float64(38.645161290322584),
     'Machine Head': np.float64(36.21212121212121),
     'Los Caballeros De La Quema': np.float64(47.0),
     'The Wallflowers': np.float64(23.0),
     'Los Hermanos': np.float64(42.421052631578945),
     'Eels': np.float64(0.0),
     'Steve Earle': np.float64(0.0),
     'Los Estramboticos': np.float64(39.666666666666664),
     'White Zombie': np.float64(7.290322580645161),
     "Jane's Addiction": np.float64(18.333333333333332),
     'Rend Collective': np.float64(39.77777777777778),
     'We The Kingdom': np.float64(37.77777777777778),
     'Clutch': np.float64(0.0),
     'blackbear': np.float64(62.55555555555556),
     'Incubus': np.float64(68.45454545454545),
     'A Rocket To The Moon': np.float64(52.77777777777778),
     'Lupa': np.float64(46.0),
     'Three Days Grace': np.float64(65.45238095238095),
     'Maria Rita / 1134893': np.float64(47.0),
     'Café Tacvba': np.float64(46.0),
     'Virus': np.float64(40.416666666666664),
     'Bombay Bicycle Club': np.float64(34.714285714285715),
     'Michael W. Smith': np.float64(32.28947368421053),
     'Fuel': np.float64(11.5),
     'Las Pelotas': np.float64(46.0),
     'Type O Negative': np.float64(29.48148148148148),
     'The Dandy Warhols': np.float64(0.0),
     'No Te Va Gustar': np.float64(45.666666666666664),
     'Estelares': np.float64(45.25),
     'Serú Girán': np.float64(44.666666666666664),
     'Los Gardelitos': np.float64(45.75),
     'The Glitch Mob': np.float64(56.333333333333336),
     'Toploader': np.float64(66.0),
     'Erich Lee': np.float64(64.0),
     'Rammstein': np.float64(63.94565217391305),
     ...}



Теперь создадим новую колонку, которая будет считать среднее арифметическое популярноти певцов, это позволит нам точнее выявить популярные треки, т.к. они создаются популярными исполнителями


```python
for i in origin_data.index:
  list_artists = origin_data['artists'][i].split(';')
  working_data.loc[i, "artists_avg_popularity"] = sum(artists_avg.get(artist, 0) for artist in list_artists) / len(list_artists)
working_data['artists_avg_popularity']
```




    0         58.000000
    1         42.923077
    2         57.875000
    3         51.090909
    4         42.916667
                ...    
    113995    23.500000
    113996    23.500000
    113997    26.312500
    113998    32.289474
    113999    26.312500
    Name: artists_avg_popularity, Length: 113999, dtype: float64



Посмотрим, что у нас получилось


```python
popular = working_data[working_data['popularity'] > 90]
unpopular = working_data[working_data['popularity'] < 40]
medium = working_data[working_data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

perfect_artists_avg_popularity = popular['artists_avg_popularity'].mean()

plt.figure(figsize=(10, 6))

plt.scatter(popular['artists_avg_popularity'], popular['popularity'], c='orange', label='popular')
plt.axvline(x=perfect_artists_avg_popularity, color='red', linestyle='--', label='mean popular')

plt.scatter(medium['artists_avg_popularity'], medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium['artists_avg_popularity'].mean(), color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['artists_avg_popularity'], unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular['artists_avg_popularity'].mean(), color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('artists_avg_popularity')
plt.ylabel('popularity')

plt.show()
```


    
![png](spotify_files/spotify_98_0.png)
    


Зелёная линия имеет большую сребнюю популярность, чем красная. Следовательно можно сделать вывод, что не обязательно иметь хороший послужной список для популярности. Следовательно, можно не добавлять этот признак в датасет, тк он не будет иметь большой корреляции с Popularity. Но мы это проверим

### Топ жанров

Выведем топ жанров


```python
popular_track = origin_data.groupby('track_genre')['popularity'].mean()
best = pd.DataFrame(popular_track.sort_values().tail(20))

plt.bar(list(best.index), list(best.popularity))
plt.xlabel('track genre')
plt.ylabel('popularity')
plt.xticks(rotation=90)
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
     [Text(0, 0, 'metal'),
      Text(1, 0, 'british'),
      Text(2, 0, 'ambient'),
      Text(3, 0, 'pagode'),
      Text(4, 0, 'electronic'),
      Text(5, 0, 'brazil'),
      Text(6, 0, 'deep-house'),
      Text(7, 0, 'mandopop'),
      Text(8, 0, 'piano'),
      Text(9, 0, 'progressive-house'),
      Text(10, 0, 'pop'),
      Text(11, 0, 'sertanejo'),
      Text(12, 0, 'emo'),
      Text(13, 0, 'anime'),
      Text(14, 0, 'indian'),
      Text(15, 0, 'grunge'),
      Text(16, 0, 'sad'),
      Text(17, 0, 'chill'),
      Text(18, 0, 'k-pop'),
      Text(19, 0, 'pop-film')])




    
![png](spotify_files/spotify_102_1.png)
    


Это топ 20 жанров по популярности, в топе pop-film, k-pop, cill, sad и тд

### Длина названия трека

Ну и на последок проверим поговорку "Краткость сестра таланта" и проверим зависимость популярности от длины названия


```python
short = 0
short_cnt = 0
normal = 0
normal_cnt = 0
long = 0
long_cnt = 0
for i in origin_data.index:
  l = len(origin_data['track_name'][i])
  if l < 7:
    short += origin_data['popularity'][i]
    short_cnt += 1
  elif l < 17:
    normal += origin_data['popularity'][i]
    normal_cnt += 1
  else:
    long += origin_data['popularity'][i]
    long_cnt += 1

print(f'Short: {short/short_cnt}')
print(f'Normal: {normal/normal_cnt}')
print(f'Long: {long/long_cnt}')
```

    Short: 37.22143497027445
    Normal: 33.33713300711744
    Long: 32.104044548651814


Наше предположение оправдалось - треки с коротким названием в среднем более популярны

Сделаем график


```python
working_data['track_name_length'] = 0
for i in origin_data.index:
  working_data.loc[i, "track_name_length"] = len(origin_data['track_name'][i])

working_data['track_name_length']
```




    0          6
    1         16
    2         14
    3         26
    4          7
              ..
    113995    19
    113996    16
    113997    14
    113998     7
    113999     9
    Name: track_name_length, Length: 113999, dtype: int64




```python
popular = working_data[working_data['popularity'] > 90]
unpopular = working_data[working_data['popularity'] < 40]
medium = working_data[working_data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

plt.figure(figsize=(10, 6))

plt.scatter(popular['track_name_length'], popular['popularity'], c='orange', label='popular')
plt.axvline(x=popular['track_name_length'].mean(), color='red', linestyle='--', label='mean popular')

plt.scatter(medium['track_name_length'], medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium['track_name_length'].mean(), color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['track_name_length'], unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular['track_name_length'].mean(), color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('track_name_length')
plt.ylabel('popularity')

plt.show()
```


    
![png](spotify_files/spotify_110_0.png)
    


Выведем данные, где название меньше 100 символов


```python
popular = working_data[working_data['popularity'] > 90]
unpopular = working_data[working_data['popularity'] < 40]
medium = working_data[working_data['popularity'] >= 40]
medium = medium[medium['popularity'] <= 90]

popular_med_track_name_length = popular['track_name_length'].mean()
medium_med_track_name_length = medium['track_name_length'].mean()
unpopular_med_track_name_length = unpopular['track_name_length'].mean()

popular = popular[popular['track_name_length'] < 100]
medium = medium[medium['track_name_length'] < 100]
unpopular = unpopular[unpopular['track_name_length'] < 100]

plt.figure(figsize=(10, 6))

plt.scatter(popular['track_name_length'], popular['popularity'], c='orange', label='popular')
plt.axvline(x=popular_med_track_name_length, color='red', linestyle='--', label='mean popular')

plt.scatter(medium['track_name_length'], medium['popularity'], c='lightgreen', label='medium')
plt.axvline(x=medium_med_track_name_length, color='green', linestyle='--', label='mean medium')

plt.scatter(unpopular['track_name_length'], unpopular['popularity'], c='lightblue', label='unpopular')
plt.axvline(x=unpopular_med_track_name_length, color='blue', linestyle='--', label='mean unpopular')

plt.legend()

plt.xlabel('track_name_length')
plt.ylabel('popularity')

plt.show()

print(f'Типичная длина названия популярного трека {popular_med_track_name_length}')
```


    
![png](spotify_files/spotify_112_0.png)
    


    Типичная длина названия популярного трека 13.764705882352942


Разница между медианами маленькая, но она есть. Идеальные названия состоят из 6-20 символов, но есть и такие где 40 символов

## Feature engineering на основе нащих графиков

Создадим датасет в нём мы будем хранить числовые характеристики в формате удалённости от "типичных идеальных" значений, которые мы вывели выше


```python
perfect_data = working_data
```

Удаляем значения, которые не имеют смысла:


1.   artists - мы сделали столбцы artists_cnt	и artists_avg_popularity, поэтому оригинал нам не нужен
2.   time_signature - слишком мало значений, да и отклонение медиан минимально, поэтому удаляем
3.  explicit - слишком мало значений, если будет большая важность вернём
4.  mode - тоже мало значений







```python
perfect_data = perfect_data.drop(columns=['artists', 'time_signature', 'explicit', 'mode'])
perfect_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>danceability</th>
      <th>speechiness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>track_genre</th>
      <th>energy_loudness_acousticness</th>
      <th>artists_cnt</th>
      <th>artists_avg_popularity</th>
      <th>track_name_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8100</td>
      <td>11741</td>
      <td>73</td>
      <td>230666</td>
      <td>0.676</td>
      <td>0.1430</td>
      <td>0.000001</td>
      <td>0.3580</td>
      <td>0.7150</td>
      <td>87.917</td>
      <td>0</td>
      <td>-3.009767</td>
      <td>1</td>
      <td>58.000000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14796</td>
      <td>22528</td>
      <td>55</td>
      <td>149610</td>
      <td>0.420</td>
      <td>0.0763</td>
      <td>0.000006</td>
      <td>0.1010</td>
      <td>0.2670</td>
      <td>77.489</td>
      <td>0</td>
      <td>-0.217437</td>
      <td>1</td>
      <td>42.923077</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39162</td>
      <td>60774</td>
      <td>57</td>
      <td>210826</td>
      <td>0.438</td>
      <td>0.0557</td>
      <td>0.000000</td>
      <td>0.1170</td>
      <td>0.1200</td>
      <td>76.332</td>
      <td>0</td>
      <td>-2.760660</td>
      <td>2</td>
      <td>57.875000</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8580</td>
      <td>9580</td>
      <td>71</td>
      <td>201933</td>
      <td>0.266</td>
      <td>0.0363</td>
      <td>0.000071</td>
      <td>0.1320</td>
      <td>0.1430</td>
      <td>181.740</td>
      <td>0</td>
      <td>-0.104832</td>
      <td>1</td>
      <td>51.090909</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16899</td>
      <td>25689</td>
      <td>82</td>
      <td>198853</td>
      <td>0.618</td>
      <td>0.0526</td>
      <td>0.000000</td>
      <td>0.0829</td>
      <td>0.1670</td>
      <td>119.949</td>
      <td>0</td>
      <td>-2.277291</td>
      <td>1</td>
      <td>42.916667</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113995</th>
      <td>66</td>
      <td>53329</td>
      <td>21</td>
      <td>384999</td>
      <td>0.172</td>
      <td>0.0422</td>
      <td>0.928000</td>
      <td>0.0863</td>
      <td>0.0339</td>
      <td>125.995</td>
      <td>113</td>
      <td>-1.386848</td>
      <td>1</td>
      <td>23.500000</td>
      <td>19</td>
    </tr>
    <tr>
      <th>113996</th>
      <td>66</td>
      <td>65090</td>
      <td>22</td>
      <td>385000</td>
      <td>0.174</td>
      <td>0.0401</td>
      <td>0.976000</td>
      <td>0.1050</td>
      <td>0.0350</td>
      <td>85.239</td>
      <td>113</td>
      <td>-0.012859</td>
      <td>1</td>
      <td>23.500000</td>
      <td>16</td>
    </tr>
    <tr>
      <th>113997</th>
      <td>5028</td>
      <td>38207</td>
      <td>22</td>
      <td>271466</td>
      <td>0.629</td>
      <td>0.0420</td>
      <td>0.000000</td>
      <td>0.0839</td>
      <td>0.7430</td>
      <td>132.378</td>
      <td>113</td>
      <td>-0.476733</td>
      <td>1</td>
      <td>26.312500</td>
      <td>14</td>
    </tr>
    <tr>
      <th>113998</th>
      <td>7238</td>
      <td>21507</td>
      <td>41</td>
      <td>283893</td>
      <td>0.587</td>
      <td>0.0297</td>
      <td>0.000000</td>
      <td>0.2700</td>
      <td>0.4130</td>
      <td>135.960</td>
      <td>113</td>
      <td>-3.410587</td>
      <td>1</td>
      <td>32.289474</td>
      <td>7</td>
    </tr>
    <tr>
      <th>113999</th>
      <td>24357</td>
      <td>5999</td>
      <td>22</td>
      <td>241826</td>
      <td>0.526</td>
      <td>0.0725</td>
      <td>0.000000</td>
      <td>0.0893</td>
      <td>0.7080</td>
      <td>79.198</td>
      <td>113</td>
      <td>-1.585222</td>
      <td>1</td>
      <td>26.312500</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>113999 rows × 15 columns</p>
</div>



<p>Заменяем значения в столбиках их отклонением от медианы лучших треков, там где это имеет смысл</p>
<p>Например, заменять artists_avg_popularity его отклонением не имеет смысл, т.к. мы поняли, что этот столбец не имеет однозначной корреляции с популярностью (точнее значения самых популярных треков)</p>


```python
perfect_data['duration_ms'] = abs(popular_med_duration_ms - working_data['duration_ms'])

perfect_danceability = working_data[working_data['popularity'] > 90]['danceability'].mean()
perfect_data['danceability'] = abs(perfect_danceability - working_data['danceability'])

perfect_speechiness = working_data[working_data['popularity'] > 90]['speechiness'].mean()
perfect_data['speechiness'] = abs(perfect_speechiness - working_data['speechiness'])

perfect_instrumentalness = working_data[working_data['popularity'] > 90]['instrumentalness'].mean()
perfect_data['instrumentalness'] = abs(perfect_instrumentalness - working_data['instrumentalness'])

perfect_liveness = working_data[working_data['popularity'] > 90]['liveness'].mean()
perfect_data['liveness'] = abs(perfect_liveness - working_data['liveness'])

perfect_data['track_name_length'] = abs(popular_med_track_name_length - working_data['track_name_length'])

perfect_energy_loudness_acousticness = working_data[working_data['popularity'] > 90]['energy_loudness_acousticness'].mean()
perfect_data['energy_loudness_acousticness'] = abs(perfect_liveness - working_data['energy_loudness_acousticness'])

perfect_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>album_name</th>
      <th>track_name</th>
      <th>popularity</th>
      <th>duration_ms</th>
      <th>danceability</th>
      <th>speechiness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>track_genre</th>
      <th>energy_loudness_acousticness</th>
      <th>artists_cnt</th>
      <th>artists_avg_popularity</th>
      <th>track_name_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8100</td>
      <td>11741</td>
      <td>73</td>
      <td>28739.882353</td>
      <td>0.044912</td>
      <td>0.064194</td>
      <td>0.002834</td>
      <td>0.162475</td>
      <td>0.7150</td>
      <td>87.917</td>
      <td>0</td>
      <td>3.205292</td>
      <td>1</td>
      <td>58.000000</td>
      <td>7.764706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14796</td>
      <td>22528</td>
      <td>55</td>
      <td>52316.117647</td>
      <td>0.300912</td>
      <td>0.002506</td>
      <td>0.002830</td>
      <td>0.094525</td>
      <td>0.2670</td>
      <td>77.489</td>
      <td>0</td>
      <td>0.412962</td>
      <td>1</td>
      <td>42.923077</td>
      <td>2.235294</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39162</td>
      <td>60774</td>
      <td>57</td>
      <td>8899.882353</td>
      <td>0.282912</td>
      <td>0.023106</td>
      <td>0.002835</td>
      <td>0.078525</td>
      <td>0.1200</td>
      <td>76.332</td>
      <td>0</td>
      <td>2.956185</td>
      <td>2</td>
      <td>57.875000</td>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8580</td>
      <td>9580</td>
      <td>71</td>
      <td>6.882353</td>
      <td>0.454912</td>
      <td>0.042506</td>
      <td>0.002765</td>
      <td>0.063525</td>
      <td>0.1430</td>
      <td>181.740</td>
      <td>0</td>
      <td>0.300357</td>
      <td>1</td>
      <td>51.090909</td>
      <td>12.235294</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16899</td>
      <td>25689</td>
      <td>82</td>
      <td>3073.117647</td>
      <td>0.102912</td>
      <td>0.026206</td>
      <td>0.002835</td>
      <td>0.112625</td>
      <td>0.1670</td>
      <td>119.949</td>
      <td>0</td>
      <td>2.472816</td>
      <td>1</td>
      <td>42.916667</td>
      <td>6.764706</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113995</th>
      <td>66</td>
      <td>53329</td>
      <td>21</td>
      <td>183072.882353</td>
      <td>0.548912</td>
      <td>0.036606</td>
      <td>0.925165</td>
      <td>0.109225</td>
      <td>0.0339</td>
      <td>125.995</td>
      <td>113</td>
      <td>1.582373</td>
      <td>1</td>
      <td>23.500000</td>
      <td>5.235294</td>
    </tr>
    <tr>
      <th>113996</th>
      <td>66</td>
      <td>65090</td>
      <td>22</td>
      <td>183073.882353</td>
      <td>0.546912</td>
      <td>0.038706</td>
      <td>0.973165</td>
      <td>0.090525</td>
      <td>0.0350</td>
      <td>85.239</td>
      <td>113</td>
      <td>0.208384</td>
      <td>1</td>
      <td>23.500000</td>
      <td>2.235294</td>
    </tr>
    <tr>
      <th>113997</th>
      <td>5028</td>
      <td>38207</td>
      <td>22</td>
      <td>69539.882353</td>
      <td>0.091912</td>
      <td>0.036806</td>
      <td>0.002835</td>
      <td>0.111625</td>
      <td>0.7430</td>
      <td>132.378</td>
      <td>113</td>
      <td>0.672258</td>
      <td>1</td>
      <td>26.312500</td>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>113998</th>
      <td>7238</td>
      <td>21507</td>
      <td>41</td>
      <td>81966.882353</td>
      <td>0.133912</td>
      <td>0.049106</td>
      <td>0.002835</td>
      <td>0.074475</td>
      <td>0.4130</td>
      <td>135.960</td>
      <td>113</td>
      <td>3.606112</td>
      <td>1</td>
      <td>32.289474</td>
      <td>6.764706</td>
    </tr>
    <tr>
      <th>113999</th>
      <td>24357</td>
      <td>5999</td>
      <td>22</td>
      <td>39899.882353</td>
      <td>0.194912</td>
      <td>0.006306</td>
      <td>0.002835</td>
      <td>0.106225</td>
      <td>0.7080</td>
      <td>79.198</td>
      <td>113</td>
      <td>1.780747</td>
      <td>1</td>
      <td>26.312500</td>
      <td>4.764706</td>
    </tr>
  </tbody>
</table>
<p>113999 rows × 15 columns</p>
</div>



### Проверка корреляции


```python
corr_matrix = perfect_data.corr()
plt.figure(figsize=(21, 21))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
```


    
![png](spotify_files/spotify_122_0.png)
    


Особой корреляции с таргетом нет ни у чего, кроме artists_avg_popularity. Что логично, чем популярней исполнитель, тем больше просмотров у его треков, да и создан этот столбец с использованием таргета. 

Далее наша команда решила сделать два датасета для удобства

### <b>Совпадающее название трека и альбома</b>

Названия, и трека, и альбома хорошо коррелируют с популярностью, возможно совпадающие названия будут коррелировать еще лучше


```python
working_data.loc[working_data["album_name"] == working_data["track_name"], "is_same_album"] = True

working_data["is_same_album"] = working_data["is_same_album"].fillna(False)
```

    C:\Users\Тимофей\AppData\Local\Temp\ipykernel_10056\1082106543.py:3: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      working_data["is_same_album"] = working_data["is_same_album"].fillna(False)


### <b>Danceability и energy</b>

Анализ показал, что danceability и energy прямопропорцианальны популярности на определенных отрезках, создадим соответствующий признак


```python
working_data["dance_energy"] = data["danceability"] * data["energy"]
```

### <b>Корреляция новых признаков с таргетом</b>


```python
corr_matrix = working_data[['popularity','dance_energy','is_same_album']].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
```


    
![png](spotify_files/spotify_132_0.png)
    


Тут тоже ваимосвязи не найдено

## Feature importance

### Data

Проверим оригинальный датасет, для того чтобы понять какие выведенные нами признаки улучшат обучение


```python
#Разделяю на обучающие и тестовые выборки
X = data.drop(columns=['popularity']) #Обучающий без целевого признака
y = data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%
```


```python
import tensorflow as tf
from tensorflow import keras
```


```python
scaler = StandardScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Строим модель. Трёхслойная архитектура 1ый - входной, 2ой - скрытый, 3й - выходной
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error') #Стандартные для регресси параметры

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Тестовый loss: {loss}')
```

    Epoch 1/10


    c:\Users\Тимофей\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 642.4625 - val_loss: 479.4274
    Epoch 2/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 893us/step - loss: 474.2557 - val_loss: 470.2638
    Epoch 3/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 897us/step - loss: 466.1706 - val_loss: 460.8526
    Epoch 4/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 875us/step - loss: 455.5454 - val_loss: 457.2035
    Epoch 5/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 875us/step - loss: 442.7635 - val_loss: 450.2833
    Epoch 6/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 930us/step - loss: 440.8005 - val_loss: 454.1546
    Epoch 7/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 897us/step - loss: 433.5335 - val_loss: 442.0698
    Epoch 8/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 873us/step - loss: 426.0803 - val_loss: 438.5063
    Epoch 9/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 906us/step - loss: 422.0136 - val_loss: 433.6412
    Epoch 10/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 856us/step - loss: 417.4763 - val_loss: 430.0100
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 624us/step - loss: 433.1049
    Тестовый loss: 426.6445617675781



```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
```


```python
#Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

#Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 464us/step
    MAE: 16.635852813720703
    RMSE: 20.655373882233256



```python
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')
```

    MAPE: 1.969827909258445e+16



```python
#Важность признаков с помощью permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
importances = pd.Series(result.importances_mean, index=X.columns)

#Визуализация важности признаков
importances.sort_values().plot(kind='barh', figsize=(10,6))
plt.title('Важность признаков')
plt.xlabel('Среднее уменьшение MAE')
plt.show()
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 448us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 408us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 439us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 443us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 452us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 492us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 440us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 450us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 431us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 436us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 466us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 477us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 542us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 499us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 485us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 477us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 495us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 422us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 491us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 422us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 452us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 438us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 451us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 446us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 432us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 431us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 440us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 439us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 397us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 440us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 437us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 425us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 422us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 487us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 480us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 456us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 473us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 491us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 500us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 459us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 476us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 484us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 477us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 438us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 429us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 425us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 404us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 509us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 452us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 470us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 401us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 422us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 404us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 458us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 437us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step



    
![png](spotify_files/spotify_143_1.png)
    


В этом датасете остались ненужные колонки track_id и unnamed: 0, а также есть колонки дублирующие energy_loudness_acousticness

Мы получили информацию permutation_importance нашего почти оригиального датасета, теперь сравним с остальными

### perfect_data


```python
#Разделяю на обучающие и тестовые выборки
X = perfect_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = perfect_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%
```


```python
scaler = StandardScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Строим модель. Трёхслойная архитектура 1ый - входной, 2ой - скрытый, 3й - выходной
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error') #Стандартные для регресси параметры

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Тестовый loss: {loss}')
```

    Epoch 1/10


    c:\Users\Тимофей\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 908us/step - loss: 411.3362 - val_loss: 171.0394
    Epoch 2/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 855us/step - loss: 170.4821 - val_loss: 168.1203
    Epoch 3/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 860us/step - loss: 169.9360 - val_loss: 167.9566
    Epoch 4/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 898us/step - loss: 165.9877 - val_loss: 169.0047
    Epoch 5/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 862us/step - loss: 169.0824 - val_loss: 166.4472
    Epoch 6/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 844us/step - loss: 164.0618 - val_loss: 166.3557
    Epoch 7/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 984us/step - loss: 163.7334 - val_loss: 165.0711
    Epoch 8/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 855us/step - loss: 164.8539 - val_loss: 166.0044
    Epoch 9/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 860us/step - loss: 167.4994 - val_loss: 165.4149
    Epoch 10/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 900us/step - loss: 162.1530 - val_loss: 165.4988
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 614us/step - loss: 166.4454
    Тестовый loss: 167.76124572753906



```python
#Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

#Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    MAE: 7.968059062957764
    RMSE: 12.952270921405418



```python
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')
```

    MAPE: 7787140779868160.0



```python
#Важность признаков с помощью permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
importances = pd.Series(result.importances_mean, index=X.columns)

#Визуализация важности признаков
importances.sort_values().plot(kind='barh', figsize=(10,6))
plt.title('Важность признаков')
plt.xlabel('Среднее уменьшение MAE')
plt.show()
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 431us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 437us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 481us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 425us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 538us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 505us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 503us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 435us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 487us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 426us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 429us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 586us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 480us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 501us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 483us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 478us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 474us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 497us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 466us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 472us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 472us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 468us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 469us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 472us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 492us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 505us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 484us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 482us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 480us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 499us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 491us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 493us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 483us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 465us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 486us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 477us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 487us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 507us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 492us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 451us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 422us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 467us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 439us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 459us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 468us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 427us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 450us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 452us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 459us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 467us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 426us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 489us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 558us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 489us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 465us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 446us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 453us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 440us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 456us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 461us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 455us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 713us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 569us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 550us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 556us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 558us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 548us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 540us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 550us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 596us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 624us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 626us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 543us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 502us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 499us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 482us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 508us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 482us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 538us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 546us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 425us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 431us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 441us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 466us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 432us/step



    
![png](spotify_files/spotify_151_1.png)
    


<p>С самого начала в датасете был проблемный параметр - artists_avg_popularity. Как мы заметили раньше в топы часто попадали исполнители и с маленьким послужным списком, что не позволет нам говорить о нормальной с популярностью корреляции. Однако, этот показатель основывается на таргете, что и делает среднее уменьшение mae таким большим</p>
<p>Теперь стало очевидно, что параметр artists_avg_popularity сломал нам всю картину, удалим этот параметр</p>

### perfect_data но без artists_avg_popularity


```python
perfect_data = perfect_data.drop(columns=['artists_avg_popularity'])
```


```python
#Разделяю на обучающие и тестовые выборки
X = perfect_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = perfect_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%
```


```python
scaler = StandardScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Строим модель. Трёхслойная архитектура 1ый - входной, 2ой - скрытый, 3й - выходной
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error') #Стандартные для регресси параметры

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Тестовый loss: {loss}')
```

    Epoch 1/10


    c:\Users\Тимофей\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 987us/step - loss: 641.3469 - val_loss: 481.7422
    Epoch 2/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 880us/step - loss: 473.2812 - val_loss: 470.3751
    Epoch 3/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 891us/step - loss: 463.2887 - val_loss: 463.5867
    Epoch 4/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 872us/step - loss: 456.7450 - val_loss: 458.7802
    Epoch 5/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 903us/step - loss: 452.0393 - val_loss: 452.3161
    Epoch 6/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 946us/step - loss: 442.5210 - val_loss: 450.9184
    Epoch 7/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 940us/step - loss: 443.3243 - val_loss: 448.6157
    Epoch 8/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 890us/step - loss: 439.3868 - val_loss: 445.9702
    Epoch 9/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 867us/step - loss: 438.9438 - val_loss: 444.0168
    Epoch 10/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 860us/step - loss: 435.7084 - val_loss: 440.7797
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 624us/step - loss: 443.9819
    Тестовый loss: 438.6603088378906


Тестовый loss у этой модельки чуть меньше чем у оригинального датасета


```python
#Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

#Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 444us/step
    MAE: 17.07296371459961
    RMSE: 20.944216069540012



```python
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')
```

    MAPE: 1.9184573086695424e+16


Это уже говорит о многом, mape просто запредельный


```python
#Важность признаков с помощью permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
importances = pd.Series(result.importances_mean, index=X.columns)

#Визуализация важности признаков
importances.sort_values().plot(kind='barh', figsize=(10,6))
plt.title('Важность признаков')
plt.xlabel('Среднее уменьшение MAE')
plt.show()
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 427us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 426us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 408us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 404us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 481us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 500us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 484us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 498us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 488us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 483us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 425us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 555us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 557us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 557us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 507us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 561us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 634us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 562us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 559us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 554us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 555us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 563us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 552us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 580us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 496us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 506us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 458us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 438us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 500us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 600us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 581us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 540us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 540us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 486us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 552us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 479us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 503us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 482us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 408us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 497us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 556us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 451us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 500us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 552us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 544us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 566us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 497us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 556us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 442us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 505us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 452us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 456us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 569us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 441us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 425us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 428us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 454us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 419us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 431us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 458us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 438us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 478us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 506us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 447us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 471us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 444us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 478us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 471us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 426us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 441us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 427us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 410us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 441us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step



    
![png](spotify_files/spotify_161_1.png)
    


![image.png](data:image/png;base64,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)

Очень странно, что название альбома и трека в данном датасете имеют такое большое среднее уменьшение mae, в сравнении с оригиналом, хоть кодировали его мы также. Да и изменение метрик, через удаление от среднего значения идеальных показателей результатов не дало, а многие значения и вовсе упали

Если следующие эксперименты не дадут лучших результатов, мы будем вынуждены использовать perfect_data с искусственно добавленным признаком, поскольку текущая модель на этих данных показывает более высокую точность.

### working_data

Удаляем artists_avg_popularity, чтобы неполучить perfect_data 2.0


```python
working_data = working_data.drop(columns=['artists_avg_popularity'])
```

Пришла очередь и этого датасета


```python
#Разделяю на обучающие и тестовые выборки
X = working_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = working_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%
```


```python
scaler = StandardScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Строим модель. Трёхслойная архитектура 1ый - входной, 2ой - скрытый, 3й - выходной
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error') #Стандартные для регресси параметры

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Тестовый loss: {loss}')
```

    Epoch 1/10


    c:\Users\Тимофей\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 629.9026 - val_loss: 473.6077
    Epoch 2/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 947us/step - loss: 466.3256 - val_loss: 461.4564
    Epoch 3/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 881us/step - loss: 447.1480 - val_loss: 453.6538
    Epoch 4/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 956us/step - loss: 441.6978 - val_loss: 448.9626
    Epoch 5/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 980us/step - loss: 439.2640 - val_loss: 445.3980
    Epoch 6/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 956us/step - loss: 436.8534 - val_loss: 443.5292
    Epoch 7/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 954us/step - loss: 432.5876 - val_loss: 441.9325
    Epoch 8/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 944us/step - loss: 427.8484 - val_loss: 439.6813
    Epoch 9/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 840us/step - loss: 428.1078 - val_loss: 438.7204
    Epoch 10/10
    [1m2280/2280[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 884us/step - loss: 424.3502 - val_loss: 435.7430
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 599us/step - loss: 433.2946
    Тестовый loss: 427.58447265625


Показатели примерно такиеже как и у perfect_data без artists_avg_popularity, что не так уж и плохо


```python
#Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

#Вычисление метрик
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    MAE: 16.69524574279785
    RMSE: 20.67811800452315



```python
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}')
```

    MAPE: 1.7835917922271232e+16



```python
#Важность признаков с помощью permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
importances = pd.Series(result.importances_mean, index=X.columns)

#Визуализация важности признаков
importances.sort_values().plot(kind='barh', figsize=(10,6))
plt.title('Важность признаков')
plt.xlabel('Среднее уменьшение MAE')
plt.show()
```

    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 457us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 433us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 482us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 421us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 408us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 381us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 404us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 401us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 384us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 397us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 404us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 477us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 454us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 446us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 467us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 436us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 430us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 427us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 408us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 445us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 482us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 471us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 485us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 460us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 471us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 461us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 478us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 464us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 453us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 485us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 454us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 431us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 423us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 463us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 429us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 389us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 395us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 408us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 439us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 401us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 420us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 397us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 397us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 395us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 414us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 413us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 389us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 437us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 389us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 401us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 394us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 406us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 471us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 411us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 395us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 401us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 404us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 391us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 398us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 405us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 400us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 395us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 396us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 416us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 402us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 392us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 395us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 393us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 403us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 424us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 434us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 401us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 407us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 409us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 399us/step
    [1m713/713[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 415us/step



    
![png](spotify_files/spotify_174_1.png)
    


Из всех добавленых нами параметров хорошо себя показал dance_energy, этот показатель смог стать 2ым по важности в датасете

<p>Нами было принято решение оставить working_data как пример обычного датасета, без таргет переменных в в нём.</p>
<p>Но также если наши модели не будут показывать достойных результатов, то мы вернём в датасет artists_avg_popularity</p>

## Модели

### Линейная регрессия


```python
from sklearn.linear_model import LinearRegression
```


```python
X = working_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = working_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%

scaler = StandardScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = mse_lr ** 0.5
mape_lr = mean_absolute_percentage_error(y_test, y_pred)

print(f'MAE: {mae_lr}')
print(f'MSE: {mse_lr}')
print(f'RMSE: {rmse_lr}')
print(f'MAPE: {mape_lr}')
```

    MAE: 18.295311091729335
    MSE: 482.3078745223844
    RMSE: 21.961508930908742
    MAPE: 2.1262093467835052e+16


<p>MAE: 18.295311091729335</p>
<p>MSE: 482.3078745223844</p>
<p>RMSE: 21.961508930908742</p>
<p>MAPE: 2.126209346783505e+16</p>

Это самая простая модель, быстро обучается, но результат получился довольно слабый. Возможно, для неё больше подойдёт датасет perfect_data, т.к. значения в нём построены на основе удаления собственного значения столбца от "идеальных", что больше вырожает их линейную природу. Однако таблица корреляции perfect_data говорит нам об обратном

### Деревья


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
```


```python
X = working_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = working_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%

scaler = MinMaxScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tree_model = DecisionTreeRegressor(
    max_depth = 10,
    min_samples_split = 5,
    min_samples_leaf = 2,
    random_state = 42
)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

mae_tree = mean_absolute_error(y_test, y_pred)
mse_tree = mean_squared_error(y_test, y_pred)
rmse_tree = mse_tree ** 0.5
mape_tree = mean_absolute_percentage_error(y_test, y_pred)
r2_tree = r2_score(y_test, y_pred)

print(f'MAE: {mae_tree}')
print(f'MSE: {mse_tree}')
print(f'RMSE: {rmse_tree}')
print(f'MAPE: {mape_tree}')
print(f'R^2: {r2_tree}')
```

    MAE: 15.87115448320749
    MSE: 405.4044369134041
    RMSE: 20.134657606063335
    MAPE: 1.5973787125501242e+16
    R^2: 0.18279975976940488


<p>MAE: 15.871168756295095</p>
<p>MSE: 405.4060695074149</p>
<p>RMSE: 20.13469814790912</p>
<p>MAPE: 1.597342465466773e+16</p>
<p>R^2: 0.18279646884287237</p>

Модель дерева показала себя куда лучше, но всё равно не достаточно. Главный её плюс - скорость

### Нейронная сеть


```python
from sklearn.neural_network import MLPRegressor
```


```python
X = working_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = working_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%

scaler = StandardScaler() #Масштабируем данные
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp_model = MLPRegressor(
    hidden_layer_sizes = (100, 50, 25),
    max_iter = 1000,
    random_state = 42,
    solver = 'adam',
    learning_rate_init = 0.001,
    tol = 1e-4
)
mlp_model.fit(X_train, y_train)

predictions = mlp_model.predict(X_test)

mae_mlp = mean_absolute_error(y_test, predictions)
mse_mlp = mean_squared_error(y_test, predictions)
rmse_mlp = mse ** 0.5
mape_mlp = mean_absolute_percentage_error(y_test, predictions)
r2_mlp = r2_score(y_test, predictions)

print(f'MAE: {mae_mlp}')
print(f'MSE: {mse_mlp}')
print(f'RMSE: {rmse_mlp}')
print(f'MAPE: {mape_mlp}')
print(f'R^2: {r2_mlp}')
```

<p>MAE: 14.970817960477026</p>
<p>MSE: 367.0926667656163</p>
<p>RMSE: 20.449484779839537</p>
<p>MAPE: 1.3046750357828898e+16</p>
<p>R^2: 0.2600273007573668</p>

Модель показаля себя крайне хорошо, однако обучение было длительным. Сейчас это - наша лучшая модель. Рассмотрим ещё одну и сравним их!

### Градиентный бустинг


```python
from sklearn.ensemble import GradientBoostingRegressor
```


```python
X = working_data.drop(columns=['popularity']) #Обучающий без целевого признака
y = working_data['popularity'] #Тестовый только popularity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #Делим 80% на 20%

gradient_boosting = GradientBoostingRegressor(
    learning_rate = 0.01,
    n_estimators = 500,
    max_depth = 10,
    # min_samples_leaf = 2,
    # min_samples_split = 5,
    # alpha = 0.1
)
gradient_boosting.fit(X_train, y_train)

y_pred = gradient_boosting.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')
print(f'R^2: {r2}')
```

    MAE: 11.740423918242977
    MSE: 237.62355827904665
    RMSE: 15.415043246097191
    MAPE: 1.1458389161013526e+16
    R^2: 0.5210066510654282


Загружалось 28 минут, но и показатели довольно не плохие

<p>MAE: 11.740423918242977</p>
<p>MSE: 237.62355827904665</p>
<p>RMSE: 15.415043246097191</p>
<p>MAPE: 1.1458389161013526e+16</p>
<p>R^2: 0.5210066510654282</p>

Тирлист моделей:


1.   Модель градиентного бустинга mse - 237
2.   Нейронная сеть mse - 367
3.   Дерево mse - 405
4.   Линейная модель - mse - 482



Получившаяся модель градиентного бустинга нас устраивает, поэтому добавлять в датасет колонку artists_avg_popularity мы не будем

## Кросс-валидация лучшей модели

Проведём кросс-валидацию градиентного бустинга на 5ти фолдах


```python
from sklearn.model_selection import cross_val_score
```


```python
cv_mse = cross_val_score(gradient_boosting, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error')
cv_mae = cross_val_score(gradient_boosting, X_train, y_train, cv = 5, scoring = 'neg_mean_absolute_error')
cv_r2 = cross_val_score(gradient_boosting, X_train, y_train, cv = 5, scoring = 'r2')

cv_mse = -cv_mse
cv_mae = -cv_mae

print(f'mae: {np.mean(cv_mae)} ± {np.std(cv_mae)}')
print(f'mse: {np.mean(cv_mse)} ± {np.std(cv_mse)}')
print(f'r2: {np.mean(cv_r2)} ± {np.std(cv_r2)}')
```

    mae: 11.783840406756983 ± 0.1287962070623241
    mse: 240.98510408477364 ± 4.114154541746161
    r2: 0.5154830333901202 ± 0.004516469985537712


Считается очень долго

<p>mae: 11.783840406756983 ± 0.1287962070623241</p>
<p>mse: 240.98510408477364 ± 4.114154541746161</p>
<p>r2: 0.5154830333901202 ± 0.004516469985537712</p>

Значения у итоговой модели получились неплохие, у нас получилось не переобучить модель об этом говорит небольшое отклонение показателей кросс-валидации от показателей модели
