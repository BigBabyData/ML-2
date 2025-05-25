import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv("dataset.csv")

print("Пропущенные значения:\n", df.isnull().sum())

#Заполнение пропущенных значений
df.fillna(0, inplace=True)

# 3. Описательная статистика
print("\nОписательная статистика:")
print(df.describe())

# 4. Распределение популярности треков
plt.figure(figsize=(10, 5))
sns.histplot(df['popularity'], bins=30, kde=True, color='skyblue')
plt.title("Распределение популярности треков")
plt.xlabel("Популярность")
plt.ylabel("Количество треков")
plt.grid(True)
plt.show()

#5 Корреляция числовых признаков
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляция числовых признаков")
plt.show()

#6 Средняя популярность по жанрам
plt.figure(figsize=(14, 6))
genre_popularity = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=genre_popularity.index, y=genre_popularity.values, palette='viridis')
plt.title("Средняя популярность по жанрам (Top-10)")
plt.xlabel("Жанр")
plt.ylabel("Средняя популярность")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#7 Влияние акустичности на популярность
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="acousticness", y="popularity", hue="explicit", alpha=0.6)
plt.title("Зависимость популярности от акустичности")
plt.xlabel("Акустичность")
plt.ylabel("Популярность")
plt.grid(True)
plt.show()

#8 Выборки по ключу и темпу
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="key", y="tempo")
plt.title("Темп в зависимости от музыкального ключа")
plt.xlabel("Ключ (key)")
plt.ylabel("Темп (BPM)")
plt.grid(True)
plt.show()
