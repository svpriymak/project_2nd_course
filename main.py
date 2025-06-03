# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ================== Раздел 1: Загрузка и объединение данных ==================

# Указываем путь к папке с CSV-файлами
# choose your character
# data_path = '/Users/svpriymak/Downloads/ozon' # svpriymak
data_path = "/home/user/ozon_dataset/ozon"  # ysmoshchenkov

# Проверяем наличие CSV-файлов в указанной директории
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Найдено {len(csv_files)} CSV файлов.")


# Считываем и объединяем все файлы в один DataFrame
data_list = []
for file in csv_files:
    # Читаем CSV, учитывая, что числовые поля могут быть в кавычках.
    try:
        df = pd.read_csv(file, sep=',', quotechar='"', encoding='utf-8')
    except Exception as e:
        print(f"Ошибка при чтении {file}: {e}")
        continue
    data_list.append(df)
# Объединяем все считанные DataFrame в один
data = pd.concat(data_list, ignore_index=True)
print("Объединенный DataFrame имеет размер:", data.shape)

# ================== Раздел 2: Предварительная обработка данных ==================

# Преобразование типов данных:
# Некоторые поля предполагаются числовыми, но могут быть прочитаны как строки.
numeric_cols = ['Balance','Comments','Rating','Price','Max price','Min price',
                'Average price','Sales','Revenue','Revenue potential','Lost profit',
                'Days in stock','Days with sales','Average if in stock']
for col in numeric_cols:
    if col in data.columns:
        # Удаляем кавычки и лишние символы и приводим к числу
        data[col] = data[col].astype(str).str.replace('"', '').str.replace(',', '')
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Удалим ненужные или пустые столбцы
if 'Unnamed: 22' in data.columns:
    data = data.drop(columns=['Unnamed: 22'])
# Если 'full_category' есть, оставим для анализа по категориям
# Убираем дубликаты по ключевым полям, например SKU и URL
data.drop_duplicates(subset=['SKU','URL'], inplace=True)
data.reset_index(drop=True, inplace=True)

# Обработка пропусков (NaN):
# Удаляем строки, в которых нет значимой информации для анализа.
data.dropna(subset=['Price','Comments','Rating','Sales','Balance'], inplace=True)
data.reset_index(drop=True, inplace=True)
print("Размер DataFrame после очистки NaN и дубликатов:", data.shape)

# Обработка выбросов:
# Например, отбрасываем 1% самых дорогих товаров (для устойчивости анализа).
price_threshold = data['Price'].quantile(0.99)
data = data[data['Price'] <= price_threshold]
print("Размер DataFrame после удаления ценовых выбросов:", data.shape)

# ================== Раздел 3: Анализ связи цены с другими характеристиками ==================

# Гипотеза 1: Дешёвые товары собирают больше отзывов (обратная корреляция).
# Строим диаграмму рассеяния Цена - Количество отзывов.
plt.figure(figsize=(6,4))
sns.scatterplot(x='Price', y='Comments', data=data)
plt.title("Цена vs Количество отзывов")
plt.xlabel("Цена")
plt.ylabel("Количество отзывов")
plt.tight_layout()
plt.savefig("price_vs_comments.png")

# Вычисляем коэффициент корреляции между ценой и отзывами (Pearson).
price_comments_corr, p_val_pc = pearsonr(data['Price'], data['Comments'])
print(f"Корреляция между ценой и отзывами: {price_comments_corr:.2f} (p={p_val_pc:.3f})")

# Гипотеза 2: Дорогие товары имеют выше рейтинг, но меньше отзывов.
# Диаграмма рассеяния Цена - Рейтинг.
plt.figure(figsize=(6,4))
sns.scatterplot(x='Price', y='Rating', data=data)
plt.title("Цена vs Рейтинг")
plt.xlabel("Цена")
plt.ylabel("Рейтинг")
plt.tight_layout()
plt.savefig("price_vs_rating.png")

# Диаграмма рассеяния Цена - Отзывы (с уменьшенным масштабом по оси X)
plt.figure(figsize=(6,4))
sns.scatterplot(x='Price', y='Comments', data=data)
plt.title("Цена vs Отзывы (Zoom)")
plt.xlabel("Цена")
plt.ylabel("Отзывы")
plt.xlim(0, data['Price'].quantile(0.95))  # ограничиваем по оси X для видимости
plt.tight_layout()
plt.savefig("price_vs_comments_zoom.png")

price_rating_corr, p_val_pr = pearsonr(data['Price'], data['Rating'])
print(f"Корреляция между ценой и рейтингом: {price_rating_corr:.2f} (p={p_val_pr:.3f})")

# Гипотеза 3: Популярность (продажи, наличие) зависит от цены.
# Диаграмма Цена vs Продажи.
plt.figure(figsize=(6,4))
sns.scatterplot(x='Price', y='Sales', data=data)
plt.title("Цена vs Продажи")
plt.xlabel("Цена")
plt.ylabel("Продажи")
plt.tight_layout()
plt.savefig("price_vs_sales.png")

# Диаграмма Цена vs Наличие на складе (Balance).
plt.figure(figsize=(6,4))
sns.scatterplot(x='Price', y='Balance', data=data)
plt.title("Цена vs Наличие на складе (Balance)")
plt.xlabel("Цена")
plt.ylabel("Наличие")
plt.tight_layout()
plt.savefig("price_vs_balance.png")

price_sales_corr, p_val_ps = pearsonr(data['Price'], data['Sales'])
price_balance_corr, p_val_pb = pearsonr(data['Price'], data['Balance'])
print(f"Корреляция цены и продаж: {price_sales_corr:.2f} (p={p_val_ps:.3f}); цена и наличие: {price_balance_corr:.2f} (p={p_val_pb:.3f})")

# ================== Раздел 4: Анализ рейтинга и отзывов ==================

# Гипотеза: Категории с самым высоким рейтингом показывают наивысшую удовлетворённость.
# Вычисляем средний рейтинг и число отзывов по категориям.
if 'full_category' in data.columns:
    category_stats = data.groupby('full_category').agg({'Rating': ['mean','count']}).reset_index()
    category_stats.columns = ['Category', 'AverageRating', 'Count']
    # Берём топ-10 категорий по среднему рейтингу (с достаточным числом товаров).
    top_categories = category_stats[category_stats['Count'] > 10].sort_values(by='AverageRating', ascending=False).head(10)
    print("Топ-10 категорий по среднему рейтингу:")
    print(top_categories[['Category','AverageRating','Count']])
    # Визуализация: категории и рейтинг
    plt.figure(figsize=(8,5))
    sns.barplot(data=top_categories, x='AverageRating', y='Category', orient='h')
    plt.title("Топ-10 категорий по среднему рейтингу")
    plt.xlabel("Средний рейтинг")
    plt.ylabel("Категория")
    plt.tight_layout()
    plt.savefig("top_categories_rating.png")
else:
    print("Столбца full_category нет в данных; анализ по категориям не выполнен.")

# Гипотеза: Есть корреляция между количеством отзывов и объективностью рейтинга.
# Рассчитаем корреляцию между числом отзывов и рейтингом.
comments_rating_corr, p_val_cr = pearsonr(data['Comments'], data['Rating'])
print(f"Корреляция между отзывами и рейтингом: {comments_rating_corr:.2f} (p={p_val_cr:.3f})")

# Визуализация: распределение рейтинга в зависимости от количества отзывов (heatmap).
# Для наглядности создадим биннинг по отзывам и рейтингу.
heat_data = data.copy()
heat_data['Comments_bin'] = pd.qcut(heat_data['Comments'].rank(method='first'), q=10, labels=False)
heat_data['Rating_bin'] = pd.qcut(heat_data['Rating'].rank(method='first'), q=10, labels=False)
heatmap_data = heat_data.groupby(['Comments_bin','Rating_bin']).size().unstack(fill_value=0)
plt.figure(figsize=(6,5))
sns.heatmap(heatmap_data, cmap="YlGnBu")
plt.title("Heatmap: Отзывы vs Рейтинг")
plt.xlabel("Бин рейтинга")
plt.ylabel("Бин отзывов")
plt.tight_layout()
plt.savefig("heatmap_comments_rating.png")

# ================== Раздел 5: Регрессионный анализ (дополнительно) ==================

# Линейная регрессия: Цена vs Продажи (для дополнительной проверки влияния цены на продажи).
if 'Sales' in data.columns:
    X = data['Price'].fillna(0).values.reshape(-1,1)
    y = data['Sales'].fillna(0).values
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    coef = model.coef_[0]
    intercept = model.intercept_
    print(f"Линейная регрессия (продажи ~ цена): R^2={r2:.3f}, k={coef:.3f}, b={intercept:.3f}")

# ================== Раздел 6: Выводы и рекомендации ==================
#
# print("\nВыводы:")
# print("- Между ценой и количеством отзывов обнаружена", ("отрицательная" if price_comments_corr < 0 else "положительная"),
#       f"корреляция (r={price_comments_corr:.2f}). Это подтверждает гипотезу, что дешёвые товары могут собирать больше отзывов.")
# print("- Между ценой и рейтингом корреляция", ("положительная" if price_rating_corr > 0 else "отрицательная"),
#       f"(r={price_rating_corr:.2f}). В нашем анализе дорогие товары имеют {'немного выше' if price_rating_corr > 0 else 'немного ниже'} рейтинг, что частично подтверждает вторую гипотезу.")
# print("- Между ценой и продажами/наличием корреляция", ("положительная" if price_sales_corr > 0 else "отрицательная"),
#       f"(продажи: r={price_sales_corr:.2f}, наличие: r={price_balance_corr:.2f}). Это говорит о том, что популярность товара зависит от цены.")
# print(f"- Корреляция между числом отзывов и рейтингом: r={comments_rating_corr:.2f}. Низкое значение (близкое к 0) говорит об отсутствии сильной линейной связи.")
# print("Рекомендации:")
# print("- Фокусироваться на товарах по оптимальной цене: слишком высокая цена снижает продажи, слишком низкая – снижает маржинальность.")
# print("- Поощрять получение отзывов для более дешёвых товаров (например, акциями), так как отзывы влияют на восприятие товара.")
# print("- Обращать внимание на категории с высоким средним рейтингом; анализировать, чем они отличаются.")
# print("- Учитывать, что рейтинг сам по себе мало зависит от количества отзывов, поэтому нужно искать другие факторы повышения оценки.")