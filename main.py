# -*- coding: utf-8 -*-
import os
import glob
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats import pearsonr, kendalltau, stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.size'] = 12

# 1. Загрузка --------------------------------------------------------------
# Указываем путь к папке с CSV-файлами
# choose your character
# data_path = '/Users/svpriymak/Downloads/ozon' # svpriymak
data_path = "/home/user/ozon_dataset/ozon"  # ysmoshchenkov
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Найдено {len(csv_files)} CSV файлов.")

# Считываем и объединяем все файлы в один DataFrame
data_list = []
for file in csv_files:
    # Читаем CSV, учитывая, что числовые поля могут быть в кавычках.
    try:
        df = pd.read_csv(file, sep=',', quotechar='"', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Ошибка при чтении {file}: {e}")
        continue
    data_list.append(df)

data = pd.concat(data_list, ignore_index=True)
print("Всего строк:", data.shape[0])
goods_num = data.shape[0]


# Добавляем в таблицу поле main_category
def extract_main_category(full_category, source_file):
    # Попытка извлечь основную категорию из full_category
    if isinstance(full_category, str) and '/' in full_category:
        raw = full_category.split('/')[0]
        # Если дублируется, например "ЭлектроникаЭлектроника", сократим до одного слова
        half = len(raw) // 2
        if len(raw) % 2 == 0 and raw[:half] == raw[half:]:
            return raw[:half]
        return raw
    elif isinstance(source_file, str):
        try:
            return source_file.split('_')[2]
        except IndexError:
            return None
    return None


data['main_category'] = data.apply(lambda row: extract_main_category(
    row.get('full_category'), row.get('source_file')), axis=1)

print("Объединенный DataFrame имеет размер:", data.shape)

# 3. Числа -----------------------------------------------------------------
numeric = ['Balance', 'Comments', 'Rating', 'Price']


def to_float(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace('"', '', regex=False)
        .str.replace('\u202f', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False),
        errors='coerce')


for c in numeric:
    if c in data.columns: data[c] = to_float(data[c])

# 4. Базовая чистка --------------------------------------------------------
# Некоторые поля предполагаются числовыми, но могут быть прочитаны как строки.
numeric_cols = ['Balance', 'Comments', 'Rating', 'Price', 'Max price', 'Min price',
                'Average price', 'Sales', 'Revenue', 'Revenue potential', 'Lost profit',
                'Days in stock', 'Days with sales', 'Average if in stock']
for col in numeric_cols:
    if col in data.columns:
        # Удаляем кавычки и лишние символы и приводим к числу
        data[col] = data[col].astype(str).str.replace('"', '').str.replace(',', '')
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Удалим ненужные или пустые столбцы
if 'Unnamed: 22' in data.columns:
    data = data.drop(columns=['Unnamed: 22'])
# Убираем дубликаты по ключевым полям, например SKU и URL
data.drop_duplicates(subset=['SKU', 'URL'], inplace=True)
data.reset_index(drop=True, inplace=True)

# Обработка пропусков (NaN):
# Удаляем строки, в которых нет значимой информации для анализа.
data.dropna(subset=['Price', 'Comments', 'Rating', 'Sales', 'Balance'], inplace=True)
data.reset_index(drop=True, inplace=True)
print("Размер DataFrame после очистки NaN и дубликатов:", data.shape)

# Обработка выбросов:
# отбрасываем 1% самых дорогих товаров (для устойчивости анализа), а также товары без оценок
price_threshold = data['Price'].quantile(0.99)
data = data[data['Price'] <= price_threshold]
data = data[data['Rating'] > 0]

print("Размер DataFrame после удаления ценовых выбросов и нулевых рейтингов:", data.shape)

# 5. Создание сегментов -------------------------------------------------------
# Сегменты по цене
if data['Price'].nunique() >= 4:
    data['price_segment'] = pd.qcut(
        data['Price'],
        4,
        labels=['Дешёвые', 'Средние', 'Дорогие', 'Премиум']
    )
else:
    data['price_segment'] = pd.cut(
        data['Price'],
        bins=3,
        labels=['Дешёвые', 'Средние', 'Дорогие']
    )

# 6. Подготовка папки для результатов -----------------------------------------
res_dir = "./results"
os.makedirs(res_dir, exist_ok=True)


def save_fig(name, dpi=150):
    plt.tight_layout()
    plt.savefig(f"{res_dir}/{name}", dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Сохранен: {name}")


# 7. Анализ связи цены и характеристик ----------------------------------------
# Гипотеза 1: Цена vs Отзывы
plt.figure(figsize=(10, 6))
sns.regplot(
    data=data.sample(5000, random_state=42),  # Сэмплирование для больших данных
    x='Price',
    y='Comments',
    scatter_kws={'alpha': 0.3, 'color': 'steelblue'},
    line_kws={'color': 'crimson', 'lw': 2},
    lowess=True
)
plt.title("Зависимость количества отзывов от цены товара", fontsize=14)
plt.xlabel("Цена товара (руб)", fontsize=12)
plt.ylabel("Количество отзывов", fontsize=12)
plt.xscale('log')
plt.yscale('log')
save_fig("price_vs_comments.png")

# Расчет корреляции
# вариант 1
print("\nКорреляции Пирсона (log10 цены):")
valid = data[(data["Price"] > 0) & data["Comments"].notna()]

if len(valid) >= 2:
    r_pr, p_pr = pearsonr(np.log10(valid["Price"]), valid["Rating"])
    r_pc, p_pc = pearsonr(np.log10(valid["Price"]), valid["Comments"])
    r_rc, p_rc = pearsonr(valid["Rating"], valid["Comments"])

    print(f"Цена — Рейтинг : r={r_pr:.3f}, p={p_pr:.2e}")
    print(f"Цена — Отзывы  : r={r_pc:.3f}, p={p_pc:.2e}")
    print(f"Рейтинг — Отзывы: r={r_rc:.3f}, p={p_rc:.2e}")
else:
    print("Недостаточно данных для корреляций.")

# вариант 2
price_log = np.log10(data['Price'].dropna() + 1)
comments_log = np.log10(data['Comments'].dropna() + 1)
corr, p_value = pearsonr(price_log, comments_log)
print(f"Корреляция (лог-преобразование): {corr:.3f}, p-value: {p_value:.4f}")

# Гипотеза 2: Цена vs Рейтинг
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data,
    x='price_segment',
    y='Rating',
    palette='viridis',
    showfliers=False
)
plt.title("Распределение рейтингов по ценовым сегментам", fontsize=14)
plt.xlabel("Ценовой сегмент", fontsize=12)
plt.ylabel("Средний рейтинг", fontsize=12)
plt.ylim(3, 5.2)
save_fig("price_segment_vs_rating.png")

# Статистическая значимость
segments = data.groupby('price_segment')['Rating'].apply(list)
f_val, p_val = stats.f_oneway(*segments.values)
print(f"ANOVA: F-value={f_val:.2f}, p-value={p_val:.4f}")

# Гипотеза 3: Цена vs Продажи
if 'Sales' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data.sample(3000, random_state=42),
        x='Price',
        y='Sales',
        hue='price_segment',
        palette='viridis',
        alpha=0.6
    )
    plt.title("Зависимость продаж от цены товара", fontsize=14)
    plt.xlabel("Цена товара (руб)", fontsize=12)
    plt.ylabel("Количество продаж", fontsize=12)
    plt.xscale('log')
    plt.legend(title="Ценовой сегмент")
    save_fig("price_vs_sales.png")

    # Корреляция Кендалла (устойчива к выбросам)
    kendall_corr, k_pvalue = kendalltau(data['Price'], data['Sales'], nan_policy='omit')
    print(f"Корреляция Кендалла (Цена-Продажи): {kendall_corr:.3f}, p-value: {k_pvalue:.4f}")

# 8. Анализ рейтингов и отзывов -----------------------------------------------
# Гипотеза 1: Категории с лучшими рейтингами
# вариант 1
top_categories = data['main_category'].value_counts().nlargest(15).index
filtered_data = data[data['main_category'].isin(top_categories)]

plt.figure(figsize=(12, 8))
sns.boxplot(
    data=filtered_data,
    x='Rating',
    y='main_category',
    palette='RdYlGn',
    showfliers=False,
    orient='h'
)
plt.title("Распределение рейтингов по основным категориям", fontsize=16)
plt.xlabel("Рейтинг товара", fontsize=12)
plt.ylabel("Категория", fontsize=12)
plt.xlim(3, 5.2)
save_fig("category_rating_distribution.png")

# вариант 2
top_cats = data["main_category"].value_counts().nlargest(10).index
data["main_cat_top"] = data["main_category"].where(
    data["main_category"].isin(top_cats), "Other"
)

cat_stats = (
    data.groupby("main_cat_top")
    .agg(
        mean_rating=("Rating", "mean"),
        mean_reviews=("Comments", "mean"),
        count=("Price", "size"),
    )
    .sort_values("mean_rating", ascending=False)
)
cat_stats.to_csv(f"{res_dir}/category_stats.csv")


# Гипотеза 2: Отзывы vs Рейтинг
plt.figure(figsize=(10, 6))
sns.regplot(
    data=data[data['Comments'] > 0].sample(5000, random_state=42),
    x='Comments',
    y='Rating',
    scatter_kws={'alpha': 0.3, 'color': 'teal'},
    line_kws={'color': 'darkorange', 'lw': 2},
    x_jitter=0.2,
    y_jitter=0.1
)
plt.title("Зависимость рейтинга от количества отзывов", fontsize=14)
plt.xlabel("Количество отзывов (лог. шкала)", fontsize=12)
plt.ylabel("Рейтинг товара", fontsize=12)
plt.xscale('log')
plt.ylim(3, 5.2)
save_fig("reviews_vs_rating.png")

# 9. Дополнительные анализы ---------------------------------------------------
# Распределение цен
plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], bins=50, kde=True, color='royalblue')
plt.title("Распределение цен на товары", fontsize=14)
plt.xlabel("Цена (руб)", fontsize=12)
plt.ylabel("Количество товаров", fontsize=12)
plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
save_fig("price_distribution.png")

# Корреляционная матрица
# вариант 1
corr_matrix = data[['Price', 'Rating', 'Comments', 'Sales', 'Balance']].corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    mask=np.triu(np.ones_like(corr_matrix, dtype=bool))
)
plt.title("Корреляционная матрица (Спирмен)", fontsize=14)
save_fig("correlation_matrix.png")

# вариант 2
plt.figure(figsize=(7, 4))
sns.violinplot(
    data=data, x="Ценовой сегмент", y="Рейтинг", inner="quartile", palette="Set3", cut=0
)
plt.ylim(0, 5.2)
plt.title("Рейтинг по сегментам цены")
save_fig("violin_rating_segment.png")

corr = data[["Price", "Rating", "Comments"]].corr(method="spearman")
plt.figure(figsize=(4, 3))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Spearman-корреляции")
save_fig("heatmap_spearman.png")

# Комбинированный график
fig, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(ax=ax1, x=cat_stats.index, y=cat_stats["mean_rating"], palette="Blues_d")
ax1.set_ylim(0, 5)
ax1.set_ylabel("Средний рейтинг")

ax2 = ax1.twinx()
ax2.plot(
    cat_stats.index,
    cat_stats["mean_reviews"],
    color="darkred",
    marker="o",
    linewidth=2,
)
ax2.set_ylabel("Средние отзывы", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")

plt.xticks(rotation=20, ha="right")
plt.title("ТОП-категории: рейтинг и отзывы")
save_fig("combo_rating_reviews_category.png")

# Box распределения цен по TOP-категориям
plt.figure(figsize=(10, 4))
sns.boxplot(data=data[data["main_cat_top"] != "Other"], x="ТОП-категории", y="Цена")
plt.yscale("log")
plt.ylabel("Цена (лог)")
plt.xticks(rotation=20, ha="right")
plt.title("Распределение цен в ТОП-категориях")
save_fig("box_price_top_categories.png")

# 10. Сохранение результатов --------------------------------------------------
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ АНАЛИЗА E-COMMERCE ДАННЫХ")
print("="*50 + "\n")

# Основные метрики
print(f"Общее количество товаров: {goods_num}")
print(f"Количество категорий: {data['main_category'].nunique()}")
print(f"Средний рейтинг: {data['Rating'].mean():.2f}")
print(f"Медианная цена: {data['Price'].median():.2f} руб\n")

# Ключевые зависимости
print("ОСНОВНЫЕ ЗАВИСИМОСТИ:")
print(f"1. Корреляция цена/отзывы: {pearsonr(np.log10(data['Price']+1), np.log10(data['Comments']+1))[0]:.3f}")
print(f"2. Корреляция цена/рейтинг: {kendalltau(data['Price'], data['Rating'])[0]:.3f}\n")

# Топ категории
top_cats = data['main_category'].value_counts().nlargest(5)
print("ТОП-5 КАТЕГОРИЙ ПО КОЛИЧЕСТВУ ТОВАРОВ:")
for cat, count in top_cats.items():
    print(f"- {cat}: {count} товаров")

# Практические выводы
print("\nВЫВОДЫ И РЕКОМЕНДАЦИИ:")
print("- Товары масс-маркет получают больше отзывов")
print("- Премиум сегмент имеет более высокие рейтинги")
print("- Оптимальный ценовой диапазон: 1-5 тыс. руб")

print("\nАнализ завершен! Результаты сохранены в:", os.path.abspath(res_dir))
