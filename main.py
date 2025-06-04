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
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Указываем путь к папке с ozon
# choose your character
# data_path = "..." ваш путь здесь
# data_path = '/Users/svpriymak/Downloads/ozon' # svpriymak
data_path = "/home/user/ozon_dataset/ozon"  # ysmoshchenkov
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
print(f"Найдено {len(csv_files)} CSV файлов.")

# Считываем и объединяем все файлы в один DataFrame
data_list = []
for file in csv_files:
    try:
        df = pd.read_csv(file, sep=',', quotechar='"', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Ошибка при чтении {file}: {e}")
        continue
    data_list.append(df)

data = pd.concat(data_list, ignore_index=True)
print("Всего строк:", data.shape[0])
goods_num = data.shape[0]


# Добавляем в таблицу столбец main_category
def extract_main_category(full_category, source_file):
    if isinstance(full_category, str) and '/' in full_category:
        raw = full_category.split('/')[0]
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

# Приводим все числовые поля к правильному формату
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

# Удаляем ненужные или пустые столбцы, а также дубликаты полей
if 'Unnamed: 22' in data.columns:
    data = data.drop(columns=['Unnamed: 22'])
data.drop_duplicates(subset=['SKU', 'URL'], inplace=True)
data.reset_index(drop=True, inplace=True)

# data.dropna(subset=['Price', 'Comments', 'Rating', 'Sales', 'Balance'], inplace=True)
# data.reset_index(drop=True, inplace=True)

# Обработка выбросов и фильтрация:
price_threshold = data['Price'].quantile(0.99)
data = data[data['Price'] <= price_threshold]
data = data[data['Rating'] > 0]

print("Размер DataFrame после всех фильтраций:", data.shape)

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

# Хранение результатов
res_dir = "./results"
os.makedirs(res_dir, exist_ok=True)


def save_fig(name, dpi=150):
    plt.tight_layout()
    plt.savefig(f"{res_dir}/{name}", dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Сохранен: {name}")


# Анализ связи цены и характеристик
# Гипотеза 1: Цена vs Отзывы
plt.figure(figsize=(10, 6))
sns.regplot(
    data=data.sample(5000, random_state=42),
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
print("Корреляции Пирсона (log10 цены):")
valid = data[(data["Price"] > 0) & data["Comments"].notna()]
if len(valid) >= 2:
    r_pr, p_pr = pearsonr(np.log10(valid["Price"]), valid["Rating"])
    r_pc, p_pc = pearsonr(np.log10(valid["Price"]), valid["Comments"])
    r_rc, p_rc = pearsonr(valid["Rating"], valid["Comments"])

    print(f"Цена — Рейтинг : r={r_pr:.3f}, p={p_pr:.2e}")
    print(f"Цена — Отзывы  : r={r_pc:.3f}, p={p_pc:.2e}")
    print(f"Рейтинг — Отзывы: r={r_rc:.3f}, p={p_rc:.2e}\n")
else:
    print("Недостаточно данных для корреляций.")

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
if p_val < 1e-323:
    p_display = "p < 1e-323 (практически 0)"
else:
    p_display = f"p = {p_val:.4e}"
print(f"Дисперсионный анализ (ANOVA): F-value={f_val:.2f}, {p_display}\n")

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
    if k_pvalue < 0.0001:
        p_value_str = "< 0.0001"
    else:
        p_value_str = f"{k_pvalue:.4f}"
    print(f"Корреляция Кендалла (Цена-Продажи): {kendall_corr:.3f}, p-value: {p_value_str}\n")

# Анализ рейтингов и отзывов
# Категории с лучшими рейтингами
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

# Распределение цен по категориям
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=filtered_data,
    x='Price',
    y='main_category',
    palette='RdYlGn',
    showfliers=False,
    orient='h',
    width=0.7
)
plt.xscale('log')
plt.title("Распределение цен по основным категориям", fontsize=16)
plt.xlabel("Цена, руб (лог. шкала)", fontsize=12)
plt.ylabel("Категория", fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.yticks(fontsize=10)
plt.tight_layout()
save_fig("category_price_distribution.png")

# сsv файл со средним рейтингом, средним кол-вом отзывов и общим кол-вом товаров в категории
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

# Гипотеза 4: Отзывы vs Рейтинг
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

# Распределение цен
plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], bins=50, kde=True, color='royalblue')
plt.title("Распределение цен на товары", fontsize=14)
plt.xlabel("Цена (руб.)", fontsize=12)
plt.ylabel("Количество товаров", fontsize=12)
plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
save_fig("price_distribution.png")

# Матрица корреляции Спирмена по 5 параметрам
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

# Комбинированный график
fig, ax1 = plt.subplots(figsize=(12, 5))
sns.barplot(ax=ax1, x=cat_stats.index, y=cat_stats["mean_rating"], palette="Blues_d")
ax1.set_ylim(0, 5)
ax1.set_ylabel("Средний рейтинг")
ax1.set_xlabel("Основные категории")
plt.xticks(rotation=45, ha="right", fontsize=9)

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

plt.title("ТОП-категории: рейтинг и отзывы")
plt.tight_layout()
save_fig("top_cats_rating_and_reviews.png")

# Результаты
print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ АНАЛИЗА E-COMMERCE ДАННЫХ НА ДАТАСЕТЕ OZON")
print("=" * 50 + "\n")

print(f"Общее количество товаров: {goods_num}")
print(f"Количество категорий: {data['main_category'].nunique()}")
print(f"Средний рейтинг: {data['Rating'].mean():.2f}")
print(f"Медиана рейтинга: {valid['Rating'].median():.2f}")
print(f"Медиана отзывов: {valid['Comments'].median():.1f}")
print(f"Медианная цена: {data['Price'].median():.2f} руб\n")

print("Основные зависимости:")
print(f"1. Корреляция цена/отзывы: {pearsonr(np.log10(data['Price'] + 1), np.log10(data['Comments'] + 1))[0]:.3f}")
print(f"2. Корреляция цена/рейтинг: {kendalltau(data['Price'], data['Rating'])[0]:.3f}")

print("\nВыводы:")
print("- Товары масс-маркет получают больше отзывов")
print("- Премиум сегмент имеет более высокие рейтинги")
print("- Оптимальный ценовой диапазон: 1-5 тыс. руб")

print("\nАнализ завершен! Результаты сохранены в:", os.path.abspath(res_dir))
