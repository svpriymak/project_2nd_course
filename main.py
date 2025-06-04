# -*- coding: utf-8 -*-
import os, glob, warnings, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from  scipy.stats import pearsonr
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# 1. Загрузка --------------------------------------------------------------
data_path = '/Users/svpriymak/Downloads/ozon'
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
frames = []
for f in csv_files:
    try:
        df = pd.read_csv(f, sep=',', quotechar='"', encoding='utf-8', low_memory=False)
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    except Exception as e:
        print("Ошибка:", f, e)
data = pd.concat(frames, ignore_index=True)
print("Cклеено строк:", data.shape[0])

# 2. main_category ---------------------------------------------------------
def main_cat(full, src):
    if isinstance(full, str) and '/' in full:
        head = full.split('/')[0]
        h=len(head)//2
        return head[:h] if len(head)%2==0 and head[:h]==head[h:] else head
    try: return src.split('_')[2]
    except: return None
data["main_category"] = data.apply(lambda r: main_cat(r.get("full_category"), r["source_file"]), axis=1)

# 3. Числа -----------------------------------------------------------------
numeric = ['Balance','Comments','Rating','Price']
def to_float(s):
    return pd.to_numeric(
        s.astype(str)
         .str.replace('"','', regex=False)
         .str.replace('\u202f','', regex=False)
         .str.replace(' ','', regex=False)
         .str.replace(',','.', regex=False),
        errors='coerce')
for c in numeric:
    if c in data.columns: data[c] = to_float(data[c])

# 4. Базовая чистка --------------------------------------------------------
data = data.dropna(subset=['Price','Rating'])
data = data[(data['Rating'] > 0) & (data['Rating'] <= 5)]           # <-- ключевая строка
thr = data['Price'].quantile(0.99)
data = data[data['Price'] <= thr]
print("После чистки:", data.shape[0], "строк")

# 5. Сегменты --------------------------------------------------------------
if data['Price'].nunique() >= 4:
    data['price_segment'] = pd.qcut(data['Price'], 4,
                                    labels=['Дешёвые','Средние','Дорогие','Премиум'])
else:
    data['price_segment'] = "Один сегмент"

# 6. Папка результатов -----------------------------------------------------
res = "./results"; os.makedirs(res, exist_ok=True)
def save(name): plt.tight_layout(); plt.savefig(f"{res}/{name}", dpi=110); plt.close()

# 7. scatter (Price–Rating / Price–Comments) ------------------------------
plt.figure(figsize=(7,4))
sns.scatterplot(data=data, x='Price', y='Rating', alpha=.25)
plt.xscale('log'); plt.title("Цена vs Рейтинг"); plt.xlabel("Цена, руб (log)")
save("price_vs_rating.png")
r,p = pearsonr(np.log10(data['Price']), data['Rating']); print("corr(P,R)=",round(r,3))

if 'Comments' in data and data['Comments'].notna().any():
    plt.figure(figsize=(7,4))
    sns.scatterplot(data=data, x='Price', y='Comments', alpha=.25)
    plt.xscale('log'); plt.title("Цена vs Отзывы"); plt.ylabel("Отзывы")
    save("price_vs_comments.png")

# 8. violin по сегментам ---------------------------------------------------
plt.figure(figsize=(7,4))
sns.violinplot(data=data, x='price_segment', y='Rating', inner='quartile', palette='Set3', cut=0)
plt.ylim(0,5.2); plt.title("Рейтинг по сегментам цены"); save("violin_rating_segment.png")

# 9. TOP-категории: рейтинг-столбцы, отзывы-линия --------------------------
topN = 10
top = data['main_category'].value_counts().nlargest(topN).index
data['main_cat_top'] = data['main_category'].where(data['main_category'].isin(top),'Other')
cat = (data.groupby('main_cat_top')
         .agg(mean_rating=('Rating','mean'),
              mean_reviews=('Comments','mean'),
              count=('Price','size'))
         .sort_values('mean_rating', ascending=False))
cat.to_csv(f"{res}/category_stats.csv")

fig, ax1 = plt.subplots(figsize=(10,4))
sns.barplot(ax=ax1, x=cat.index, y=cat['mean_rating'], palette='Blues_d')
ax1.set_ylim(0,5); ax1.set_ylabel("Средний рейтинг")
ax2 = ax1.twinx()
ax2.plot(cat.index, cat['mean_reviews'], color='darkred', marker='o', linewidth=2)
ax2.set_ylabel("Средние отзывы", color='darkred'); ax2.tick_params(axis='y', labelcolor='darkred')
plt.xticks(rotation=20, ha='right'); plt.title("TOP-категории: рейтинг и отзывы")
save("combo_rating_reviews_category.png")

print("\nГОТОВО ✔ Файлы →", os.path.abspath(res))

