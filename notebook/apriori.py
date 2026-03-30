# Apriori Algorithm for Market Basket Analysis
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load preprocessed data
df = pd.read_csv("data/weather_vn_cleaned.csv")
df = df.fillna(False)
df = df.astype(bool)

frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True, max_len = 3, verbose=1, low_memory=True)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)
rules = rules[rules["lift"] > 1]

# Sắp xếp theo confidence
rules = rules.sort_values(by="confidence", ascending=False)

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
