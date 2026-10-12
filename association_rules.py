import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample Dataset
data = {
    'Milk': [1, 1, 0, 1, 0],
    'Bread': [1, 1, 1, 0, 1],
    'Butter': [0, 1, 1, 1, 0],
    'Eggs': [1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Frequent Itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])