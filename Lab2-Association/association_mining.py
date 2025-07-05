import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
data = {
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [
        ['Bread', 'Milk'],
        ['Bread', 'Diaper', 'Beer', 'Eggs'],
        ['Milk', 'Diaper', 'Beer', 'Coke'],
        ['Bread', 'Milk', 'Diaper', 'Beer'],
        ['Bread', 'Milk', 'Diaper', 'Coke']
    ]
}

df = pd.DataFrame(data)
print("Initial Data:\n", df)

# Step 2: One-hot encode using boolean values
df_items = df['Items'].apply(lambda x: pd.Series(True, index=x)).fillna(False)
print("\nOne-Hot Encoded Data:\n", df_items.astype(int))  # Show as 0/1 in print

# Step 3: Find frequent itemsets
frequent_itemsets = apriori(df_items, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules)

# Step 5: Print rules in detailed format
print("\nDetailed Rule Descriptions:")
for _, row in rules.iterrows():
    print(f"\nRule: {set(row['antecedents'])} -> {set(row['consequents'])}")
    print(f"Support: {row['support']:.2f}")
    print(f"Confidence: {row['confidence']:.2f}")
    print(f"Lift: {row['lift']:.2f}")

# Step 6: Save all outputs to Excel (one file)
rules_cleaned = rules.copy()
rules_cleaned['antecedents'] = rules_cleaned['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_cleaned['consequents'] = rules_cleaned['consequents'].apply(lambda x: ', '.join(list(x)))

with pd.ExcelWriter("association_mining_results.xlsx") as writer:
    df.to_excel(writer, sheet_name='Initial Data', index=False)
    df_items.astype(int).to_excel(writer, sheet_name='One-Hot Encoded', index=False)
    frequent_itemsets.to_excel(writer, sheet_name='Frequent Itemsets', index=False)
    rules_cleaned.to_excel(writer, sheet_name='Association Rules', index=False)

print("\n✅ All results saved to 'association_mining_results.xlsx'")
