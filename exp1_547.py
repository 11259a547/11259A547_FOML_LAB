import pandas as pd
import seaborn as sns
data = sns.load_dataset("titanic")
data = data[["survived","pclass","sex","age","fare","embarked"]].copy()
print(data.isnull().sum())
data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna("S")
data = pd.get_dummies(data, columns=["sex","embarked"], dtype=int)
print(data.head())
print("Any blanks left?", data.isnull().values.any())

new = pd.DataFrame([{"pclass":2,"sex":"female","age":27,"fare":30,"embarked":"C"}])
new = pd.get_dummies(new, columns=["sex","embarked"], dtype=int)
new = new.reindex(columns=data.drop("survived", axis=1).columns, fill_value=0)
print(new)

import matplotlib.pyplot as plt
raw = sns.load_dataset("titanic")[["survived","pclass","sex","age","fare","embarked"]]
blanks = raw.isnull().sum()
plt.bar(blanks.index, blanks.values, color="#2F49D1")
plt.xticks(rotation=30); plt.ylabel("empty cells")
plt.title("Empty cells before cleaning (after cleaning: 0)")
plt.tight_layout(); plt.show()
