import pandas as pd

label_mappping = {"entailment": 0, "neutral": 1, "contradiction": 2}

anli_test = pd.read_csv("anli.test.tsv", sep="\t")
anli_test["label"] = anli_test["label"].map(label_mappping)
anli_test.to_parquet("amnli.test.parquet")

anli = pd.read_csv("anli.dev.tsv", sep="\t")
xnli = pd.read_csv("xnli.dev.tsv", sep="\t")

es = []
en = []
for a_id in anli["id"].unique():
    row = xnli.iloc[a_id,:].to_dict()
    es.append({"id": a_id, "language": "es", "premise": row["sentence1"], "hypothesis": row["sentence2"], "label": row["gold_label"]})
    row_en = xnli[(xnli["language"] == "en") & (xnli["promptID"] == row["promptID"]) & (xnli["gold_label"] == row["gold_label"])].iloc[0].to_dict()
    en.append({"id": a_id, "language": "en", "premise": row_en["sentence1"], "hypothesis": row_en["sentence2"], "label": row_en["gold_label"]})

anli_matched = pd.concat([anli, pd.DataFrame(es), pd.DataFrame(en)], axis=0)
anli_matched["label"] = anli_matched["label"].map(label_mappping)

anli_matched.to_parquet("amnli.dev.parquet")