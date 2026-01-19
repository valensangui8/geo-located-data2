import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def _split_tags(tags_str: str) -> List[str]:
    if not tags_str:
        return []
    return [t.strip().lower() for t in str(tags_str).split(',') if t.strip()]


def _prepare_transactions(cluster_df: pd.DataFrame,
                           sample_size: int = 5000,
                           max_tags: int = 30) -> List[List[str]]:
    if len(cluster_df) > sample_size:
        cluster_df = cluster_df.sample(n=sample_size, random_state=42)

    tag_lists = cluster_df['tags'].dropna().map(_split_tags).tolist()
    all_tags = [t for tags in tag_lists for t in tags]
    if not all_tags:
        return []

    tag_counts = pd.Series(all_tags).value_counts().head(max_tags)
    keep_tags = set(tag_counts.index.tolist())

    transactions = []
    for tags in tag_lists:
        filtered = [t for t in tags if t in keep_tags]
        if len(filtered) >= 2:
            transactions.append(filtered)
    return transactions


def mine_rules_for_cluster(cluster_df: pd.DataFrame,
                           min_support: float = 0.02,
                           min_confidence: float = 0.3,
                           min_lift: float = 1.2,
                           max_rules: int = 5) -> Tuple[List[dict], List[str]]:
    transactions = _prepare_transactions(cluster_df)
    if not transactions:
        return [], []

    encoder = TransactionEncoder()
    arr = encoder.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(arr, columns=encoder.columns_)

    itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return [], []

    rules = association_rules(itemsets, metric='confidence', min_threshold=min_confidence)
    if rules.empty:
        return [], []

    rules = rules[rules['lift'] >= min_lift]
    if rules.empty:
        return [], []

    rules = rules.sort_values(['lift', 'confidence'], ascending=False).head(max_rules)

    rows = []
    display = []
    for _, row in rules.iterrows():
        antecedents = sorted(list(row['antecedents']))
        consequents = sorted(list(row['consequents']))
        rows.append({
            'antecedents': ', '.join(antecedents),
            'consequents': ', '.join(consequents),
            'support': float(row['support']),
            'confidence': float(row['confidence']),
            'lift': float(row['lift'])
        })
        display.append(f"{' + '.join(antecedents)} → {' + '.join(consequents)}")

    return rows, display


def mine_rules_per_cluster(df: pd.DataFrame, labels: np.ndarray,
                           min_support: float = 0.02,
                           min_confidence: float = 0.3,
                           min_lift: float = 1.2,
                           max_rules: int = 5) -> Tuple[Dict[int, List[str]], pd.DataFrame]:
    df_work = df.copy()
    df_work['cluster'] = labels

    rules_by_cluster = {}
    rows = []

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster_df = df_work[df_work['cluster'] == cluster_id]
        rule_rows, display = mine_rules_for_cluster(
            cluster_df,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_rules=max_rules
        )
        rules_by_cluster[cluster_id] = display
        for r in rule_rows:
            rows.append({
                'cluster_id': cluster_id,
                **r
            })

    return rules_by_cluster, pd.DataFrame(rows)
