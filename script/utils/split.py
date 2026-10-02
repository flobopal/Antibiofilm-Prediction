from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, GroupKFold, KFold
from scipy.spatial.distance import jensenshannon
import tqdm

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles # type: ignore

class GroupSplitter:

    def __init__(self, n_splits, train_size, random_state):
        self.group_splitter = GroupShuffleSplit(n_splits, train_size=train_size, random_state=random_state)
        self.random_splitter = ShuffleSplit(n_splits, train_size=train_size, random_state=random_state)

    def split(self, dataset: pd.DataFrame, group_column:str):
        with_groups_dataset = dataset[dataset[group_column].ne("")].copy()
        no_groups_dataset = dataset[dataset[group_column].eq("")].copy()

        if not no_groups_dataset.size:
            for itrain, itest in self.group_splitter.split(with_groups_dataset, groups = with_groups_dataset[group_column]):
                yield dataset.iloc[itrain], dataset.iloc[itest]
            return

        if not with_groups_dataset.size:
            for itrain, itest in self.random_splitter.split(no_groups_dataset):
                yield dataset.iloc[itrain], dataset.iloc[itest]
            return

        for (group_train, group_test), (random_train, random_test) in zip(
            self.group_splitter.split(with_groups_dataset, groups = with_groups_dataset[group_column]),
            self.random_splitter.split(no_groups_dataset)
        ):
            train_dataset = pd.concat([
                with_groups_dataset.iloc[group_train],
                no_groups_dataset.iloc[random_train]
            ])
            test_dataset = pd.concat([
                with_groups_dataset.iloc[group_test],
                no_groups_dataset.iloc[random_test]
            ])

            yield train_dataset, test_dataset

class GroupKSplitter(GroupSplitter):

    def __init__(self, n_splits, random_state):
        self.group_splitter = GroupKFold(n_splits, shuffle=True, random_state=random_state)
        self.random_splitter = KFold(n_splits, shuffle=True, random_state=random_state)


def js_distance(dataset:pd.DataFrame, partition:pd.DataFrame, cols:list[str]):
    ref = dataset.groupby(cols).size().div(len(dataset))
    sub = partition.groupby(cols).size().div(len(partition)).reindex(ref.index, fill_value=0)
    return jensenshannon(ref.values, sub.values)
    
def passes_min_sample(train, test, cols, min_samples):
    for col, min_sample in zip(cols, min_samples):
        if set(train[col].unique()).symmetric_difference(set(test[col].unique())):
            return False
        if train[col].value_counts().min() < min_sample:
            return False
        if test[col].value_counts().min() < min_sample:
            return False
    return True
        

def evaluate_split(
        splitter: GroupSplitter,
        group_col:str,
        split_cols: list[str],
        min_samples: list[int],
        dataset: pd.DataFrame,
        train_size:float):

    scores = []

    for train, test in splitter.split(dataset, group_col):
        if not passes_min_sample(train, test, split_cols, min_samples):
            return float('inf')
        score_train = js_distance(dataset, train, split_cols)
        score_test = js_distance(dataset, test, split_cols)
        score = train_size*score_train + (1-train_size)*score_test
        scores.append(score)
    
    return 0.7*sum(scores) / len(scores) + 0.3*max(scores)
    

def split(
        dataset: pd.DataFrame,
        group_col: str,
        split_cols: list[str],
        min_samples: list[int],
        n_splits:int,
        train_size: float = 0.8,
        n_attempts = 1_000,
        k_folds = False,
        n_out = 1):
    if not k_folds and train_size == 0:
        raise ValueError("If k_folds is False, train_size cannot be 0")
    best_seeds, best_splitters = {}, {}
    for seed in tqdm.tqdm(range(n_attempts), desc=f"Finding best split"):
        if k_folds:
            splitter = GroupKSplitter(n_splits, random_state=seed)
        else:
            splitter = GroupSplitter(n_splits, train_size=train_size, random_state=seed)
        score = evaluate_split(splitter, group_col, split_cols, min_samples, dataset, train_size)
        best_seeds[seed] = score
        best_splitters[seed] = splitter
        if len(best_seeds) <= n_out:
            continue
        worst_seed = max(best_seeds, key=best_seeds.get)
        del(best_seeds[worst_seed])
        del(best_splitters[worst_seed])
    datasets = []
    for seed in sorted(best_seeds, key=best_seeds.get):
        print(seed, best_seeds[seed])
        datasets.append(tuple(best_splitters[seed].split(dataset, group_col)))
    if n_out == 1:
        return datasets[0]
    return datasets

def split_and_cv(by_group: str, filename_prefix: str, dataset: pd.DataFrame):
    dataset["scaffold"] = dataset.curated_smiles.apply(MurckoScaffoldSmilesFromSmiles)
    dataset["binact"] = pd.qcut(dataset["pIC50"], 5)
    for i, ((train, test),) in enumerate(
        split(
            dataset,
            by_group,
            ["target_organism","binact"],
            [5,1],
            1,
            n_attempts=100_000,
            n_out=5)):
        new_data = dataset.copy()
        columns = new_data.columns.to_list()
        columns.insert(columns.index("train")+1, "kfold")
        new_data["kfold"] = None
        new_data=new_data[columns]
        new_data["train"] = False
        new_data.loc[train.index, "train"] = True
        columns = new_data.columns.to_list()
        columns.append("kfold")
        for k, (ktrain, ktest) in enumerate(
            split(
                train,
                by_group,
                ["target_organism", "binact"],
                [2,1],
                5,
                k_folds=True)):
            new_data.loc[ktest.index, "kfold"] = k
        new_data.drop(columns=["scaffold", "binact"], inplace=True)
        new_data.to_csv(f"dataset/{filename_prefix}{i}.csv")


if __name__ == "__main__":
    data = pd.read_csv("dataset/Antibiofilm data.csv", index_col=0)
    split_and_cv("scaffold", "scaffold_split_", data)
    split_and_cv("curated_smiles", "compound_split_", data)
