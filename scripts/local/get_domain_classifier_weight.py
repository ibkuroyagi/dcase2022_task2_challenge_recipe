import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from asd_tools.utils import seed_everything
import warnings
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from asd_tools.utils import sigmoid

warnings.simplefilter("ignore")
seed_everything(42)


def reset_columns(df):
    df_col = df.columns
    df = df.T.reset_index(drop=False).T
    for i in range(df.shape[1]):
        rename_col = {i: "_".join(df_col[i])}
        df = df.rename(columns=rename_col)
    df = df.drop(["level_0", "level_1"], axis=0).reset_index()
    return df


machines = ["gearbox", "bearing", "fan", "valve", "slider", "ToyCar", "ToyTrain"]
models = [
    "domain_classifier_0.1/best_loss/best_loss",
    "domain_classifier_0.1/checkpoint-50epochs/checkpoint-50epochs",
]
n_section = 6
domain_scores = np.zeros((len(machines), 2, 3))  # (machine, post_process, section)
n_sample = 200
n_model = len(models) * 2
domain_weights = np.zeros((len(machines), n_model, n_section * n_sample))
is_target = np.zeros((len(machines), 3 * n_sample))
hp = 1
path_list = []
for no, num in enumerate(models):
    for i, machine in enumerate(machines):
        eval_df = pd.read_csv(f"exp/{machine}/{num}_eval.csv")
        valid_df = pd.read_csv(f"exp/{machine}/{num}_valid.csv")
        eval_df["is_target"] = eval_df["path"].map(lambda x: int("target" in x))
        valid_df["is_target"] = valid_df["path"].map(lambda x: int("target" in x))
        feature_cols = (
            ["pred_machine", "embed_norm", "is_target_pred"]
            + [f"pred_section{j}" for j in range(6)]
            + [f"e{i}" for i in range(128)]
        )
        for section in range(n_section):
            input_valid = valid_df[
                (valid_df["section"] == section)
                & (valid_df["is_target"] == 0)
                & (valid_df["path"].map(lambda x: int("1.00" in x)))
            ]
            input_eval = eval_df[
                (eval_df["section"] == section)
                & (eval_df["path"].map(lambda x: int("1.00" in x)))
            ]

            gmm = GaussianMixture(n_components=hp, random_state=42)
            gmm.fit(input_valid[feature_cols])
            gmm_score = np.tanh(-gmm.score_samples(input_eval[feature_cols]))
            input_eval["im_is_target_pred"] = gmm_score
            agg_df = reset_columns(
                input_eval[
                    [
                        "path",
                        "is_normal",
                        "is_target",
                        "is_target_pred",
                        "im_is_target_pred",
                    ]
                ]
                .groupby("path")
                .agg(["mean", "max"])
            )
            if (no == 0) and (i == 0):
                path_list += list(agg_df["path"].map(lambda x: x.split("/")[-1]).values)
            domain_weights[
                i, no * 2, section * n_sample : (section + 1) * n_sample
            ] = agg_df["is_target_pred_mean"].astype(float)
            domain_weights[
                i, no * 2 + 1, section * n_sample : (section + 1) * n_sample
            ] = agg_df["im_is_target_pred_mean"].astype(float)
            if section in [0, 1, 2]:
                is_target[i, section * n_sample : (section + 1) * n_sample] = agg_df[
                    "is_target_max"
                ].astype(int)
                auc = roc_auc_score(
                    agg_df["is_target_max"].astype(int),
                    agg_df["is_target_pred_mean"].astype(float),
                )
                domain_scores[i, 0, section] = auc
                print(f"{machine} section{section} mean auc:{auc*100:.2f}")
                auc = roc_auc_score(
                    agg_df["is_target_max"].astype(int),
                    agg_df["im_is_target_pred_mean"].astype(float),
                )
                domain_scores[i, 1, section] = auc


scores = np.zeros((len(machines), 5))
weights = np.ones((len(machines), 5, n_model))
# Calculate weights for mixing each machine type and evaluate with 5fold-CV
for i, machine in enumerate(machines):
    sections = np.zeros(n_sample * 3)
    for sec in range(3 * 2 * 2):
        sections[50 * sec : 50 * (sec + 1)] = sec

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(sections, sections)):
        y_pred = domain_weights[i, :, train_idx]
        y_true = is_target[i, train_idx]

        def calc_score(weight=np.ones(n_model)):
            y_pred_blended = np.dot(y_pred, weight)
            return 1.0 - roc_auc_score(y_true, y_pred_blended)

        max_iter = 20
        initial_weights = np.random.uniform(size=(max_iter, n_model))
        score = 1
        for iter in range(max_iter):
            bounds = [(0, 1)] * n_model
            result = minimize(
                calc_score, initial_weights[iter], method="Nelder-Mead", bounds=bounds
            )
            if score > result["fun"]:
                score = result["fun"]
                weight = result["x"]
                weight /= np.sum(weight)
        print(f"{machine}, fold{fold}, auc:{1-score:.4f}")
        scores[i, fold] = score
        weights[i, fold] = weight
    print(weights[i].mean(0))
    y_pred_blended_ = np.dot(weights[i].mean(0), domain_weights[i])
    ave_auc = roc_auc_score(is_target[i], y_pred_blended_[:600])
    print(f"{machine}, blended auc:{ave_auc:.4f}")

label = [
    "best",
    "best gmm",
    "50 epoch",
    "50 epoch gmm",
]
plt.figure(figsize=(10, 14))
for i, machine in enumerate(machines):
    y_pred_blended = np.dot(weights[i].mean(0), domain_weights[i])
    ave_auc = roc_auc_score(is_target[i], y_pred_blended[:600])
    plt.subplot(len(machines), 1, i + 1)
    plt.bar(
        np.arange(n_model),
        weights[i].mean(0),
        yerr=weights[i].std(0),
        tick_label=label,
        align="center",
    )
    plt.title(f"{machine} auc:{ave_auc:.2f}")
    plt.grid(True)
plt.tight_layout()
os.makedirs("exp/all/weights", exist_ok=True)
plt.savefig(f"exp/all/weights/model{n_model}.jpg")

columns = [f"w{i}" for i in range(n_model)]
columns += [f"std{i}" for i in range(n_model)]
weight_df = pd.DataFrame(
    np.concatenate([weights.mean(1), weights.std(1)], axis=1),
    columns=columns,
    index=machines,
)
weight_df.to_csv(f"exp/all/weights/weight{n_model}.csv")
domain_weight_fixed = np.zeros((1200, 7))
for div in [1, 0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
    machines = ["gearbox", "bearing", "fan", "valve", "slider", "ToyCar", "ToyTrain"]
    weight = pd.read_csv(f"exp/all/weights/weight{n_model}.csv", index_col=0)
    for i, machine in enumerate(machines):
        domain_weight_fixed[:, i] = np.dot(
            weight.loc[machine, [f"w{i}" for i in range(n_model)]].values,
            domain_weights[i],
        )
        domain_weight_fixed[:, i] -= domain_weight_fixed[:, i].mean()
        domain_weight_fixed[:, i] /= domain_weight_fixed[:, i].std() * div
        auc = roc_auc_score(is_target[i], sigmoid(domain_weight_fixed[:600, i])) * 100
        # print(f"{machine} auc {auc:.4f}")
    domain_weight_df = pd.DataFrame(sigmoid(domain_weight_fixed), columns=machines)
    domain_weight_df["path"] = path_list
    domain_weight_df.to_csv(
        f"exp/all/weights/is_target_weight{n_model}_div{div}.csv", index=False
    )
    plt.figure(figsize=(6, 20))
    for i in range(7):
        auc = roc_auc_score(is_target[i], sigmoid(domain_weight_fixed[:600, i]))
        plt.subplot(7, 1, i + 1)
        plt.hist(sigmoid(domain_weight_fixed[:600, i]), bins=50, alpha=0.4, label="dev")
        plt.hist(
            sigmoid(domain_weight_fixed[600:, i]), bins=50, alpha=0.4, label="eval"
        )
        plt.title(f"{machines[i]} auc:{auc:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"exp/all/weights/is_target_weight{n_model}_hist_div{div}.jpg")
