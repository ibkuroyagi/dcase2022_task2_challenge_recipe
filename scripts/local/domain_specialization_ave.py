import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from asd_tools.utils import seed_everything, hauc, zscore
import warnings
from scipy.stats import hmean

warnings.simplefilter("ignore")


seed_everything(42)

fnames = ["score.csv", "score_target.csv", "score_embed.csv", "score_target_embed.csv"]
score_csv_list = []
for fname in fnames:
    score_csv_list += glob.glob(f"exp/all/**/{fname}")
score_csv_list.sort()
print(f"ALL:{len(score_csv_list)}")
rows_list = []
n_top = 5
seed_df_dict = {}  # {"fan":seed_df for fan}
machines = ["bearing", "fan", "gearbox", "valve", "slider", "ToyCar", "ToyTrain"]
domains = ["source", "target"]
hauc_list = []
std_list = []
row_list = []
n_mix = 20
for domain in domains:
    top_list = []
    for csv_path in score_csv_list:
        tmp = pd.read_csv(csv_path)
        col = f"dev_{domain}_hauc"
        tmp = tmp.sort_values(by=col, ascending=False).reset_index()
        top_list.append(tmp.loc[:n_top, ["post_process", "path", col]])
    top_df = (
        pd.concat(top_list, axis=0)
        .sort_values(by=col, ascending=False)
        .reset_index(drop=True)
    )
    for machine in machines:
        path_list = list(top_df["path"])
        pp_list = list(top_df["post_process"])
        for idx in range(n_mix):
            path = path_list[idx]
            pp = pp_list[idx]
            if ("all" not in pp) and ("so" not in pp) and ("ta" not in pp):
                continue
            checkpoint = path.split("/")[-2]
            agg_path = (
                path.replace("all", machine)
                .replace("score", checkpoint)
                .replace(".csv", "_agg.csv")
            )
            agg_df = pd.read_csv(agg_path)
            if idx == 0:
                seed_df = agg_df.loc[:, ["path", "section", "is_normal", "mode"]]
                y_true = 1 - seed_df["is_normal"].values
                section_list = seed_df["section"].values
                seed_df["domain"] = seed_df["path"].map(
                    lambda x: "target" if "target" in x else "source"
                )
                domain_list = seed_df["domain"].values
            seed_df[f"{path}:{pp}"] = agg_df[pp]
        ave_pred = -seed_df.iloc[:, 5:].values.mean(1)
        score, score_std = hauc(
            y_true, ave_pred, section_list, domain_list, mode=domain
        )
        hauc_list.append(score)
        std_list.append(score_std)
        print(
            f"{domain} {machine} hauc:{score:.4f}, std:{score_std:.4f} n_mix:{n_mix} n_top:{n_top}"
        )
        seed_df_dict[f"{domain}_{machine}"] = seed_df

hauc_score = hmean(hauc_list)
hauc_std = np.array(std_list).mean()
print(f"All hauc:{hauc_score:.4f}, std:{hauc_std:.4f} n_mix:{n_mix}")

save_df = seed_df_dict[f"{domain}_{machines[0]}"][
    ["section", "is_normal", "mode", "domain"]
]
save_df["is_anomaly"] = 1 - save_df["is_normal"]
for domain in domains:
    hauc_list = []
    std_list = []
    for machine in machines:
        save_df[f"{domain}_{machine}_path"] = seed_df_dict[f"{domain}_{machine}"][
            "path"
        ]
        for sec in range(6):
            use_cols = list(seed_df_dict[f"{domain}_{machine}"].columns)
            is_normal_pred = (
                seed_df_dict[f"{domain}_{machine}"]
                .loc[save_df["section"] == sec, use_cols[5 : n_mix + 5]]
                .values.mean(1)
            )
            save_df.loc[
                save_df["section"] == sec, f"{domain}_{machine}_normal_score"
            ] = is_normal_pred
            save_df.loc[
                save_df["section"] == sec, f"{domain}_{machine}_anomaly_score"
            ] = -zscore(is_normal_pred)
        score, score_std = hauc(
            save_df["is_anomaly"].values,
            save_df[f"{domain}_{machine}_anomaly_score"].values,
            save_df["section"].values,
            save_df["domain"].values,
            mode=domain,
        )
        print(f"{domain} {machine} hauc:{score:.4f} std:{score_std:.4f}")
        hauc_list.append(score)
        std_list.append(score_std)
    hauc_score = hmean(hauc_list)
    hauc_std = np.array(std_list).mean()
    print(f"{domain} hauc:{hauc_score:.4f}, std:{hauc_std:.4f}")

# load weight
hauc_list = []
std_list = []
weight_df = pd.read_csv("exp/all/weights/is_target_weight4_div0.15.csv")
for machine in machines:
    is_ta_weight = weight_df[machine]
    save_df[f"{machine}_anomaly_score"] = (1.0 - is_ta_weight) * save_df[
        f"source_{machine}_anomaly_score"
    ] + is_ta_weight * save_df[f"target_{machine}_anomaly_score"]
    score, score_std = hauc(
        save_df["is_anomaly"].values,
        save_df[f"{machine}_anomaly_score"].values,
        save_df["section"].values,
        save_df["domain"].values,
        mode="all",
    )
    print(f"{machine} hauc:{score:.4f} std:{score_std:.4f}")
    hauc_list.append(score)
    std_list.append(score_std)
hauc_score = hmean(hauc_list)
hauc_std = np.array(std_list).mean()
print(f"All hauc: {hauc_score:.4f}, std:{hauc_std:.4f}")

save_name = "pred_2"
save_df.to_csv(f"exp/all/pred/{save_name}.csv", index=False)
hauc_list = []
std_list = []
save_df = pd.read_csv(f"exp/all/pred/{save_name}.csv")
plt.figure(figsize=(16, 13))
for i, machine in enumerate(machines):
    score, score_std = hauc(
        save_df["is_anomaly"].values,
        save_df[f"{machine}_anomaly_score"].values,
        save_df["section"].values,
        save_df["domain"].values,
        mode="all",
    )
    print(f"{machine} hauc:{score:.4f} std:{score_std:.4f}")
    hauc_list.append(score)
    std_list.append(score_std)
    plt.subplot(4, 2, i + 1)
    save_df.loc[:600, f"{machine}_anomaly_score"].hist(bins=50, alpha=0.4, label="dev")
    save_df.loc[600:, f"{machine}_anomaly_score"].hist(bins=50, alpha=0.4, label="eval")
    plt.legend()
    plt.title(f"{machine} hauc:{score:.4f} std:{score_std:.4f}")
hauc_score = hmean(hauc_list)
hauc_std = np.array(std_list).mean()
print(f"All hauc:{hauc_score:.4f}, std:{hauc_std:.4f}")
plt.tight_layout()
plt.savefig(f"exp/all/pred/{save_name}_{hauc_score:.4f}_std{hauc_std:.4f}.jpg")


os.makedirs("exp/all/pred", exist_ok=True)
ver = 2
sub_df = pd.read_csv(f"exp/all/pred/pred_{ver}.csv")
all_hauc = []
for machine in machines:
    machine_f1 = []
    machine_hauc = []
    for section in range(3):
        use_dcm = "source_" if ver % 2 == 0 else ""
        anomaly_score_df = pd.DataFrame()
        path_list = sub_df.loc[
            sub_df["section"] == section, f"{use_dcm}{machine}_path"
        ].map(
            lambda x: "_".join(
                x.split("/")[-1].split("_")[:2]
                + [x.split("_")[-1].replace("h5", "wav")]
            )
        )
        is_anomaly = sub_df.loc[sub_df["section"] == section, "is_anomaly"]
        anomaly_score_df["path"] = path_list
        anomaly_score_df["anomaly_score"] = sub_df.loc[
            sub_df["section"] == section, f"{machine}_anomaly_score"
        ]

        auc = roc_auc_score(is_anomaly, anomaly_score_df["anomaly_score"])
        pauc = roc_auc_score(is_anomaly, anomaly_score_df["anomaly_score"], max_fpr=0.1)
        machine_hauc += [auc, pauc]
        print(f"{machine} section {section:02} auc:{auc:.4f} pauc:{pauc:.4f}")
    print("-" * 30)
    print(f"{machine}  f1:{hmean(machine_f1):.4f} hauc:{hmean(machine_hauc):.4f}")
    print("-" * 30)
    all_hauc += machine_hauc
hauc_score = hmean(all_hauc)
hauc_std = np.array(all_hauc).std()
print(f"ver{ver} All hauc:{hauc_score:.4f}, std:{hauc_std:.4f}")
