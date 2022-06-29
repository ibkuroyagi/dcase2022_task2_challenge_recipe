import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
seed_df_dict = {}  # {"fan":seed_df for fan}
rows_list = []
n_top = 5
top_list = []
machines = ["bearing", "fan", "gearbox", "valve", "slider", "ToyCar", "ToyTrain"]
machine = machines[6]
domains = ["source", "target"]
domain = domains[1]
n_mix = 20

for domain in domains:
    for machine in machines:
        for csv_path in score_csv_list:
            tmp = pd.read_csv(csv_path)
            col = f"dev_{domain}_{machine}_hauc"
            tmp = tmp.sort_values(by=col, ascending=False).reset_index()
            top_list.append(tmp.loc[:n_top, ["post_process", "path", col]])
        top_df = (
            pd.concat(top_list, axis=0)
            .sort_values(by=col, ascending=False)
            .reset_index(drop=True)
        )
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
        score, score_std = hauc(y_true, ave_pred, section_list, domain_list, mode="all")
        seed_df_dict[f"{domain}_{machine}"] = seed_df
        print(
            f"{machine} hauc:{score:.4f} std:{score_std:.4f} n_top:{n_top} n_mix:{n_mix}"
        )


save_df = seed_df_dict[f"source_{machines[0]}"][
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
            domain_machine_normal_score = (
                seed_df_dict[f"{domain}_{machine}"]
                .loc[seed_df["section"] == sec, use_cols[5:]]
                .values.mean(1)
            )
            save_df.loc[
                seed_df["section"] == sec, f"{domain}_{machine}_normal_score"
            ] = domain_machine_normal_score
            save_df.loc[
                seed_df["section"] == sec, f"{domain}_{machine}_anomaly_score"
            ] = -zscore(domain_machine_normal_score)
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
    print(f"{domain} hauc: {hauc_score:.4f}, std:{hauc_std:.4f}")
# load weight
hauc_list = []
std_list = []
os.makedirs("exp/all/pred", exist_ok=True)
weight_df = pd.read_csv("exp/all/weights/is_target_weight4_div0.1.csv")
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
save_name = "pred_4"
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
