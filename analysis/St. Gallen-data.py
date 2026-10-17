# # ==========================================================
# # BCMPM Guideline-informed Risk Stratification
# # Based on St. Gallen consensus principles
# #

import pandas as pd
from scipy.stats import chi2_contingency


# =====================================================
# 1. Load data
# =====================================================

input_file = (
    "./xxx/xx/data/val330.csv"
)

output_file = (
    "./xxx/xx/"
    "val330_Guideline_informed_risk_v3_treatment.csv"
)


df = pd.read_csv(input_file)


print("\nDataset size:")
print(df.shape)



# =====================================================
# 2. Guideline-informed risk classification
# =====================================================


def guideline_risk_v3(row):


    subtype = row["molecular subtypes"]

    tumor = row["tumor size"]

    node = row["lymph node metastasis"]

    stage = row["clinical stage"]

    grade = row["histological grade"]

    HER2 = row["HER2"]


    reasons=[]


    # ===========================
    # High risk
    # ===========================


    # Stage III
    if stage == 3:

        reasons.append(
            "Clinical stage III"
        )


    # N>=4
    if node == 2:

        reasons.append(
            "Extensive nodal disease"
        )


    # Tumor >5cm
    if tumor == 3:

        reasons.append(
            "Large tumor size"
        )



    # TNBC
    if subtype == 4:

        if tumor >=2 or node>=1:

            reasons.append(
                "TNBC with adverse features"
            )



    # HER2+
    if HER2 == 1:

        if tumor>=2 or node>=1:

            reasons.append(
                "HER2-positive with adverse features"
            )



    # Luminal B
    if subtype == 2:

        if (
            node>=1
            and
            (
                tumor>=2
                or
                grade==3
            )
        ):

            reasons.append(
                "Luminal B with high clinical burden"
            )



    # Grade 3 aggressive subtype
    if grade==3:

        if subtype in [2,3,4]:

            reasons.append(
                "High grade aggressive subtype"
            )



    # ===========================
    # Final
    # ===========================

    if len(reasons)>0:

        return pd.Series(
            [
                "High",
                "; ".join(sorted(set(reasons)))
            ]
        )

    else:

        return pd.Series(
            [
                "LowModerate",
                "Favorable clinicopathological profile"
            ]
        )



# =====================================================
# 3. Apply risk classification
# =====================================================


df[
    [
        "Guideline_Risk",
        "Risk_reason"
    ]

]=df.apply(
    guideline_risk_v3,
    axis=1
)



df["Guideline_Risk_binary"] = (
    df["Guideline_Risk"]
    .map(
        {
            "LowModerate":0,
            "High":1
        }
    )
)



# =====================================================
# 4. Risk distribution
# =====================================================


print("\n==============================")
print("Guideline Risk Distribution")
print("==============================")


risk_table = (
    df["Guideline_Risk"]
    .value_counts()
)


print(risk_table)


print(
    (
        risk_table
        /
        len(df)
        *
        100
    )
    .round(2)
)



# =====================================================
# 5. Treatment validation
#
# Hypothesis:
#
# High:
#   CT ↑
#   TT ↑
#
# LowModerate:
#   ET ↑
#
# =====================================================


print("\n================================")
print("Treatment distribution by risk")
print("================================")



treatment_vars = [
    "CT",
    "ET",
    "RT",
    "TT"
]



treatment_summary=[]



for var in treatment_vars:


    table = pd.crosstab(
        df["Guideline_Risk"],
        df[var]
    )


    print("\n----------------------")
    print(var)

    print(table)


    # percentage
    pct = (
        table
        .div(
            table.sum(axis=1),
            axis=0
        )
        *
        100
    )


    print("\nPercentage (%)")

    print(
        pct.round(2)
    )


    # Chi-square

    chi2,p,_,_=chi2_contingency(
        table
    )


    treatment_summary.append(
        {
            "Treatment":var,
            "High_rate":
                pct.loc["High",1]
                if 1 in pct.columns
                else None,

            "LowModerate_rate":
                pct.loc["LowModerate",1]
                if 1 in pct.columns
                else None,

            "P_value":p
        }
    )



# =====================================================
# 6. Export treatment validation table
# =====================================================


treatment_df = pd.DataFrame(
    treatment_summary
)


print("\n==============================")
print("Treatment validation summary")
print("==============================")


print(
    treatment_df
)



# =====================================================
# 7. Save
# =====================================================


df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


treatment_df.to_csv(
    output_file.replace(
        ".csv",
        "_treatment_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)



print("\nFinished!")

print(
    "Risk file:"
)

print(
    output_file
)

print(
    "\nTreatment summary:"
)

print(
    output_file.replace(
        ".csv",
        "_treatment_summary.csv"
    )
)