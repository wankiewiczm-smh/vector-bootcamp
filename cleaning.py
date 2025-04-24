import pandas as pd
from collections import defaultdict
### load in table and clean
def load_and_clean(df):
    """
    Cleans the diabetes130 dataset for machine learning models.

    The dataset is used is located in: data/diabetic_data.csv
    """

    ## create total_previous_visits
    df['total_previous_visits'] = df['number_outpatient'] + df['number_inpatient'] + df['number_emergency']

    ## create outcome column, convert to binary
    df['readmit30'] = (df['readmitted'] == '<30').astype(int)

    ## remove unnecessary columns
    df = df.drop(['weight', 'encounter_id', 'patient_nbr', 'medical_specialty', 'payer_code', 'readmitted'], axis = 1)

    ## transform ages
    age_transform = {'[0-10)': 5,
                     '[10-20)': 15,
                     '[20-30)': 25,
                     '[30-40)': 35,
                     '[40-50)': 45,
                     '[50-60)': 55,
                     '[60-70)': 65,
                     '[70-80)': 75,
                     '[80-90)': 85,
                     '[90-100)': 95
                     }
    df['age'] = df['age'].apply(lambda x: age_transform[x])

    ## Drop missing gender
    df = df.dropna(subset=['gender'])

    ## binarize gender as 1 = Female 0 = Male
    df['female'] = df['gender'].apply(
        lambda x: 1 if x == 'Female' else 0)

    df = df.drop(['gender'], axis = 1)

    ## Assign ? Race with "Other"
    df['race'] = df['race'].apply(lambda x: "Other" if x == "?" else x)

    ccs_mapping = pd.read_csv("./data/ccs_multi_dx_tool_2015.csv", dtype = str)
    # Clean CCS mapping column names and values
    ccs_mapping.columns = ccs_mapping.columns.str.strip("'").str.strip()
    ccs_mapping = ccs_mapping.rename(columns={
        "ICD-9-CM CODE": "icd9_code",
        "CCS LVL 1": "ccs_level_1",
        "CCS LVL 1 LABEL": "ccs_description"
    })
    ccs_mapping["icd9_code"] = ccs_mapping["icd9_code"].str.strip("'").str.strip()
    # SRemove dots from ICD-9 codes
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_clean"] = df[col].str.replace(".", "", regex=False)

    # Build prefix-based lookup table for fallback matching
    prefix_map = defaultdict(lambda: None)
    for code, desc in zip(ccs_mapping["icd9_code"], ccs_mapping["ccs_description"]):
        for i in range(3, len(code) + 1):  # Use prefixes starting at 3 digits
            prefix = code[:i]
            if prefix not in prefix_map:
                prefix_map[prefix] = desc

    # Matching function using longest prefix available
    def map_icd_to_ccs(code):
        if pd.isna(code):
            return None
        for i in range(len(code), 2, -1):
            prefix = code[:i]
            if prefix in prefix_map:
                return prefix_map[prefix]
        return None

    # Map each diagnosis column to CCS group using prefix match
    for col in ["diag_1_clean", "diag_2_clean", "diag_3_clean"]:
        df[col.replace("_clean", "_ccs")] = df[col].map(map_icd_to_ccs)
    ## return the table
    return(df)
