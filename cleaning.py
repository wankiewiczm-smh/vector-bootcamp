import pandas as pd

### load in table and clean
def load_and_clean(df):
    """
    Cleans the diabetes130 dataset for machine learning models.

    The dataset is used is located in: data/diabetic_data.csv
    """

    ## create total_previous_visits
    df_clean['total_previous_visits'] = df_clean['number_outpatient'] + df_clean['number_inpatient'] + df_clean['number_emergency']

    ## create outcome column, convert to binary
    df_clean = (df_clean['readmitted'] == '<30').astype(int)

    ## remove unnecessary columns
    df_clean = df.drop(['weight', 'encounter_id', 'patient_nbr', 'medical_specialty', 'payer_code', 'readmitted'], axis = 1)

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
    df_clean['age'] = df_clean['age'].apply(lambda x: age_transform[x])

    ## Drop missing gender
    df_clean = df_clean.dropna(subset=['gender'])

    ## binarize gender as 1 = Female 0 = Male
    df_clean['female'] = df_clean['gender'].apply(
        lambda x: 1 if x == 'Female' else 0)

    ## Assign ? Race with "Other"
    df_clean['race'] = df_clean['race'].apply(lambda x: "Other" if x == "?" else x)

    ## return the table
    return(df_clean)

