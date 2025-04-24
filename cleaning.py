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

    ## drop expired encounters: discharge_disposition in {11, 19, 20, 21}
    df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

    # imputing missing values
    df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x: 'Unknown' if type(x) != str else x)
    df['A1Cresult'] = df['A1Cresult'].apply(lambda x: 'Unknown' if type(x) != str else x)


    ## return the table
    return(df)
