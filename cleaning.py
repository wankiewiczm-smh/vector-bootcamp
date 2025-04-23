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

    ## return the table
    return(df_clean)



