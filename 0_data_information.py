# Script 2.1: Checking number of patients by cohort, timepoint and generating new dataset.
import pandas as pd
import os


# 1 Define Functions
# 1.1 Function to get the number of times each patient is repeated by cohort
def count_patients(data):
    count = data.groupby(['ID', 'Cohort']).size().reset_index(name='Count')
    return count

# 1.2 Fuction to get both the number of patients by cohort and the number of repetitions across all the dataset
def cohort_patient_count(count):
    patient_count = count.groupby('Cohort')['ID'].nunique().reset_index(name= 'Patients')
    patient_repetition = count['Count'].value_counts().reset_index()
    patient_repetition.columns = ['Count', 'Number_of_Patients']

    return patient_count, patient_repetition

# 1.3 Function to count the number of repetitions of patients by cohort
def cohort_count(count):
    # Contar cuántos pacientes tienen 1, 2, 3... repeticiones en cada cohorte
    cohort_count = count.groupby(['Cohort', 'Count'])['ID'].nunique().reset_index()
    cohort_count.columns = ['Cohort', 'Count', 'Number_of_Patients']

    return cohort_count

# 1.4 Function to generate the txt
def txt_gen(count, patient_count, patient_repetition, cohort_count, filename = "patient_info.txt", directory = "../data/data_info"):
    filepath = os.path.join(directory, filename)
    
    with open(filepath, "w") as f:
        f.write("Patient count by cohort:\n")
        f.write(count.to_string(index=False))
        f.write("\n\nPatients by cohort:\n")
        f.write(patient_count.to_string(index=False))
        f.write("\n\nDistribution of repetitions:\n")
        f.write(patient_repetition.to_string(index=False))
        f.write("\n\nDistribution of repetitions by cohort:\n")
        f.write(cohort_count.to_string(index=False))

    print(f"File saved at: {filepath}")



