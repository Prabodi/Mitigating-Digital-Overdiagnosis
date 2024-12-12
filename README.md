Mitigating Digital Overdiagnosis

This repository focuses on mitigating overdiagnosis in healthcare by retraining a classifier with an updated label that combines disease diagnostic criteria and disease trajectories. The goal is to reclassify overdiagnosed patients as ‘HIT-negative’ based on clinical evolution and diagnostic guidelines, thereby reducing overdiagnosis and improving clinical decision-making.

Project Overview

Overdiagnosis in healthcare can result in unnecessary treatments, patient distress, and increased healthcare costs. This project aims to identify overdiagnosed patients and retrain the HIT (Heparin-Induced Thrombocytopenia) classifier using an updated label that incorporates both disease trajectories and diagnostic criteria. By reclassifying overdiagnosed patients as ‘HIT-negative’, the project seeks to improve classification accuracy and clinical outcomes.

Key Concepts

Original Label: Based solely on HIT diagnostic criteria, patients are classified as either 'HIT-positive' or 'HIT-negative'.

Updated Label: Incorporates both HIT diagnostic criteria and disease trajectories, modifying the label for overdiagnosed patients whose clinical evolution resembles true negative (TN) patients.


Hypothesis

By retraining the classifier with the updated labels, previously overdiagnosed patients will be reclassified as 'HIT-negative', reducing the rate of overdiagnosis and improving the classifier’s performance.

Files and Scripts

1. cohort_extraction_for_trajectories.sql

This SQL script extracts phenotype data from the MIMIC-IV database to identify true positive (TP) and true negative (TN) patients. The extracted data is used to create the basis for trajectory analysis and classifier retraining.

2. 1_Trajectory_clustering_with_abstraction.py

This Python script performs trajectory clustering using patient-level data, with an abstraction method applied to simplify and group related clinical events. The abstraction reduces the complexity of the data and enhances the quality of the clustering for further analysis.

3. 2_retrain_classifier_with_updated_labels.py

This Python script retrains the HIT classifier with the updated labels. The classifier is retrained using the new labels, which are based on both diagnostic criteria and disease trajectories. Overdiagnosed patients are reclassified as ‘HIT-negative’, and the classifier is expected to improve accuracy in distinguishing true positive and true negative cases.

Usage

To use this repository, follow the steps below:

1. Clone the repository:

git clone https://github.com/Prabodi/Mitigating-Digital-Overdiagnosis.git


2. Install dependencies: Install the required Python libraries and SQL tools. For example:

pip install pandas numpy scikit-learn


3. Run the SQL script: Execute the cohort_extraction_for_trajectories.sql script to extract phenotype data from the MIMIC-IV database. This data will be used for trajectory analysis and classifier retraining.


4. Trajectory Clustering: Run 1_Trajectory_clustering_with_abstraction.py to perform trajectory clustering on the extracted patient data. This script will group patients based on their clinical trajectories, applying the abstraction method to simplify the event log.


5. Retrain Classifier: Execute 2_retrain_classifier_with_updated_labels.py to retrain the HIT classifier with the updated labels. This step incorporates the new label logic to reduce overdiagnosis by reclassifying patients who meet certain criteria as ‘HIT-negative’.



Results

After running the scripts, the retrained classifier will have an updated model that reduces overdiagnosis by reclassifying patients with clinical trajectories similar to TN patients as ‘HIT-negative’. The results will be evaluated based on classifier performance, including metrics like accuracy, precision, recall, and F1 score.

Acknowledgements

This project builds on the methodology developed in the Clinical Decision Support System repository, which was used for phenotype extraction and event log creation. Special thanks to the developers and contributors of the MIMIC-IV database for providing the clinical data used in this project.
