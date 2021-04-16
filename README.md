**Learning sleep quality from dialy logs project** 

It is implemented using python 3.6.6 

**Required Libraries:**

pandas 0.23.4
pyprind 2.11.2
numpy 1.16.0
scikit-learn 0.20.2
tensorflow 1.12.0

**Part 1: Missing Data Imputation**
How to Run:
Step 1: add the path of your dataset(full_data_sleeps) to dataset_path variable in main.py
Step 2: add the path where you want to save your imputed dataset to imputeddata_path variable in main.py
step 3: after step1 and step2 run main.py to get the imputed data

**Part 2: Predicting Sleep Efficiency**
How to Run:
Step 1: add path of GAIN, average and blank data set path in dataset variable of fetch_dataset() function and also in datacleaning_average() and datacleaning_BLANK()
Step 2: provide metadata(full_meta-data_sleeps) path to metadataset variable of get_metadata() function
Step 3: Open terminal and go to Sleep_Efficiency_Prediction folder using cd Sleep_Efficiency_Prediction
step 4: Run main.py by main.py --modelname samplemodelname --dataimpute imputemethod
where --modelname can be any model and --dataimpute can be gain, average and blank

If we run main function our model's train loss and test loss in each phase is generated in log folder and attention scores are stored in result folder