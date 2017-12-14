import ImplementQuotients

FE = ImplementQuotients.FeatureEngineering()
FE.create_features()

training_data = FE.GetTrainingData(2003,2014)
testing_data = FE.GetTestData(2015)