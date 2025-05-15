from config_reader import read_config
from data_ingestion import DataIngestion
from data_processing import DataProcessing
from model_training import ModelTraining

config = read_config("config/config.yaml")

data_ingestion = DataIngestion(config)
data_ingestion.run()

data_processing = DataProcessing(config)
data_processing.run()

model_training = ModelTraining(config)
model_training.run(config["model_training"]["model_type"])
