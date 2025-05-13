from config_reader import read_config
from data_ingestion import DataIngestion

data_ingestion_config = read_config("config/config.yaml")
data_ingestion = DataIngestion(data_ingestion_config)
data_ingestion.run()
