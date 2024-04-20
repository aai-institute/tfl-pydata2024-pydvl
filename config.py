import os

file_directory = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(file_directory, "data")
data_cell_path = os.path.join(data_path, "cell_images")
result_path = os.path.join(file_directory, "results")
logging_path = os.path.join(file_directory, "logs")
plot_base_path = os.path.join(result_path, "plots")
