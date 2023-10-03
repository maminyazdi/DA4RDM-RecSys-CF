from create_matrix import create_operations_matrix
from evaluate_weight import calculate_weight_decay
from recommendation import resource_recommender
from extract_data import extract_data
import pandas as pd
import numpy as np


def get_recommendations(ref_user, num_neighbours=3, num_recommendation=3, outlier_detection_method="percentile"):
    # data_extracted = extract_data("SampleData/17-02-2023.csv")
    # data_extracted.to_csv("Data/Data_Extract_17_02_2023.csv")
    # data_extracted.to_csv("Data/Data_Extract.csv")
    data_extracted = pd.read_csv("SampleData/Data_Extract_17_02_2023.csv", error_bad_lines=False)
    # data_extracted = pd.read_csv("SampleData/Data_Extract - singleUser.csv", error_bad_lines=False)
    data_all_users = data_extracted[~data_extracted["UserId"].isnull()]
    data_all_resources = data_all_users[~data_all_users["ResourceId"].isnull()]
    data_file_selected = data_all_resources[~data_all_resources["FileId"].isnull()]
    unique_operations = data_file_selected["Operation"].unique()
    user_operation_dictionary = calculate_weight_decay(data_all_resources, data_all_users)

    matrix_columns = ["UserId", "ResourceId", "ProjectId", "Roles"]
    for operation in unique_operations:
        matrix_columns.append(operation)
    matrix_columns.append("Duration")
    matrix_columns.append("Rating")
    data_matrix = pd.DataFrame(columns=[matrix_columns])
    matrix_data = create_operations_matrix(user_operation_dictionary, data_matrix, outlier_detection_method)
    resource_recommender(matrix_data, ref_user, num_neighbours, num_recommendation)

    matrix_data.replace(0.0, np.nan, inplace=True)
    count_row = matrix_data.count(axis='columns')
    matrix_data = matrix_data.assign(Resource_Interaction_Count=count_row)
    matrix_data.replace(np.nan, 0, inplace=True)
    matrix_data = matrix_data.sort_values(by=['Resource_Interaction_Count'], ascending=False)
    matrix_data.to_csv("Data/Matrix_Interaction.csv")
