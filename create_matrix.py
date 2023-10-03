"""Function to create operation matrix for each unique user resource pair and then create the recommendation matrix
to get resource recommendation for the reference user"""
import numpy as np
import pandas as pd
from remove_outliers import remove_outliers
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px
import plotting


def standardize(df):
    from sklearn.preprocessing import StandardScaler
    df[['StandardRating']] = StandardScaler().fit_transform(df[["Rating"]])

    return df


def normalizeDf(df):
    from sklearn.preprocessing import MinMaxScaler
    df[['NormalizedRating']] = MinMaxScaler().fit_transform(df[["Rating"]])
    df['NormalizedRating'] = round(df.iloc[:, 6], 2)
    return df


def create_recommendation_matrix(matrix_dat):
    user_list = matrix_dat["UserId"].squeeze().unique()
    res_list = matrix_dat["ResourceId"].squeeze().unique()
    column_list = []
    for val in user_list:
        column_list.append(val)
    rlist = []
    for valu in res_list:
        rlist.append(valu)
    index = rlist
    recommendation_matrix = pd.DataFrame(index=index, columns=column_list)
    for index, row in matrix_dat.iterrows():
        user = row['UserId']
        res = row['ResourceId']
        NormalizedRating = row['NormalizedRating']
        recommendation_matrix.at[res, user] = NormalizedRating
    recommendation_matrix.replace(np.nan, 0, inplace=True)

    return recommendation_matrix


def create_operations_matrix(feature_operation_dict, matrix, outlier_method):
    for feature_name, operation_name in feature_operation_dict.items():
        labels = feature_name.split("#")
        matrix.loc[-1, ["UserId"]] = labels[0]
        matrix.loc[-1, ["ResourceId"]] = labels[1]
        matrix.loc[-1, ["Duration"]] = labels[2]
        matrix.loc[-1, ["ProjectId"]] = labels[3]
        matrix.loc[-1, ["Roles"]] = labels[4]
        matrix = matrix.sort_index()
        for ind, value in operation_name.items():
            if ind == "Upload File":
                matrix.loc[-1, [ind]] = value * 0.7
            if ind == "Upload MD":
                matrix.loc[-1, [ind]] = value * 0.7
            if ind == "Update File":
                matrix.loc[-1, [ind]] = value * 0.9
            if ind == "Update MD":
                matrix.loc[-1, [ind]] = value * 0.9
            if ind == "View MD":
                matrix.loc[-1, [ind]] = value * 0.5
            if ind == "Download File":
                matrix.loc[-1, [ind]] = value * 1
            if ind == "Delete File":
                matrix.loc[-1, [ind]] = value * 0.3
        matrix.index = matrix.index + 1
    matrix.replace(np.nan, 0, inplace=True)
    matrix.to_csv("Data/Matrix_Operations_Ratings.csv")
    matrix['Rating'] = round((matrix.iloc[:, 4:10].sum(axis=1)) / 7, 2)

    matrix = standardize(matrix)
    matrix = matrix[["UserId", "ResourceId", "ProjectId", "Rating", "StandardRating", "Roles"]]
    matrix.to_csv("Data/Matrix_Ratings.csv")
    matrix = remove_outliers(matrix, outlier_method)

    matrix_normalized = normalizeDf(matrix)
    # Added for evaluating categorized ratings
    conditions = [
        (matrix_normalized['NormalizedRating'] < 0.2),
        (matrix_normalized['NormalizedRating'] >= 0.2) & (matrix_normalized['NormalizedRating'] < 0.4),
        (matrix_normalized['NormalizedRating'] >= 0.4) & (matrix_normalized['NormalizedRating'] < 0.6),
        (matrix_normalized['NormalizedRating'] >= 0.6) & (matrix_normalized['NormalizedRating'] < 0.8),
        (matrix_normalized['NormalizedRating'] >= 0.8)
    ]
    values = [1, 2, 3, 4, 5]
    matrix_normalized['CategorizedRating'] = np.select(conditions, values)
    matrix_normalized['1'] = np.where(matrix_normalized['CategorizedRating'] == 1, 1, 0)
    matrix_normalized['2'] = np.where(matrix_normalized['CategorizedRating'] == 2, 1, 0)
    matrix_normalized['3'] = np.where(matrix_normalized['CategorizedRating'] == 3, 1, 0)
    matrix_normalized['4'] = np.where(matrix_normalized['CategorizedRating'] == 4, 1, 0)
    matrix_normalized['5'] = np.where(matrix_normalized['CategorizedRating'] == 5, 1, 0)
    matrix_normalized = matrix_normalized.drop(['ProjectId', 'Roles', 'StandardRating'], axis=1)
    matrix_normalized.to_csv("Data/Data_Categorized_Ratings.csv")
    matrix_normalized = matrix_normalized[["UserId", "ResourceId", "Rating", "NormalizedRating"]]
    plotting.plot_normalized_matrix(matrix_normalized)

    matrix_filtered = matrix[["UserId", "ResourceId", "Rating", "NormalizedRating"]]
    matrix_filtered.to_csv("Data/Matrix_Ratings_Columns_Filtered.csv")
    recommendation_matrix = create_recommendation_matrix(matrix_filtered)

    plotting.create_plot_resource_to_resource_matrix(recommendation_matrix)

    recommendation_matrix.to_csv("Data/Matrix_Interaction_Recommendation.csv")
    return recommendation_matrix
