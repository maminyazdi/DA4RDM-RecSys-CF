import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# Matrix for the task of plotting values between 0 & 1
def plot_normalized_matrix(matrix_normalized):
    user_list = matrix_normalized["UserId"].squeeze().unique()
    res_list = matrix_normalized["ResourceId"].squeeze().unique()
    column_list = []
    for val in user_list:
        column_list.append(val)
    rlist = []
    for valu in res_list:
        rlist.append(valu)
    index = rlist
    recommendation_matrix = pd.DataFrame(index=index, columns=column_list)
    for index, row in matrix_normalized.iterrows():
        user = row['UserId']
        res = row['ResourceId']
        NormalizeRating = row['NormalizedRating']
        recommendation_matrix.at[res, user] = NormalizeRating
    recommendation_matrix.replace(np.nan, 0, inplace=True)
    sns.heatmap(recommendation_matrix, )
    plt.show()


def create_plot_resource_to_resource_matrix(matrix_df):


    # create plot for selected resources
    resource_data = pd.read_csv("SampleData/tomography.csv", sep='\|')
    resource_list = resource_data.Resource.unique()
    resource_list = resource_list.tolist()
    for i in range(len(resource_list)):
        resource_list[i] = resource_list[i].lower()


    new_df = matrix_df.filter(items=resource_list, axis=0)

    pairwise = pd.DataFrame(
        squareform(pdist(new_df, metric='cosine')),
        columns=new_df.index,
        index=new_df.index
    )
    # plt.figure(figsize=(10, 10))
    sns.heatmap(
        pairwise,
        cmap="autumn",
        annot=True,
        annot_kws={"size": 7},
        linewidths=2
    )


    long_form = pairwise.unstack()
    long_form.index.rename(['Resource-A', 'Resource-B'], inplace=True)
    long_form = long_form.to_frame('cosine distance').reset_index()

    plt.show()


