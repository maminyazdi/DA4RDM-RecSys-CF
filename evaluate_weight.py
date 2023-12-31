"""Function to evaluate weight and decay for each user resource pair"""

import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from get_roles import get_role
from evaluate_interest_rating import get_interest_rating


def calculate_weight_decay(data_all_resources, data_all_users):
    alluser_list = data_all_resources["UserId"].unique()
    exception_users_list = ["C56573BD-601F-4F36-B55A-F03A9F47FE4C", "59FEA692-6983-4755-861E-D9DAA588991E",
                            "54c852ad-3bab-4fef-a236-8390b7c21335", "4489bbd3-72c3-432b-a193-c134ba28f925",
                            "e3801c04-35d3-46b7-ba50-b12732fc25d0", "9588b80e-86b0-4387-ac1e-c3213390981e",
                            "30165a82-081c-4f87-b815-26295cc7737e", "8df9105f-2faa-4f17-be29-2111dddb6d0b",
                            "1efbe205-ef11-4301-99a8-945a00891df0","32008cbc-b532-468a-83dd-b4ea52d9e207",
                            "a5aac704-72f5-4e9b-9ccb-6259981b4dd1","a04ae09d-7d96-47e4-9d24-3cfed5225491",
                            "d390d8a7-2ee5-43e8-b66b-62e2f6c5c102"]
    exception_users_list_lower = []
    for except_users in exception_users_list:
        exception_users_list_lower.append(except_users.lower())

    userid_list = [x for x in alluser_list if x not in exception_users_list_lower]

#    dataset_resources = 'Data/tomography.csv'
#    resource_data = pd.read_csv(dataset_resources, sep='\|')
#    resource_list = resource_data.Resource.unique()
#    resource_list = resource_list.tolist()
#    for i in range(len(resource_list)):
#        resource_list[i] = resource_list[i].lower()

    most_recent_time = pd.to_datetime(data_all_users["Timestamp"].max())

    user_operation_dictionary = {}
    for userid in userid_list:
        data_user_selected = data_all_resources[data_all_users["UserId"] == userid]
        resource_id_list = data_user_selected["ResourceId"].unique()
        role_ids = data_user_selected["RoleId"].unique()
        user_operations = data_user_selected["Operation"].unique()
        user_role = get_role(role_ids, user_operations)

        for resource in resource_id_list:
            data_resource_selected = data_user_selected[data_user_selected["ResourceId"] == resource]
            project_id = data_resource_selected.loc[data_resource_selected["ResourceId"] == resource, 'ProjectId'].unique()[0]
            # time_rating = get_interest_rating(data_resource_selected, data_user_selected)
            data_file_selected = data_resource_selected[~data_resource_selected["FileId"].isnull()]
            if data_file_selected.empty:
                continue
            else:
                data_feature_counts = data_file_selected["Operation"].value_counts()
                data_features = data_file_selected["Operation"].unique()
                data_time_sorted = data_file_selected.sort_values(['Timestamp'], ascending=True)
                for feature in data_features:
                    data_feature_selected = data_time_sorted[data_time_sorted["Operation"] == feature]
                    val = 0
                    for i, row in data_feature_selected.iterrows():
                        timestamp = row["Timestamp"]
                        reference_time = pd.to_datetime(timestamp)
                        delta = relativedelta(most_recent_time, reference_time)
                        num_months = delta.months + (delta.years * 12)
                        if num_months <= 1:
                            val = val + 1
                        elif num_months > 1 and num_months <= 2:
                            val = val + 0.8
                        elif num_months > 2 and num_months <= 3:
                            val = val + 0.6
                        elif num_months > 3 and num_months <= 4:
                            val = val + 0.4
                        elif num_months > 4 and num_months <= 5:
                            val = val + 0.2
                        elif num_months > 5:
                            val = val + 0.1
                    data_feature_counts.update((pd.Series(val, index=[feature])))
                reference_time = data_time_sorted['Timestamp'].iloc[0]
                end_time = data_time_sorted['Timestamp'].iloc[-1]
                duration = pd.to_datetime(end_time) - pd.to_datetime(reference_time)
                duration = str(duration)
                feature_str = str(userid) + "#" + str(resource) + "#" + duration + "#" + str(project_id) + "#" + str(
                    user_role)
                user_operation_dictionary.update({feature_str: data_feature_counts})
    return user_operation_dictionary
