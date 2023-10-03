## Research Data Reusability with User-Based Recommender System

## Overview
`DA4RDM_RecSys_UserBased` is designed to enhance the discoverability and reusability of research data collections. It employs an Item-Item Collaborative Filtering (IICF) approach, incorporating implicit factors such as Time Decay, Activity Weights, and Frequency of Occurring Activities to provide robust and personalized recommendations.

## Methodology Abstract
The IICF methodology in `DA4RDM_RecSys_UserBased` consists of four main components:

1. **Implicit Rating Calculation**: The system deduces users’ implicit interest in resources by calculating an implicit rating. This rating considers the frequency of user activities, the weight assigned to each activity (reflecting its importance), and a time decay factor that accounts for the diminishing value of activities over time.

2. **Outlier Detection and Elimination**: Outliers in the implicit ratings are identified and removed using the z-score standardization technique, improving the recommendation system’s accuracy and performance.

3. **Pre-Processing**: Ratings are normalized to a consistent range between 0 and 1, enhancing the recommender system’s performance. The normalized ratings are then categorized into five classes, and resampling techniques are applied to balance the dataset, ensuring robust and unbiased recommendations.

4. **Resource Recommendation**: A user-resource matrix is created to represent users' ratings for resources. Pairwise distances between resources are computed using Cosine and Pearson distance metrics, facilitating the identification of similar resources. Recommendations are then generated for users based on these similarities and user preferences.

```shell
pip install DA4RDM-RecSys-UserBased
```
------------------



