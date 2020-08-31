# Finding Sub-types of Diseases 

Some diseases may have sub-types that react differently to treatment. Therefore, finding a cure for such diseases requires identifying these subtypes. 
In this work we are taking a method that has been presented theoretically in previous work and implement it for the purpose of finding sub-types of diseases, using machine learning tools. The aim of the current study is to take advantage of the differences between separate populations, using an additional observed signal which is corelated with the unobserved target variable (sub-type), to better recover the underlying structure of a disease. 

## The Algorithm 

In General, the algorithm builds a clustering tree assuming that the clusters are disjoint. Each node in  the tree is a classifier trained to separate between two populations. At the beginning we take data of patients who having a known disease and create two samples from it, according to prior knowledge about the risk factors of the disease, reweight the patients such that the two samples will have the same cumulative weight, and train the classifier. The first classifier becomes the root of the tree and it splits to two sets, as the next step we take all of the patient in each set separately, reweigh them again and train another classifier to separate between the two samples. We keep going in the same fashion until all the patients associated with a leaf are from the same cluster in which case, no classifier can split the cluster any better than random.

## The Dataset - Coronavirus

With the outbreak of the Corona epidemic in Israel, the Ministry of Health began publishing a database that includes characteristics of people who undergo Corona tests. The dataset used for this project is available as CSV file called 'corona_dataset.xlsx'. In addition, pre-processing of the data before using the algorithm is necessary. More information and the various steps are available in the file 'Preprocessing_Corona.py'. 
