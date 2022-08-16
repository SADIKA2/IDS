# IDS
# Abstract:
The subject of traffic classification is of great importance for effective network
planning, policy-based traffic management, application prioritization, and security control.
Although it has received substantial attention in the research community there are still many
unresolved issues. Security has been a crucial factor in this modern digital period due to the rapid
development of information technology, which is followed by serious computer crimes that, in
turn, led to the emergence of Intrusion Detection Systems (IDSs). Different configurations of
machine learning algorithm, which adopt Decision Tree Classifier, Logistic Regression, K
Nearest Neighbour classification, Naive Bayes as key building blocks of the proposed intrusion
detection and AdaBoost Classifier, Random Forest Classifier, SVM(Support Vector Machine)
Classifier for Multi-class traffic classification, are investigated to evaluate the impact of the
observation window of traffic profiles on the classification accuracy, prediction loss, complexity,
and convergence. The comparison with respect to conventional single-task learning approaches,
that do not use autoencoders and tackle classification and detection tasks separately, clearly
demonstrates the effectiveness of the proposed traffic classification and intrusion detection
approach under different system configurations
# Introduction
There is a significant increase of cloud and networking-enabled applications, leading to an
exponential growth of network traffic. Monitoring all these applications and their generated
traffic is a challenging and complex task, especially in regard to anonymous networks used to
access the Dark Web or darknet. Manual investigation of the network traffic to determine
patterns of malicious activity is a very difficult task and the results might not be satisfactory. The
usage of machine learning is the most viable approach for the classification of the traffic as
normal or malign activity.
The need for enhanced Intrusion Detection Systems (IDS) is more obvious than ever due to an
exponential growth in the number of cyber-attacks. Machine Learning (ML) approaches are
playing a critical role in the early categorization of assaults in the event of intrusion detection
inside the system in this respect. However, due to the enormous variety of algorithms available,
choosing the best strategy might be difficult. This study examines some of the most recent
state-of-the-art intrusion detection systems and evaluates their benefits and drawbacks in order to
address this problem.
# Objectives
● To study and train machine learning models to classify the network traffic.
● To create a network intrusion detector, a prediction model that can discriminate between
faulty connections, often known as intrusions or assaults, and good regular connections.
# Machine Learning Algorithms
We employed a total of six different types of classifiers in our study. A summary of all
classification algorithms is provided below.
## Random Forest Classifier
Random forest algorithm is one of the most powerful algorithms in machine learning technology
and it is based on the concept of decision tree algorithm. Random forest algorithm creates the
forest with several decision trees. A high number of trees gives high detection accuracy. The
creation of trees is based on the bootstrap method. In the bootstrap method features and samples
of a dataset are randomly selected with replacement to construct a single tree. Among randomly
selected features, a random forest algorithm will choose the best splitter for the classification,
and as the decision tree algorithm; the Random Forest algorithm also uses Gini index and
information gain methods to find the best splitter. This process will continue until a random
forest creates N number of trees. Each tree in the forest predicts the target value and then the
algorithm will calculate the votes for each 2 predicted targets. Finally, the random forest
algorithm considers the high-voted predicted target as a final prediction.
## Decision Tree Classifier
A Decision Tree (DT) is a classifier that exemplifies the use of tree-like structure. It gains
knowledge on classification. Each target class is denoted as a leaf node of DT and non-leaf nodes
of DT are used as a decision node that indicates a certain test. The outcomes of those tests are
identified by either of the branches of that decision node. Starting from the beginning at the root
this tree goes through it until a leaf node is reached. It is the way of obtaining classification
results from a decision tree. Decision tree learning is an approach that has been applied to spam
filtering. This can be useful for forecasting the goal based on some criterion by implementing
and training this model.
## Support Vector Machine Classifier
Support vector machine is a powerful algorithm in machine learning technology. In the support
vector machine algorithm, each data item is plotted as a point in n-dimensional space, and the
support vector machine algorithm constructs separating lines for the classification of two classes;
this separating line is well known as a hyperplane. Support vector machine seeks for the closest
points called support vectors and once it finds the closest point it draws a line connecting to
them. Support vector machine then constructs a separating line which bisects and perpendicular
to the connecting line. To classify data perfectly the margin should be maximum. Here the
margin is a distance between hyperplane and support vectors. In a real scenario, it is not possible
to separate complex and nonlinear data. To solve this problem, the support vector machine uses a
kernel trick that transforms lower-dimensional space to higher-dimensional space.
## Logistic Regression Algorithm
Logistic regression (LR) is a statistical method similar to linear regression since LR finds an
equation that predicts an outcome for a binary variable, I from one or more response variables,
X. However, unlike linear regression, the response variables can be categorical or continuous, as
the model does not strictly require continuous data. To predict group membership, LR uses the
log odds ratio rather than probabilities and an iterative maximum likelihood method rather than
the least squares to fit the final model. Logistic regression assumes independence among
variables, which is not always met in morphotropic datasets. However, as is often the case, the
applicability of the method (and how well it works, e.g., the classification error) often trumps
statistical assumptions.
## K-Nearest Neighbour Classifier
K-Nearest Neighbour Classifier often known as lazy learners, identifies objects based on
closest proximity of training examples in the feature space.The classifier considers k number
of objects as the nearest object while determining the class. The main challenge of this
classification technique relies on choosing the appropriate value of k.
## Naive Bayes Classifier
The Naive Bayes classifier is a supervised classification tool that exploits the concept of Bayes
Theorem of Conditional Probability. The decision made by this classifier is quite effective in
practice even if its probability estimates are inaccurate. This classifier obtains a very promising result
in the following scenario- when the features are independent or features are completely functionally
dependent. The accuracy of this classifier is not related to feature dependencies; rather it is the
amount of information loss of the class due to the independence assumption is needed to predict the
accuracy.
# Methodology
### Data collection:
The machine learning (ML) technique is based on the labeled dataset. The
dataset used in this project is the NSL-KDD dataset from the University of New Brunswick,
Canada. The dataset is an improved version of the KDDCUP’99 datasets from DARPA Intrusion
Detection Evaluation Program. The NSL-KDD dataset used in this project consists of 125,973
records training set and 22,544 records test set with 42 attributes/features for each connection
sample including class label containing the attack types. After loading the dataset into the Python
(Jupyter Notebook) development environment, the first task performed was mapping services
into 19 traffic classes and various attack types into four attack classes in the datasets.
### Data preprocessing: 
In this proposed technique, a machine learning classifier is trained as input,
and then using the trained sample prediction, unknown classes are classified. We first establish a
reference standard performance for three classifiers (Random Forest, Decision Tree, and SVM)
using publicly available network traffic NSL KDD datasets. The features are calculated using the
wrapper method and then the ML classifier is trained with these features with known traffic
classes and creates the classifier model known as the Memorization process. This model is then
used to classify unknown traffic known as testing or Generalization process. Three ML
algorithms are used for IP traffic classification with mentioned datasets and also four ML
algorithms are used for intrusion detection for the same datasets. Finally, performance analysis is
done with the help of several evaluation metrics.
### Feature selection: 
When building a machine learning model in real life, it’s almost rare that all
the variables in the dataset are useful to build a model. Adding redundant variables reduces the
generalization capability of the model and may also reduce the overall accuracy of a classifier.
Furthermore, adding more and more variables to a model increases the overall complexity of the
model. The goal of feature selection in machine learning is to find the best set of features that
allows one to build useful models of studied phenomena.
Recursive feature elimination (RFE) is a feature selection method that fits a model and removes
the weakest feature (or features) until the specified number of features is reached. By applying
RFE we got 10 selective features: ['src_bytes', 'dst_bytes','logged_in', 'root_shell','serror_rate',
'srv_serror_rate', 'dst_host_srv_count', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'service']

