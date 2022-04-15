#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz


st.write("""
# RNA-seq gene expression data for cancer classification
""")

#Reading the data
rna_exp = pd.read_csv('data.csv', index_col=[0])
#Reading the labels
labels = pd.read_csv('labels.csv', index_col=[0])

st.write("""
# Input data samples (EDA)
""")

st.write(labels.head())
st.write(rna_exp.head())

# labels.head()
# rna_exp.head()
rna_merged = pd.concat([rna_exp, labels], axis=1)
# rna_merged.shape
exp = rna_merged.set_index('Class').sort_index()
# exp.head()
# exp.shape
# sns.heatmap(exp)
# sns.clustermap(exp)
# exp.index.value_counts()
st.write(exp.index.value_counts())
# exp.isnull().values.any()

st.graphviz_chart('''
    digraph {
        Output models -> Unsupervised
        Output models -> Supervised
        Unsupervised -> Principle Component Analysis (PCA)
        Unsupervised -> t-distributed Stochastic Neighbor Embedding (t-SNE)
        Supervised -> K-Nearest Neighbors (KNN)
        Supervised -> Support Vector Machines (SVM)
        Supervised -> Decision Tree (DT)
    }
''')


# unsupervsed learning

# PCA
features = list(exp.columns.values)
len(features)
#x = exp.loc[:,features].values
x = rna_exp[features].reset_index(drop=True)
y = labels
# x.head()
y = y.reset_index(drop=True)
# y.head()

# rescale dataframe
x_std_scale = StandardScaler().fit_transform(x)
# x_std_scale
n_classes = len(y.groupby('Class'))
# n_classes


# PCA

pca = PCA(n_components=2)
principal_component = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_component, columns=['Principal Component 1', 'Principal Component 2'])
# principal_df.head()
concat_df = pd.concat([principal_df, y['Class']], axis=1)
# concat_df.head()


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_title('PCA', fontsize=15)
labels_list = ['BRCA', 'KIRC', 'LUAD', 'PRAD', 'COAD']
colors = ['r', 'b', 'g', 'y', 'black']

for lab, color in zip(labels_list, colors):
    indices = concat_df['Class'] == lab
    ax.scatter(concat_df.loc[indices, 'Principal Component 1'],
              concat_df.loc[indices, 'Principal Component 2'], 
              c = color,
              s = 50)
    ax.legend(labels_list)
    ax.grid()

     
# tsne
tsne = TSNE(n_components=2, random_state=0)
tsne_rnaexp = tsne.fit_transform(x)
tsne_df = pd.DataFrame(data=tsne_rnaexp, columns=['tsne 1', 'tsne 2'])
# tsne_df.head()
tsne_concat_df = pd.concat([tsne_df, y['Class']], axis=1)
# tsne_concat_df.head()


fig_2 = plt.figure(figsize=(8,8))
ax_2 = fig_2.add_subplot(1,1,1)
ax_2.set_title('tSNE', fontsize=15)
labels_list_2 = ['BRCA', 'KIRC', 'LUAD', 'PRAD', 'COAD']
colors_2 = ['r', 'b', 'g', 'y', 'black']

for lab, color in zip(labels_list_2, colors_2):
    indices_2 = tsne_concat_df['Class'] == lab
    ax_2.scatter(tsne_concat_df.loc[indices_2, 'tsne 1'],
              tsne_concat_df.loc[indices_2, 'tsne 2'], 
              c = color,
              s = 50)
    ax_2.legend(labels_list_2)
    ax_2.grid()


# Supervised
bool_array = np.random.rand(len(x)) < 0.7
x_train = x[bool_array]
y_train = y[bool_array]
x_test = x[~bool_array]
y_test = y[~bool_array]
# len(x_train), len(y_train), len(x_test), len(y_test)


# knn classification
knn = KNeighborsClassifier()
knn.fit(x_train, y_train.values.ravel())
knn_pred = knn.predict(x_test)
knn_train_acc, knn_test_acc = knn.score(x_train, y_train), knn.score(x_test, y_test)

# knn_count = 0
# knn_correct = 0
# for i in range(len(y_test)):
#     knn_count += 1
#     if knn_pred[i] == y_test.values.ravel()[i]:
#         knn_correct += 1
# knn_accuracy = knn_correct/knn_count


# SVM classification
svm = SVC()
svm.fit(x_train, y_train.values.ravel())
svm_train_acc, svm_test_acc = svm.score(x_train, y_train), svm.score(x_test, y_test)


# descisioin tree
tree_clf = DecisionTreeClassifier()
tree_clf.fit(x_train, y_train.values.ravel())
dt_train_acc, dt_test_acc = tree_clf.score(x_train, y_train), tree_clf.score(x_test, y_test)

st.write("""# Type of Learning""")
genre = st.radio(
     "",
     ('Unsupervised', 'Supervised'))
st.write("""# Output Models""")
if genre == 'Unsupervised':
     genre = st.radio("",
     ('PCA', 't-SNE'))
     if genre == 'PCA':
         st.write(concat_df.head())
         image = Image.open('PCA.png')
         st.image(image, caption='PCA plot')
     else:
         st.write(tsne_concat_df.head())
         image = Image.open('tSNE.png')
         st.image(image, caption='t-SNE plot')
else:
     st.write("Supervsed output")
     image = Image.open('RNA_exp.png')
     st.image(image, caption='model weight heatmap')

     genre = st.radio("Output Models",
     ('KNN', 'SVM', 'DT'))
     if genre == 'KNN':
         st.write('KNN Model Evaluations')
         st.write("Train Accuracy :", knn_train_acc)
         st.write("Test Accuracy :", knn_test_acc)
     elif genre == 'SVM':
         st.write('SVM Model Evaluations')
         st.write("Train Accuracy :", svm_train_acc)
         st.write("Test Accuracy :", svm_test_acc)
     else :
         st.write('DT Model Evaluations')
         image = Image.open('my_tree.png')
         st.image(image, caption='Characteristic Tree')
         st.write("Train Accuracy :", dt_train_acc)
         st.write("Test Accuracy :", dt_test_acc)