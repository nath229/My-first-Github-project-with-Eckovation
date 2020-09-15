'''<----------------Importing the required libraries------------------------->''' 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import tree
import pickle
import warnings
style.use('ggplot')
warnings.filterwarnings('ignore')

'''<------------Functions----------------------->'''
def load_data():
    dataset = pd.read_csv('heart_disease.csv')

    return dataset

def dataset_cleaning(dataset):

    print('Ckecks the dataset for null values:\n',dataset.isnull().any())


    return dataset

def plotting(dataset):

    data = dataset.drop(columns= ['chol','sex','fbs','restecg','oldpeak'])
    sns.pairplot(data,hue='target', palette='RdBu')
    plt.show()

def train_test_split(cleaned_data ,target):

    from sklearn.model_selection import train_test_split
    
    X_train,X_test,y_train,y_test = train_test_split(cleaned_data,target,test_size=0.3,random_state = 100)

    return X_train,X_test,y_train,y_test

def K_Nearest_Neighbors_Classifier(X_train,y_train,X_test,y_test):

    #Optimization technique
    error_rate= []
    for i in range(1,15):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))#error rate
    
    plt.figure(figsize=(10,6))
    plt.plot(range(1,15),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
    plt.title('KNN Error diagram')
    plt.xlabel('K Value')
    plt.ylabel('Error Rate')
    plt.show()

    #for n_neighbour 8 the error is minimum
    
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train,y_train)
    accuracy_score = knn.score(X_test,y_test)
    print('Accuracy score of KNN is',accuracy_score)

    pred = knn.predict(X_test)
    print('Confusion Matrix of KNN is:',confusion_matrix(y_test,pred))
    print('Clasification Report of KNN is:',classification_report(y_test,pred))
    return accuracy_score

def Decision_tree_Classifier(X_train,y_train,X_test,y_test):

    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import  pydotplus

    #Optimization technique
    error_rate= []
    for i in range(2,15):
        clf = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=i,random_state=100)
        clf.fit(X_train,y_train)
        pred_i = clf.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))#error
    
    plt.figure(figsize=(10,6))
    plt.plot(range(2,15),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
    plt.title('DTC Error diagram')
    plt.xlabel('max leaf')
    plt.ylabel('Error Rate')
    plt.show()

    #for n_neighbour 8 the error is minimum
    
    clf = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=8,random_state=100)
    clf.fit(X_train,y_train)
    accuracy_score = clf.score(X_test,y_test)
    print('Accuracy score of DTC is',accuracy_score)

    pred = clf.predict(X_test)
    print('Confusion Matrix of DTC is:',confusion_matrix(y_test,pred))
    print('Clasification Report of DTC is:',classification_report(y_test,pred))
    
    #Dendogram of the tree
    '''
    dot_data = export_graphviz(clf,out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)

    Image(graph.create_png())

    '''
    return accuracy_score

def Random_forest_Classifier(X_train,y_train,X_test,y_test):

    #Optimization technique
    wcss = []
    for i in range(2,15):
        ds = RandomForestClassifier(n_estimators=i,random_state=100)
        ds.fit(X_train,y_train)
    
        pred=ds.predict(X_test)
        wcss.append(np.mean(pred != y_test))
    
    plt.figure(figsize=(10,6))
    plt.plot(range(2,15),wcss,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
    plt.title('Rf classifier')
    plt.xlabel('estimator')
    plt.ylabel('Error Rate')
    plt.show()

    #for n_neighbour 8 the error is minimum
    
    model = RandomForestClassifier(n_estimators=8,random_state=0)
    model.fit(X_train,y_train)
    accuracy_score = model.score(X_test,y_test)
    print('Accuracy score of RFC is',accuracy_score)

    pred = ds.predict(X_test)
    print('Confusion Matrix of RF is:',confusion_matrix(y_test,pred))
    print('Clasification Report of RF is:',classification_report(y_test,pred))
    
    #Dendogram of the forest
    
    plt.figure(figsize=(20,20))
    tree.plot_tree(model.estimators_[1],filled=True)

    plt.show()

    return accuracy_score

def SVM(X_train,X_test,y_train,y_test):

    from sklearn import svm

    #kernel = 'linear','poly',LinearSVC,'smf'
    error_rate=[]
    for i in range(1,15):
        clf = svm.SVC(kernel='linear',C=i ).fit(X_train, y_train)
        pred_i = clf.predict(X_test)
        error_rate.append(np.mean(pred_i!=y_test))

    plt.figure(figsize=(10,10))
    plt.plot(range(1,15),error_rate,marker = 'o',markerfacecolor = 'red',linestyle='dashed',color='blue',markersize=10)
    plt.title('SVM Classifier')
    plt.xlabel('C value')
    plt.ylabel('Error rate')
    plt.show()

    # c=6 minimum error
    clf = svm.SVC(kernel='linear',C=6 ).fit(X_train, y_train)
    accuracy_score = clf.score(X_test,y_test)

    print('Accuracy score of SVM is',accuracy_score)

    pred = clf.predict(X_test)
    print('Confusion Matrix of SVM is:',confusion_matrix(y_test,pred))
    print('Clasification Report of SVM is:',classification_report(y_test,pred))
    return accuracy_score

def Feature_Selection(dataset):

    '''<------Finds the coorelation between the fetures.
    Lower correlation features can be droped out------------------>'''
    
    corr=dataset.corr().round(3)
    print("Prints the correlation\n")
    print(corr)

    '''<----Masking the dataset since it is symmteric-------->'''
    
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    a,b=plt.subplots(figsize=(30,30))

    cmap = sns.diverging_palette(20, 10, as_cmap=True)
    
    #heatmap of features
    sns.heatmap(corr, mask=mask,cmap= cmap,vmin=-1, vmax=1, center=0,
                square=True, linewidths=.7, cbar_kws={"shrink": 0.6}, annot=True)
    plt.title('Heatmap of dataset features correlation')
    plt.show()
    
    #drops out the less correlated features

    drop_columns = ['chol','fbs','restecg']
    dataset = dataset.drop(drop_columns,axis=1)

    print('New dataset',dataset.head())
    print('New dataset shape',dataset.shape)

    '''
    sns.pairplot(dataset[dataset.columns],size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
    plt.show()
    '''
    cleaned_data = dataset.iloc[:,0:10]

    target = dataset.iloc[:,10]

    return cleaned_data ,target

def Best_fit(X_train,X_test,y_train,):

    from sklearn import svm
    clf = svm.SVC(kernel='linear',C=6 ).fit(X_train, y_train)
    pred = clf.predict(X_test)

    conf_matrix = confusion_matrix(y_test,pred)
    print('Confusion Matrix of SVM is:',conf_matrix)
    print('Clasification Report of SVM is:',classification_report(y_test,pred))

    plt.figure(figsize=(7,7))
    sns.heatmap(conf_matrix,annot=True)
    plt.title('SVM Confusion matrix heatmap')
    plt.show()

    return pred,clf
    
def data_processing(cleaned_data):

    sc = StandardScaler()
    standard_data = sc.fit_transform(cleaned_data)
    print('Standard dataset is:\n',standard_data)

    return standard_data

def Application(best_clf,X_test,y_test):

    pickle.dump(best_clf,open('heart_disease_detector.pickle','wb'))#Our model application

    heart_disease_detector_model = pickle.load(open('heart_disease_detector.pickle','rb'))#Model is used to test

    y_pred = heart_disease_detector_model.predict(X_test)
    print('Accuracy score',accuracy_score(y_test,y_pred))
    
'''<-----------------main function satrts------------------->'''

dataset = load_data() #loading the dataset

print('Dataset samples:\n',dataset.head(),end='\n')
print('Shape of the dataset is: ',dataset.shape,'\n')
print('Useful information of the dataset:\n',dataset.describe(),'\n')
print(dataset.info())
print('Dataset features keys:',list(dataset.columns),'\n')


dataset = dataset_cleaning(dataset)
plotting(dataset)

cleaned_data ,target = Feature_Selection(dataset)

#Standarizing the data
standard_data = data_processing(cleaned_data)

#Splits the data    
X_train,X_test,y_train,y_test = train_test_split(standard_data ,target)


#Accuracy list of the different classifier
accuracy_list = []

decision_tree_score = Decision_tree_Classifier(X_train,y_train,X_test,y_test)
accuracy_list.append(decision_tree_score)
KNN_score = K_Nearest_Neighbors_Classifier(X_train,y_train,X_test,y_test)
accuracy_list.append(KNN_score)
Rf_score = Random_forest_Classifier(X_train,y_train,X_test,y_test)
accuracy_list.append(Rf_score)
svm_score = SVM(X_train,X_test,y_train,y_test)
accuracy_list.append(svm_score)

print(accuracy_list)


predictions,best_clf = Best_fit(X_train,X_test,y_train,)
print('predictions are',predictions )

Application(best_clf,X_test,y_test)#Heart disease detector application

#final_df = pd.DataFrame(X_test)
#print('Final Dataframe is:\n',final_df)







