'''<--------------------IMPORTING THE REQUIRED PACKAGES------------------------------------->'''
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


'''<----------------------------Functions Starts-------------------------------------->'''

def import_data():
    
    '''Function to import the PCA_dataset'''
    
    dataset=pd.read_csv('PCA_practice_dataset.csv',sep=',',header=None)

    df = pd.DataFrame(dataset)
    
    #df.keys()
    
    return dataset,df
def data_slicing(df):
    
    data=np.array(df).transpose()
    #print(data.shape)

    cov=np.cov(np.cov(data))
    #print(cov.shape)

    '''------------- mask the co varience matrix because it is symmetric---------'''
    mask = np.zeros_like(cov , dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(10,6))
    
    cmap = sns.diverging_palette(20, 10, as_cmap=True)
    sns.heatmap(cov, mask=mask,cmap= cmap, center=0, linewidths=.7)

    # FIND Eigen values and eigen vectors
    
    eig_val,eig_vec=np.linalg.eig(cov) #Calculating the  eigen vectors and eigen values
    
    plt.figure(figsize=(8,6))
    sns.lineplot(range(35),eig_val) #Ploting  the eigen values

    plt.show()

    return eig_vec,eig_val,cov

def Principal_Component(dataset,eig_vec,eig_val):

    data=np.matrix(dataset)
    eigen_vec = []
    for i in range(eig_vec.shape[1]):
        eig = data@eig_vec[:,i]
        eig = eig/eig_val[i]
        eigen_vec.append(np.ravel(eig))
    a=np.matrix(eigen_vec)
    print(a.shape)





    index = np.argsort(eig_val) 
    index = index[::-1] #Sorting the indexes in Descending order
    
    sum_of_eig_val = np.sum(eig_val)
    temp = 0
    principal_eig_vec = []
    principal_eig_val = []

    k=0 #Temporary variable

    while(temp<0.98*sum_of_eig_val):
        principal_eig_vec.append(eigen_vec[index[k]])
        principal_eig_val.append(eig_val[index[k]])
        temp += eig_val[index[k]]
        k += 1
    print("Number of  principal components are %d"%k)
    sns.heatmap(principal_eig_vec)


    print("Principal eigen vectors are:",principal_eig_vec)
    print("Principal eigen values are:",principal_eig_val)


    return principal_eig_vec

'''<-------------------------- End of Functions ---------------------------------------------->'''

if __name__=='__main__':

    dataset , df = import_data() #Import the PCA dataset
    eig_vec , eig_val ,cov = data_slicing(df) #Calling the dat_slicing function for numerical calculations 
    principal_eig_vec = Principal_Component(dataset,eig_vec,eig_val) #Function call to find Principal component

    result = np.array(principal_eig_vec) #List  to numpy array
    result = result.transpose()
    '''<---------------------GRAPHS------------------------------------------------------>'''
    
    plt.figure(figsize=(12,6))
    sns.heatmap(cov,cmap='plasma')

    plt.figure(figsize=(8,6))
    plt.scatter(result[:,0],result[:,1],cmap='plasma')
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')

    plt.show()

    
