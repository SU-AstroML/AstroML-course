import numpy as np

class Galaxy_Parser:
#Class to contain data 
    data=[]
    data_test=[]
    data_train=[]
    labels=[]
    labels_test=[]
    labels_train=[]
    datanames=[]

    def __init__(self,fname,precondition=False,replaceMean=False,ellipticity_threshold=0.5,trainfrac=0.8):
#When initialising, some options may be set: 
#-precondition: if True, scale each feature vector to have 0 mean and stdev =1. Default=False
#-replaceMean: if True, replace missing features of a galaxy with the average of that feature. Default = False
#-ellipticity_threshold: float: if ellipticity is above this value, it will be considered a "true" elliptical galaxy.
#Default = 0.5
#-trainfrac : float: fraction of data to use for training. The remainder will be used for test dataset. Default =0.8

        self.datanames=["cntr_01","name","RA","Dec","elliptical","spiral","photoz","photozErr","FUV","NUV","U","G","R","I","Z","J","H","K","W1","W2","W3","W4"]
        datanames=self.datanames
        
        data = np.loadtxt(fname,skiprows=1)
        #replace -9999 with nan: 
        data[data==-9999]=np.nan

        #index of first magnitude variable
        n_not_color=8
        n_color = data.shape[1]-n_not_color
        #shuffle the data to avoid bias on training/test data: 
        
        np.random.shuffle(data)
        
        #labels are 1 if elliptical<threshold (i.e. 1=spirals)
        labels = (ellipticity_threshold<=data[:,4])*1.
        
        #now to compute the features. There are sum_1^n_color 1 color numbers, 
        #as well as photometric redshift and the attendant error
        tdata = data[:,n_not_color-2:n_not_color]
        tdatanames = self.datanames[n_not_color-2:n_not_color] 
        
        #compute all differences between two spectral magnitudes:

        n_nonan=0
        for i in range(n_color):
            ii=i+n_not_color
            for j in range(i+1,n_color):
                jj=j+n_not_color
                color = data[:,ii]-data[:,jj]

                #scale to mean and stdev if precondition
                if precondition:
                    color = (color-np.nanmean(color))/np.nanstd(color)

                #replace nan with zeros or, if replaceMean, the column average
                if (np.isnan(color).any(axis=0)):
                    if replaceMean:
                        c1 = np.nanmean(data[:,ii])
                        c2 = np.nanmean(data[:,jj])
                        c = c1-c2
                        color[np.isnan(color)]=c
                    else:
                        color = np.zeros(len(color))
                else:
                    n_nonan+=1
                
                tdatanames.append(datanames[ii]+"-"+datanames[jj])
                tdata = np.hstack([tdata,color.reshape(-1,1)])
        
        data = tdata #avoid to keep two arrays
        self.datanames=tdatanames
        data[np.isnan(data)]=-9999.


        #split data in training and test sample, 80-20
        n_samples = data.shape[0]
        self.data_train = data[:n_samples*trainfrac, :]
        self.data_test = data[n_samples*trainfrac:, :]
        self.labels_train = labels[:n_samples*trainfrac]
        self.labels_test = labels[n_samples*trainfrac:]
        print "Dataset ",fname, " loaded."
        print "Containing ",len(data)," Galaxies"
        print "In total, ",len(self.datanames),"feature columns are included"
        if precondition:
            print "The features have been scaled to have a mean of 0 and stdev of 1"
        if replaceMean: 
            print "Missing data has been replaced with the mean of the relevant feature column"
        else:
            print "Missing data is replaced with 0"
        print len(self.data_train)," galaxies are in the training set, ",len(self.data_test),"in the training sample"
