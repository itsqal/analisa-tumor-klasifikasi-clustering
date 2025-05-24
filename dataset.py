from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import RobustScaler
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

scaler = RobustScaler()
X = scaler.fit_transform(X)