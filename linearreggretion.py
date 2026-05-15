import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv('Automobile.csv')
#data after cleaning
#single variable linear regression
theta0=0
theta1=0
alpha=0.9
#feature scaling
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['horsepower'])
df['horsepower_FS']=(df['horsepower']-df['horsepower'].min())/(df['horsepower'].max()-df['horsepower'].min())
#gradient descent
for i in range(1000):
    df['y_']=(theta1*df['horsepower_FS'])+theta0
    error=df['y_']-df['mpg']
    mse=(error**2).sum() / (2 * len(df))
    
    if i%100==0:
        print(f"Iteration {i}: MSE = {mse}")
    gradient_theta1 = (error * df['horsepower_FS']).sum() / len(df)
    theta1=theta1-(alpha*gradient_theta1)
    gradient_theta0 = error.sum() / len(df)
    theta0=theta0-(alpha*gradient_theta0)
print(f"Final parameters: theta0 = {theta0}, theta1 = {theta1}")
plt.scatter(df['horsepower_FS'], df['mpg'], color='blue', alpha=0.5, label='Actual Data')
plt.plot(df['horsepower_FS'], df['y_'], color='red', linewidth=3, label='Gradient Descent Line')
plt.xlabel('Horsepower (Scaled)')
plt.ylabel('MPG')
plt.title('Our Gradient Descent from Scratch! 🚀')
plt.legend()
plt.show()
#----------------شويه توقعات للعربيات الجديدة----------------
newHorsePower=200 
newHorsePower_FS=(newHorsePower-df['horsepower'].min())/(df['horsepower'].max()-df['horsepower'].min())
predicted_mpg=(theta1*newHorsePower_FS)+theta0
print(f"Predicted MPG for a car with {newHorsePower} horsepower: {predicted_mpg:.2f}")
#--------------
newHorsePower=300
newHorsePower_FS=(newHorsePower-df['horsepower'].min())/(df['horsepower'].max()-df['horsepower'].min())
predicted_mpg=(theta1*newHorsePower_FS)+theta0
print(f"Predicted MPG for a car with {newHorsePower} horsepower: {predicted_mpg:.2f}")
#--------------------multiple variable linear regression----------------------
#feaure scaling
df['weight_FS']=(df['weight']-df['weight'].min())/(df['weight'].max()-df['weight'].min())
df['displacement_FS']=(df['displacement']-df['displacement'].min())/(df['displacement'].max()-df['displacement'].min())
#feature matrix
features=df[['horsepower_FS','weight_FS','displacement_FS']]
#bais term
features["bais"]=1
#initial parameters
th0,th1,th2,th3=0,0,0,0
thetaV=[th0,th1,th2,th3]
alpha=0.5
#gradient descent
for i in range(1000):
    features['y_']=(thetaV[0]*features['bais'])+(thetaV[1]*features['horsepower_FS'])+(thetaV[2]*features['weight_FS'])+(thetaV[3]*features['displacement_FS'])
    error=features['y_']-df['mpg']
    mse=(error**2).sum() / (2 * len(df))
    if i%100==0:
        print(f"Iteration {i}: MSE = {mse}")
    gradient_theta0 = error.sum() / len(df)
    gradient_theta1 = (error * features['horsepower_FS']).sum() / len(features) 
    gradient_theta2 = (error * features['weight_FS']).sum() / len(features)
    gradient_theta3 = (error * features['displacement_FS']).sum() / len(features)
    thetaV[0] = thetaV[0] - (alpha * gradient_theta0)
    thetaV[1] = thetaV[1] - (alpha * gradient_theta1)
    thetaV[2] = thetaV[2] - (alpha * gradient_theta2)
    thetaV[3] = thetaV[3] - (alpha * gradient_theta3)   

print(f"Theta0 (Bias) = {thetaV[0]}")
print(f"Theta1 (Horsepower) = {thetaV[1]}")
print(f"Theta2 (Weight) = {thetaV[2]}")
print(f"Theta3 (Displacement) = {thetaV[3]}")
#--------------------توقعات للعربيات الجديدة----------------
new_cars_data = np.array([
    [65.0,  1800.0, 90.0],   
    [130.0, 3100.0, 180.0],  
    [300.0, 4800.0, 400.0]  
])
p_scaled = (new_cars_data[:, 0] - df['horsepower'].min()) / (df['horsepower'].max() - df['horsepower'].min())
weight_scaled = (new_cars_data[:, 1] - df['weight'].min()) / (df['weight'].max() - df['weight'].min())
disp_scaled = (new_cars_data[:, 2] - df['displacement'].min()) / (df['displacement'].max() - df['displacement'].min())
X_new = np.hstack([
    np.ones((3, 1)), 
   p_scaled.reshape(-1, 1),
    weight_scaled.reshape(-1, 1),
    disp_scaled.reshape(-1, 1)
])

#------------------------كود احسن------------------------
n = len(df)
X = np.hstack([
    np.ones((n, 1)), 
    features['horsepower_FS'].values.reshape(-1, 1),
    features['weight_FS'].values.reshape(-1, 1),
    features['displacement_FS'].values.reshape(-1, 1)
])
y = df['mpg'].values.reshape(-1, 1)
Theta = np.zeros((4, 1))
alpha = 0.5
for i in range(1000):
    y_pred=X.dot(Theta)
    error=y_pred-y
    mse=(error**2).sum()/(2 * n)
    if i%100==0:
        print(f"Iteration {i}: MSE = {mse}")
    gradient = np.dot(X.T, error) / n
    Theta = Theta - (alpha * gradient)    
print(f"Theta0 (Bias) = {Theta[0][0]}")
print(f"Theta1 (Horsepower) = {Theta[1][0]}")
print(f"Theta2 (Weight) = {Theta[2][0]}")
print(f"Theta3 (Displacement) = {Theta[3][0]}")    