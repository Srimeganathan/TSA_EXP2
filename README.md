# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/content/car_price.csv')
data.dropna(subset=['Year'], inplace=True)
data['Year'] = data['Year'].astype(int)
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)
data.head()
resampled_data = data['Mileage'].resample('YE').sum().to_frame()
resampled_data.head()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Year': 'Year'}, inplace=True)
resampled_data.head()
years = resampled_data['Year'].tolist()
Mileage = resampled_data['Mileage'].tolist()
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, Mileage)]
n = len(years)
b = (n * sum(xy) - sum(Mileage) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(Mileage) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, Mileage)]
coeff = [[len(X), sum(X), sum(x2)],
[sum(X), sum(x2), sum(x3)],
[sum(x2), sum(x3), sum(x4)]]
Y = [sum(Mileage), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
resampled_data.set_index('Year',inplace=True)

# Linear Trend Plot
plt.figure(figsize=(10, 6))
plt.plot(resampled_data.index, resampled_data['Mileage'], label='Original Mileage', color='blue', marker='o', linestyle='-')
plt.plot(resampled_data.index, resampled_data['Linear Trend'], label='Linear Trend', color='black', linestyle='--')
plt.title('Linear Trend Estimation of Mileage')
plt.xlabel('Year')
plt.ylabel('Mileage')
plt.legend()
plt.grid(True)
plt.show()

# Polynomial Trend Plot
plt.figure(figsize=(10, 6))
plt.plot(resampled_data.index, resampled_data['Mileage'], label='Original Mileage', color='blue', marker='o', linestyle='-')
plt.plot(resampled_data.index, resampled_data['Polynomial Trend'], label='Polynomial Trend (Degree 2)', color='red', linestyle='-.')
plt.title('Polynomial Trend Estimation (Degree 2) of Mileage')
plt.xlabel('Year')
plt.ylabel('Mileage')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT
Linear Trend: y=14092100.54 + 14917.67x

Polynomial Trend: y=13793011.09 + 21192.28x + 6274.60x²
A - LINEAR TREND ESTIMATION
<img width="1056" height="688" alt="image" src="https://github.com/user-attachments/assets/66d8604f-8cba-4297-91a8-dcd7051377b2" />

B- POLYNOMIAL TREND ESTIMATION
![Uploading image.png…]()

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
