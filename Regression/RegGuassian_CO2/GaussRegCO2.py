#Gaussian process regression (GPR) on Mauna Loa CO2 data

import numpy as np, sklearn, matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

from sklearn.datasets import fetch_openml

print(__doc__)

def load_data():
    ml_data = fetch_openml(data_id = 41187)
    months_l = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:,0]
    m = ml_data.data[:,1]
    month_float = y + (m-1)/12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months_l or month != months_l[-1]:
            months_l.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)

        else: #aggrigate monthly sum to get average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months_l).reshape(-1,1)
    avg_ppms = np.asarray(ppmv_sums) / counts
    return months, avg_ppms

x,y = load_data()

#Kernel with parameters given in GPML book
k1 = 0.66 ** 2 * RBF(length_scale = 67.0) #Long term smooth rising trend
k2 = 2.4 ** 2 * RBF(length_scale = 90.0) * ExpSineSquared(length_scale = 1.3, periodicity = 1.0) #Seasonal component
k3 = 0.66 ** 2 * RationalQuadratic(length_scale = 1.2, alpha = 0.78) #Medium term irregularity
k4 = 0.18 ** 2 * RBF(length_scale = 0.134) + WhiteKernel(noise_level = 0.19 ** 2) #Noise terms

kernel_gpml = k1 + k2 + k3 + k4

gp = GPR(kernel = kernel_gpml, alpha = 0, optimizer = None, normalize_y = True)
gp.fit(x,y)

print('GP-ML kernel: %s' %gp.kernel_, sep = '\n')
print('Log-marginal likelihood: %.3f' %round(gp.log_marginal_likelihood(gp.kernel_.theta),0))

#Kernel with optimized parameters
k21 = 50 ** 2 * RBF(length_scale = 50.0)
k22 = 0.1 ** 2 * RBF(length_scale = 100.0) * ExpSineSquared(length_scale = 1.0, periodicity = 1.0, periodicity_bounds = 'fixed')
k23 = 0.5 ** 2 * RationalQuadratic(length_scale = 1.0, alpha = 1.0)
k24 = 0.1 ** 2 * RBF(length_scale = 0.1) + WhiteKernel(noise_level = 0.1 ** 2, noise_level_bounds = (1e-3, np.inf))

kernel = k21 + k22 + k23 + k24

gp2 = GPR(kernel = kernel, alpha = 0, normalize_y = True)
gp2.fit(x,y)

print('\nLearned Kernel: %s' %gp2.kernel_, sep = '\n')
print('Log marginal likelihood: %.3f' %round(gp2.log_marginal_likelihood(gp2.kernel_.theta),0))

x_ = np.linspace(x.min(), x.max() + 50, 10000)[:, np.newaxis]
y_pred, y_std = gp2.predict(x_, return_std = True)

#Plot
plt.scatter(x,y,c='k')
plt.plot(x_, y_pred)
plt.fill_between(x_[:,0], y_pred - y_std, y_pred + y_std, alpha = 0.5, color = 'k')
plt.xlim(x_.min(), x_.max())

plt.xlabel('Year')
plt.ylabel(r'CO$_2$ (ppm)')
plt.title(r'Atmospheric CO$_2$ ppm (Mauna Loa)')
plt.tight_layout()

plt.show()