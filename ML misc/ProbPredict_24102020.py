#Probabilistic Predictions
#Gaussian Process Classification
#Link: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc.html#sphx-glr-auto-examples-gaussian-process-plot-gpc-py

print(__doc__)
import numpy as np, matplotlib.pyplot as plt, sklearn
from sklearn.metrics import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier as GPC 
from sklearn.gaussian_process.kernels import RBF

#Generate Data
train_size = 1000000
r = np.random.RandomState(0)
x = r.uniform(0, 5, 100)[:, np.newaxis]
y = np.array(x[:,0] > 2.5, dtype = int)

#Specify Guassian Process - fixed / optimised hyperparameters
gp_fix = GPC(kernel = 1 * RBF(length_scale =1.0), optimizer = None)
gp_fix.fit(x[:train_size], y[:train_size])

gp_opt = GPC(kernel = 1 * RBF(length_scale = 1.0))
gp_opt.fit(x[:train_size], y[:train_size])

print("Log Marginal Likelihood (initial): %.3f"
      % gp_fix.log_marginal_likelihood(gp_fix.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta))

print("Accuracy: %.3f (initial) %.3f (optimized)"
      % (accuracy_score(y[:train_size], gp_fix.predict(x[:train_size])),
         accuracy_score(y[:train_size], gp_opt.predict(x[:train_size]))))
print("Log-loss: %.3f (initial) %.3f (optimized)"
      % (log_loss(y[:train_size], gp_fix.predict_proba(x[:train_size])[:, 1]),
         log_loss(y[:train_size], gp_opt.predict_proba(x[:train_size])[:, 1])))

# Plot posteriors
plt.figure()
plt.scatter(x[:train_size, 0], y[:train_size], c='k', label="Train data",
            edgecolors=(0, 0, 0))
plt.scatter(x[train_size:, 0], y[train_size:], c='g', label="Test data",
            edgecolors=(0, 0, 0))
x_ = np.linspace(0, 5, 100)
plt.plot(x_, gp_fix.predict_proba(x_[:, np.newaxis])[:, 1], 'r',
         label="Initial kernel: %s" % gp_fix.kernel_)
plt.plot(x_, gp_opt.predict_proba(x_[:, np.newaxis])[:, 1], 'b',
         label="Optimized kernel: %s" % gp_opt.kernel_)
plt.xlabel("Feature")
plt.ylabel("Class 1 probability")
plt.xlim(0, 5)
plt.ylim(-0.25, 1.5)
plt.legend(loc="best")

# Plot LML landscape
plt.figure()
theta0 = np.logspace(0, 8, 30)
theta1 = np.logspace(-1, 1, 29)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML = [[gp_opt.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
LML = np.array(LML).T
plt.plot(np.exp(gp_fix.kernel_.theta)[0], np.exp(gp_fix.kernel_.theta)[1],
         'ko', zorder=10)
plt.plot(np.exp(gp_opt.kernel_.theta)[0], np.exp(gp_opt.kernel_.theta)[1],
         'ko', zorder=10)
plt.pcolor(Theta0, Theta1, LML)
plt.xscale("log")
plt.yscale("log")
plt.colorbar()
plt.xlabel("Magnitude")
plt.ylabel("Length-scale")
plt.title("Log-marginal-likelihood")

plt.show()