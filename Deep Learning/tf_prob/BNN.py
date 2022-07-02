# Probabilistic Baysean Neural Network

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

# Training / Eval data

def data_splitter(train_size, batch_size = 1):
    data = (
        tfds.load(name = 'wine_quality', as_supervised=True, split='train',)
        .map(lambda x, y : (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )

    # shuffle with buffer size = dataset size
    train_data = (
        data.take(train_size).shuffle(buffer_size = train_size).batch(batch_size)
    )

    test_data = data.skip(train_size).batch(batch_size)

    return train_data, test_data


# Compile

hidden_units = [8,8]
learning_rate = 0.01

def experiment(model, loss, train_data, test_data):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print('Experiment started...')
    model.fit(train_data, epochs = num_epochs, validation_data = test_data)
    print('Training complete...')
    _ , rmse = model.evaluate(train_data, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _ , rmse = model.evaluate(test_data, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

# Model inputs

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def create_model_inputs():
    inputs = {}
    
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(name=feature_name, shape=(1,), dtype=tf.float32)
    
    return inputs


# We create a standard deterministic neural network model as a baseline.

def create_baseline_model():
    inputs = create_model_inputs()
    input_values = [value for _, value in sorted(inputs.items())]
    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)

    # Create hidden layers with deterministic weights using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(units, activation="sigmoid")(features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# We use the tfp.layers.DenseVariational layer instead of the standard keras.layers.Dense layer in the neural network model.

def create_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Probabilistic Bayesian neural network

'''
    In this example, we model the output as a IndependentNormal distribution, 
    with learnable mean and variance parameters. 
    
    If the task was classification, we would have used IndependentBernoulli with binary classes, 
    and OneHotCategorical with multiple classes, to model distribution of the model output.
'''

def prob_BNN(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layer with weight uncertanty using DenseVariational layer
    for unit in hidden_units:
        features = tfp.layers.DenseVariational(
            units = unit,
               make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)
        

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.

    dist_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(dist_params)

    return keras.Model(inputs = inputs, outputs = outputs)


# Since the output of the model is a distribution, rather than a point estimate, we use the 
# negative loglikelihood as our loss function to compute how likely to see the 
# true data (targets) from the estimated distribution produced by the model.

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


# Compute

def compute_predictions(model, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    for idx in range(sample):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )

if __name__ == '__main__':
    # Let's split the wine dataset into training and test sets, with 85% and 15% of the examples.
    dataset_size = 4898
    batch_size = 256
    train_size = int(dataset_size * 0.85)
    train_dataset, test_dataset = data_splitter(train_size, batch_size)

    # Now let's train the baseline model. We use the MeanSquaredError as the loss function.
    num_epochs = 100
    mse_loss = keras.losses.MeanSquaredError()
    baseline_model = create_baseline_model()
    experiment(baseline_model, mse_loss, train_dataset, test_dataset)


    #Train BNN with a small training subset.
    num_epochs = 500
    train_sample_size = int(train_size * 0.3)
    small_train_dataset = train_dataset.unbatch().take(train_sample_size).batch(batch_size)

    bnn_model_small = create_bnn_model(train_sample_size)
    experiment(bnn_model_small, mse_loss, small_train_dataset, test_dataset)
    compute_predictions(bnn_model_small)


    # Train BNN with the whole training set.
    num_epochs = 500
    bnn_model_full = create_bnn_model(train_size)
    experiment(bnn_model_full, mse_loss, train_dataset, test_dataset)

    compute_predictions(bnn_model_full)


    # Prob model

    num_epochs = 100
    prob_bnn_model = prob_BNN(train_size)
    experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)

    sample = 10
    examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[0]

    predicted = baseline_model(examples).numpy()
    for idx in range(sample):
        print(f"Predicted: {round(float(predicted[idx][0]), 1)} - Actual: {targets[idx]}")

    prediction_distribution = prob_bnn_model(examples)
    prediction_mean = prediction_distribution.mean().numpy().tolist()
    prediction_stdv = prediction_distribution.stddev().numpy()

    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()

    for idx in range(sample):
        print(
            f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
            f"stddev: {round(prediction_stdv[idx][0], 2)}, "
            f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
            f" - Actual: {targets[idx]}"
        )