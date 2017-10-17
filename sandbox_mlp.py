from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, check_array, shuffle
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np

def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

def relu_derivative(Z):
    return (Z > 0).astype(Z.dtype)

def log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_prob) +
                  (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]

def _pack(layers_coef_, layers_intercept_):
    return np.hstack([l.ravel() for l in layers_coef_ + layers_intercept_])

class MultilayerPerceptronClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        if params is None:
            self.ctor({})
        else:
            self.ctor(params)

    def ctor(self, params):
        self.alpha = params.get("alpha", 0.00001)
        self.learning_rate = params.get("learning_rate", "constant")
        self.learning_rate_init = params.get("learning_rate_init", 0.5)
        self.max_iter = params.get("max_iter", 200)
        self.hidden_layer_sizes = params.get("hidden_layer_sizes", 100)
        self.shuffle = params.get("shuffle", False) 
        self.random_state = params.get("random_state",None)
        self.tol = params.get("tol", 1e-5)

        self.verbose = True
        self.layers_coef_ = None
        self.layers_intercept_ = None
        self.cost_ = None
        self.n_iter_ = None
        self.learning_rate_ = None
        self.classes_ = None
        self.t_ = None
        self.label_binarizer_ = LabelBinarizer()    

    def _unpack(self, packed_parameters):
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.layers_coef_[i] = np.reshape(packed_parameters[start:end],
                                              shape)

            start, end = self._intercept_indptr[i]
            self.layers_intercept_[i] = packed_parameters[start:end]

    def _forward_pass(self, activations, with_output_activation=True):
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],
                                                 self.layers_coef_[i])
            activations[i + 1] += self.layers_intercept_[i]

            # For the hidden layers
            if i + 1 != self.n_layers_ - 1:
                activations[i + 1] = relu(activations[i + 1])

        # For the last layer
        if with_output_activation:
            activations[i + 1] = relu(activations[i + 1])

        return activations

    def _compute_cost_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.layers_coef_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

    def _cost_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas,
                         coef_grads, intercept_grads):
        self._unpack(packed_coef_inter)
        cost, coef_grads, intercept_grads = self._backprop(X, y, activations,
                                                           deltas, coef_grads,
                                                           intercept_grads)
        self.n_iter_ += 1
        grad = _pack(coef_grads, intercept_grads)
        return cost, grad

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        n_samples = X.shape[0]

        # Step (1/3): Forward propagate
        activations = self._forward_pass(activations)

        # Step (2/3): Get cost
        cost = log_loss(y, activations[-1])
        # Add L2 regularization term to cost
        values = np.sum(np.array([np.sum(s ** 2) for s in self.layers_coef_]))
        cost += (0.5 * self.alpha) * values / n_samples

        # Step (3/3): Backward propagate
        last = self.n_layers_ - 2

        diff = y - activations[-1]
        deltas[last] = -diff

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_cost_grad(last, n_samples,
                                                              activations,
                                                              deltas,
                                                              coef_grads,
                                                              intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i],
                                            self.layers_coef_[i].T)
            deltas[i - 1] *= relu_derivative(activations[i])

            coef_grads, intercept_grads = self._compute_cost_grad(i - 1,
                                                          n_samples,
                                                          activations,
                                                          deltas,
                                                          coef_grads,
                                                          intercept_grads)

        return cost, coef_grads, intercept_grads   

    def fit(self, X, y):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        n_samples, n_features = X.shape

        self.label_binarizer_.fit(y)

        if self.classes_ is None:
            self.classes_ = self.label_binarizer_.classes_
        else:
            classes = self.label_binarizer_.classes_
            if not np.all(np.in1d(classes, self.classes_)):
                raise ValueError("`y` has classes not in `self.classes_`."
                                 " `self.classes_` has %s. 'y' has %s." %
                                 (self.classes_, classes))

        y = self.label_binarizer_.transform(y)

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # First time training the model
        if self.layers_coef_ is None:
            # Initialize parameters
            self.n_iter_ = 0
            self.t_ = 0
            self.learning_rate_ = self.learning_rate_init
            self.n_outputs_ = y.shape[1]

            # Compute the number of layers
            self.n_layers_ = len(layer_units)

            # Initialize coefficient and intercept layers
            self.layers_coef_ = []
            self.layers_intercept_ = []

            for i in range(self.n_layers_ - 1):
                rng = check_random_state(self.random_state)

                n_fan_in = layer_units[i]
                n_fan_out = layer_units[i + 1]

                # Use the initialization method recommended by
                # Glorot et al.
                weight_init_bound = np.sqrt(6. / (n_fan_in + n_fan_out))

                self.layers_coef_.append(rng.uniform(-weight_init_bound,
                                                     weight_init_bound,
                                                     (n_fan_in, n_fan_out)))
                self.layers_intercept_.append(rng.uniform(-weight_init_bound,
                                                          weight_init_bound,
                                                          n_fan_out))

        if self.shuffle:
            X, y = shuffle(X, y, random_state=self.random_state)
        batch_size = n_samples

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in, n_fan_out)) for n_fan_in,
                      n_fan_out in zip(layer_units[:-1],
                                       layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out) for n_fan_out in
                           layer_units[1:]]
        
        # START LBFGS algorithm
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.layers_coef_,
                                  self.layers_intercept_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        optimal_parameters, self.cost_, d = fmin_l_bfgs_b(
            x0=packed_coef_inter,
            func=self._cost_grad_lbfgs,
            maxfun=self.max_iter,
            iprint=iprint,
            pgtol=self.tol,
            args=(X, y, activations, deltas, coef_grads, intercept_grads))

        self._unpack(optimal_parameters)

        return self

    def _decision_scores(self, X):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + \
            [self.n_outputs_]

        # Initialize layers
        activations = []
        activations.append(X)

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations, with_output_activation=False)
        y_pred = activations[-1]

        return y_pred

    

    def decision_function(self, X):
        y_scores = self._decision_scores(X)

        if self.n_outputs_ == 1:
            return y_scores.ravel()
        else:
            return y_scores

    def predict(self, X):
        y_scores = self.decision_function(X)
        y_scores = relu(y_scores)

        return self.label_binarizer_.inverse_transform(y_scores)