from scipy.optimize import fmin_l_bfgs_b

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, check_array, shuffle
from sklearn.utils.extmath import safe_sparse_dot

# transform the input X using a rectifier
def rectified_linear_unit(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

# transform the input Z through the derivative of the rectified linear function 
def rectified_linear_unit_derivative(Z):
    return (Z > 0).astype(Z.dtype)

def log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)) / y_prob.shape[0]

# stack layers attributes to be used by the bfgs optmizer
def stack(layers_coef_, layers_intercept_):
    return np.hstack([l.ravel() for l in layers_coef_ + layers_intercept_])

# Remember to set the class name appropiately.
class MatheusAlvesMLP(BaseEstimator, ClassifierMixin):  # or RegressonMixin?
    def __init__(self, params=None):
        if params is None:
            self.ctor({})
        else:
            self.ctor(params)

    def ctor(self, params):
        self.alpha = params.get("alpha", 0.00001) # L2 regularization
        self.max_iter = params.get("max_iter", 500)# max iteration to the optimization algorithm
        self.hidden_layers_size = params.get("hidden_layers_size", (100,200,300))
        self.shuffle = params.get("shuffle", False) # shuflle samples in interactions?
        self.random_state = params.get("random_state",None) # state or seed for generating random number
        self.tol = params.get("tol", 1e-5)# Loss tolerance for optimization

        self.layers_coef = None
        self.layers_intercept = None
        self.cost = None
        self.n_iter = 0
        self.classes = None
        self.label_binarizer_ = LabelBinarizer()    

    def _unstack(self, stacked_parameters):
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.layers_coef[i] = np.reshape(stacked_parameters[start:end], shape)
            start, end = self._intercept_indptr[i]
            self.layers_intercept[i] = stacked_parameters[start:end]

    def _forward_pass(self, activations, with_output_activation=True):
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i],self.layers_coef[i])
            activations[i + 1] += self.layers_intercept[i]

            # For the hidden layers
            if i + 1 != self.n_layers_ - 1:
                activations[i + 1] = rectified_linear_unit(activations[i + 1])

        # For the last layer
        if with_output_activation:
            activations[i + 1] = rectified_linear_unit(activations[i + 1])
        return activations

    def _compute_cost_grad(self, layer, n_samples, activations, deltas, coef_grads, intercept_grads):
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += (self.alpha * self.layers_coef[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

    def _cost_grad_lbfgs(self, stacked_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads):
        self._unstack(stacked_coef_inter)
        cost, coef_grads, intercept_grads = self._backprop(X, y, activations, deltas, coef_grads, intercept_grads)
        self.n_iter += 1
        grad = stack(coef_grads, intercept_grads)
        return cost, grad

    def _backprop(self, X, y, activations, deltas, coef_grads,intercept_grads):
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get cost using log loss function
        cost = log_loss(y, activations[-1])

        # Add regularization term to the cost
        values = np.sum(np.array([np.sum(s ** 2) for s in self.layers_coef]))
        cost += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
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
            deltas[i - 1] = safe_sparse_dot(deltas[i],self.layers_coef[i].T)
            deltas[i - 1] *= rectified_linear_unit_derivative(activations[i])
            coef_grads, intercept_grads = self._compute_cost_grad(i - 1,
                                                          n_samples,
                                                          activations,
                                                          deltas,
                                                          coef_grads,
                                                          intercept_grads)
        return cost, coef_grads, intercept_grads   

    def fit(self, X, y):
        hidden_layers_size = list(self.hidden_layers_size)
        n_samples, n_features = X.shape
        self.label_binarizer_.fit(y)

        if self.classes is None:
            self.classes = self.label_binarizer_.classes_
        else:
            classes = self.label_binarizer_.classes_

        y = self.label_binarizer_.transform(y)
        self.n_outputs = y.shape[1]
        layer_units = ([n_features] + hidden_layers_size +
                       [self.n_outputs])

        # If it is the first time training the model
        if self.layers_coef is None:
            # Initialize parameters
            self.n_outputs = y.shape[1]

            # Compute the number of layers
            self.n_layers_ = len(layer_units)

            # Initialize coefficient and intercept layers
            self.layers_coef = []
            self.layers_intercept = []

            for i in range(self.n_layers_ - 1):
                rng = check_random_state(self.random_state)
                n_fan_in = layer_units[i]
                n_fan_out = layer_units[i + 1]

                # Use the Gorot initialization method
                weight_init_bound = np.sqrt(6. / (n_fan_in + n_fan_out))
                self.layers_coef.append(rng.uniform(-weight_init_bound,
                                                     weight_init_bound,
                                                     (n_fan_in, n_fan_out)))
                self.layers_intercept.append(rng.uniform(-weight_init_bound,
                                                          weight_init_bound,
                                                          n_fan_out))
        if self.shuffle:
            X, y = shuffle(X, y, random_state=self.random_state)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((n_samples, n_fan_out))
                           for n_fan_out in layer_units[1:])

        deltas = [np.empty_like(a_layer) for a_layer in activations]
        coef_grads = [np.empty((n_fan_in, n_fan_out)) for n_fan_in, n_fan_out in zip(layer_units[:-1], layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out) for n_fan_out in layer_units[1:]]
        
        # START LBFGS algorithm
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unstacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unstacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # enable pretty output for l_bfgs_b
        iprint = 1
        
        # Run L-BFGS_B opitimization
        stacked_coef_inter = stack(self.layers_coef, self.layers_intercept)

        optimal_parameters, self.cost, d = fmin_l_bfgs_b(
            x0=stacked_coef_inter,
            func=self._cost_grad_lbfgs,
            maxfun=self.max_iter,
            iprint=iprint,
            pgtol=self.tol,
            args=(X, y, activations, deltas, coef_grads, intercept_grads))

        self._unstack(optimal_parameters)

        return self  

    def decision_function(self, X):
        hidden_layers_size = list(self.hidden_layers_size)

        layer_units = [X.shape[1]] + hidden_layers_size + [self.n_outputs]

        # Initialize layers
        activations = []
        activations.append(X)

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations, with_output_activation=False)
        y_pred = activations[-1]

        if self.n_outputs == 1:
            return y_pred.ravel()
        else:
            return y_pred

    def predict(self, X):
        y_scores = self.decision_function(X)
        y_scores = rectified_linear_unit(y_scores)

        return self.label_binarizer_.inverse_transform(y_scores)