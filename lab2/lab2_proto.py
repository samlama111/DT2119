import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models according to the structure given in the image.

    Args:
        hmm1, hmm2: two dictionaries with the following keys:
            name: phonetic or word symbol corresponding to the model
            startprob: M+1 array with a priori probability of state
            transmat: (M+1)x(M+1) transition matrix
            means: MxD array of mean vectors
            covars: MxD array of variances

    Output:
        dictionary with the same keys but concatenated models:
            startprob: K+1 array with priori probability of state
            transmat: (K+1)x(K+1) transition matrix
            means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models plus one for the non-emitting state.
    """

    # Initialize start probability vector and transition matrix using nr of emitting states for hmm1 and hmm2.
    M1 = len(hmm1['startprob'])
    M2 = len(hmm2['startprob'])
    startprob = np.zeros(M1 + M2 - 1)
    transmat = np.zeros((M1 + M2 - 1, M1 + M2 - 1))

    # Calculate startprob
    startprob[:M1-1] = hmm1['startprob'][:-1]
    startprob[M1-1:] = hmm1['startprob'][-1] * hmm2['startprob']

    # Calculate transition matrix
    transmat[:M1-1, :M1-1] = hmm1['transmat'][:-1, :-1]
    transmat[:M1-1, M1-1:] = np.outer(hmm1['transmat'][:-1, -1], hmm2['startprob'])
    transmat[M1-1:, M1-1:] = hmm2['transmat']

    # Calculate means and covars
    means_concat = np.vstack((hmm1['means'], hmm2['means']))
    covars_concat = np.vstack((hmm1['covars'], hmm2['covars']))

    return {
        'name': f"{hmm1['name']}+{hmm2['name']}",
        'startprob': startprob,
        'transmat': transmat,
        'means': means_concat,
        'covars': covars_concat
    }



# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        log_alpha: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    log_alpha = np.zeros((N, M))
    
    # Initialization step
    log_alpha[0, :] = log_startprob[:-1] + log_emlik[0, :]
    
    # Recursion step
    for n in range(1, N):
        for j in range(M):
            log_alpha[n, j] = logsumexp(log_alpha[n-1, :] + log_transmat[:-1, j]) + log_emlik[n, j]

    return log_alpha

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        log_beta: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    log_beta = np.zeros((N, M))

    # Initialization step = 0

    # Recursion step, NB: reversed order
    for n in reversed(range(N - 1)):
        for i in range(M):
            log_beta[n, i] = logsumexp(log_transmat[i, :-1] + log_emlik[n + 1, :] + log_beta[n + 1, :])

    return log_beta

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    # From p.386, section 8.2.3, [gs:HuangEtAl2001Book]

    # Step 1: Initialization
    N, M  = log_emlik.shape
    viterbi_loglik = np.zeros((N, M))
    viterbi_b_mat = np.zeros((N, M), dtype=int)

    # Step 2: Induction
    viterbi_loglik[0] = log_startprob[:-1] + log_emlik[0]
    for n in range(1, N):
        for j in range(M):
            viterbi_loglik[n, j] = np.max(viterbi_loglik[n - 1, :] + log_transmat[:-1, j]) + log_emlik[n, j]
            viterbi_b_mat[n, j] = np.argmax(viterbi_loglik[n - 1, :] + log_transmat[:-1, j])

    # Step 3: Termination
    viterbi_path = [np.argmax(viterbi_b_mat[-1])]
    for n in reversed(range(N - 1)):
        viterbi_path.append(viterbi_b_mat[n, viterbi_path[-1]])
    
    # Step 4: Backtracking
    viterbi_path.reverse()
    return np.max(viterbi_loglik[-1]), np.array(viterbi_path)

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N, M = log_alpha.shape
    log_denom = logsumexp(log_alpha[N - 1])
    log_gamma = log_alpha + log_beta - log_denom

    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
        X: NxD array of feature vectors
        log_gamma: NxM state posterior probabilities in log domain
        varianceFloor: minimum allowed variance scalar
    where N is the length of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
        means: MxD mean vectors for each state
        covars: MxD covariance (variance) vectors for each state
    """

    # From https://mzaradzki.github.io/probabilistic-javascript/demos/hmm.html
    
    _, D = X.shape
    M = log_gamma.shape[1]
    gamma = np.exp(log_gamma)
    sums_gamma = np.sum(gamma, axis=0)
    
    # Compute weighted sum of data points ('X') with weights given by 'gamma'.
    # Dividing by sums_gamma normalizes to get mean for each state.
    means = np.dot(gamma.T, X) / sums_gamma[:, np.newaxis]

    # Compute the weighted sum of squared differences (diff) for each feature
    weighted_squares = np.zeros((M, D))
    weighted_squares = np.array([[np.sum(gamma[:, k] * (X[:, d] - means[k, d])**2) for d in range(D)] for k in range(M)])

    # Normalize by sums_gamma to get the variances
    covars = weighted_squares / sums_gamma[:, np.newaxis]
    
    # Apply variance floor
    covars[covars < varianceFloor] = varianceFloor
    
    return means, covars
