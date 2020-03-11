########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
import sys

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = np.array([1. / self.L for _ in range(self.L)])


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.
        probs = np.zeros((M, self.L))
        seqs =  np.zeros((M, self.L), dtype=int)

        for i in range(self.L):
            probs[0, i] = self.A_start[i] * self.O[i, x[0]]
            seqs[0, i] = 0
  

        for j in range(1, M):
            for i in range(self.L):
                v = probs[j - 1, :] * self.A[:, i] * self.O[i, x[j]]
                probs[j, i] = np.amax(v)
                seqs[j, i] = np.argmax(v)

        max_seq = ''
        
        z = np.argmax(probs[M - 1])
        max_seq += str(z)

        for j in reversed(range(1, M)):
            z = seqs[j, z]
            max_seq += str(z)

        return max_seq[::-1]


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = np.zeros((M + 1, self.L))

        for i in range(self.L):
            alphas[1, i] = self.A_start[i] * self.O[i, x[0]]  

        for i in range(1, M):
            acc = 0.0
            for z in range(self.L):
                for j in range(self.L):
                    q = alphas[i, j] * self.A[j, z] * self.O[z, x[i]]
                    alphas[i + 1, z] += q
                    acc += q
                    
            if normalize:
                alphas[i + 1] /= acc
        
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = np.zeros((M + 1, self.L))

        for i in range(self.L):
            betas[M, i] = 1

        for i in reversed(range(0, M)):
            acc = 0.0
            for z in range(self.L):
                for j in range(self.L):
                    q = betas[i + 1, j] * self.A[z][j] * self.O[j][x[i]] 
                    betas[i, z] += q
                    acc += q;
            if normalize:
                betas[i] /= acc
        
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        for a in range(self.L):
            for b in range(self.L):
                num = 0.0
                den = 0.0
                for j in range(len(Y)):
                    for i in range(len(Y[j]) - 1):
                        if Y[j][i + 1] == b and Y[j][i] == a:
                            num += 1.0
                        if Y[j][i] == a:
                            den += 1.0
            
                self.A[a][b] = num / den

          

        # Calculate each element of O using the M-step formulas.
        
        for z in range(self.L):
            for w in range(self.D):
                num = 0.0
                den = 0.0
                for j in range(len(Y)):
                    for i in range(len(Y[j])):
                        if X[j][i] == w and Y[j][i] == z:
                            num += 1.0
                        if Y[j][i] == z:
                            den += 1.0
            
                self.O[z][w] = num / den


    def drawProgressBar(self, epoch, max_epoch, percent, barLen = 20):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("epoch: %d/%d [%s] %.2f%%" % (epoch, max_epoch,progress, percent * 100))
        sys.stdout.flush()
    
    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # Calculate each element of A using the M-step formulas.
        for epoch in range(N_iters):
                        
            A_num = np.zeros((self.L, self.L))
            A_den = np.zeros((self.L, self.L))
            O_num = np.zeros((self.L, self.D))
            O_den = np.zeros((self.L, self.D))
            
           
            for x, i in zip(X, range(len(X))):
                self.drawProgressBar(epoch, N_iters, i / len(X), 50)

                alpha = self.forward(x, normalize=True)
                beta = self.backward(x, normalize=True)     

               
                prob_ones = np.zeros((len(x), self.L))
                prob_twos = np.zeros((len(x), self.L, self.L))

                def prob_one(i, z):
                    n = alpha[i][z] * beta[i][z]
                    d = np.sum(alpha[i] * beta[i])
                    return n / d

                for i in range(1, len(x) + 1):
                    for z in range(self.L):
                        prob_ones[i - 1][z] = prob_one(i, z)
                    
                def prob_two_num(i, a, b):
                    return alpha[i][a] * self.A[a][b] * self.O[b][x[i]] * beta[i+1][b]

                def prob_two_den(i):
                    acc = 0.0
                    for a in range(self.L):
                        for b in range(self.L):
                            acc += prob_two_num(i, a, b)
                    return acc

                def prob_two(k, a, b):
                    return prob_two_num(i, a, b) / prob_two_den(i)

                for i in range(1, len(x)):
                    for a in range(self.L):
                        for b in range(self.L):
                            prob_twos[i - 1][a][b] = prob_two_num(i, a, b) / prob_two_den(i)
                        
                for a in range(self.L):
                    for b in range(self.L):
                        for i in range(len(x) - 1):
                            A_num[a][b] += prob_twos[i, a, b]
                            A_den[a][b] += prob_ones[i, a]  

                # Calculate each element of O using the M-step formulas.
                for z in range(self.L):
                    for w in range(self.D):
                        for i in range(len(x)):
                            if x[i] == w:
                                O_num[z][w] += prob_ones[i, z]
                            O_den[z][w] += prob_ones[i, z]

            self.A = A_num / A_den
            self.O = O_num / O_den
            
    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        emission = []
        states = []
        weight = self.A_start
        for i in range(M + 1):
            last = np.random.choice(self.L, p=weight)
            weight2 = self.O[last]
            out = np.random.choice(self.D, p=weight2)
            states.append(last)
            emission.append(out)
            weight = self.A[last]
        return emission[1:], states[1:]


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
