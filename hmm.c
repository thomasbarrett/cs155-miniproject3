#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

typedef struct hmm {
    int L;
    int D;
    double *A;
    double *O;
    double *A_start;
} hmm_t;

int min(int a, int b) {
    return a < b? a: b;
}

double hmm_A(hmm_t *hmm, int i, int j) {
    assert(i >= 0 && i < hmm->L);
    assert(j >= 0 && j < hmm->L);
    return hmm->A[i * hmm->L + j];
} 

double hmm_O(hmm_t *hmm, int i, int j) {
    assert(i >= 0 && i < hmm->L);
    assert(j >= 0 && j < hmm->D);
    return hmm->A[i * hmm->D + j];
} 

void draw_progress_bar(int epoch, int max_epoch) {
    int length = 50;
    double percent =  (double) epoch / (max_epoch - 1);
    printf("\r[");
    for (int i = 0; i < length; i++) {
        if (i < (int) (length * percent)) {
            printf("=");
        } else {
            printf(" ");
        }  
    }
    printf("] %.2f%%", percent * 100);
    fflush(stdout);
}

void hmm_init_A(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        for (int j = 0; j < hmm->L; j++) {
            hmm->A[i * hmm->L + j] = (double) rand() / RAND_MAX;
        }
        printf("\n");
    }
}

void hmm_normalize_A(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        double acc = 0.0;
        for (int j = 0; j < hmm->L; j++) {
            acc += hmm->A[i * hmm->L + j];
        }
        for (int j = 0; j < hmm->L; j++) {
            hmm->A[i * hmm->L + j] /= acc;
        }
    }
}

void hmm_init_O(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        for (int j = 0; j < hmm->D; j++) {
            hmm->O[i * hmm->D + j] = (double) rand() / RAND_MAX;
        }
    }
}

void hmm_normalize_O(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        double acc = 0.0;
        for (int j = 0; j < hmm->D; j++) {
            acc += hmm->O[i * hmm->D + j];
        }
        for (int j = 0; j < hmm->D; j++) {
            hmm->O[i * hmm->D + j] /= acc;
        }
    }
}

void hmm_printA(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        for (int j = 0; j < hmm->L; j++) {
            printf("%.4f ", hmm->A[i * hmm->L + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void hmm_printO(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        for (int j = 0; j < hmm->D; j++) {
            printf("%.4f ", hmm->O[i * hmm->D + j]);
        }
        printf("\n");
    }
}

hmm_t* hmm_create(int L, int D, double *A, double *O) {
    hmm_t *hmm = malloc(sizeof(hmm_t));
    hmm->L = L;
    hmm->D = D;
    hmm->A = A;
    hmm->O = O;

    hmm_init_A(hmm);
    hmm_normalize_A(hmm);

    hmm_init_O(hmm);
    hmm_normalize_O(hmm);

    hmm->A_start = malloc(sizeof(double) * L);
    for (int i = 0; i < L; i++) {
        hmm->A_start[i] = 1.0 / L;
    }
    return hmm;
}

/**
 * Uses the Viterbi algorithm to find the max probability state sequence
 * corresponding to a given input sequence.
 * 
 * \param M: length of x
 * \param x: Input sequence in the form of a list of length M, consisting
 *           of integers ranging from 0 to D - 1.
 * 
 * \return state sequence corresponding to x with the highest probability.
 */
void hmm_viterbi(hmm_t* self, int M, int *x, int *res) {
    double probs[M][self->L];
    int seqs[M][self->L];
    
    /* zero-initialize local variable length arrays */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < self->L; j++) {
            probs[i][j] = 0.0;
            seqs[i][j] = 0;
        }
    }

    for (int i = 0; i < self->L; i++) {
        probs[0][i] = self->A_start[i] * self->O[i * self->D + x[0]];
    }


    for (int j = 1; j < M; j++) {
        for (int i = 0; i < self->L; i++) {
            int argmax = -1;
            double max = 0.0;
            for (int k = 0; k < self->L; k++) {

                double val = probs[j - 1][k] *
                            hmm_A(self, k, i) *
                            hmm_O(self, i, x[j]);

                if (argmax == -1 || val > max) {
                    argmax = k;
                    max = val;
                }
            }
            probs[j][i] = max;
            seqs[j][i] = argmax;
        }
    }

    /* find the index of the largest element in seqs[M - 1] */
    int argmax = -1;
    double max = 0.0;
    for (int i = 0; i < self->L; i++) {
        if (argmax == -1 || probs[M - 1][i] > max) {
            argmax = i;
            max = probs[M - 1][i];
        }
    }

    int z = argmax;
    res[0] = z;
    for (int i = 1; i < M; i++) {
        z = seqs[i][z];
        res[i] = z;
    }
}

/**
 * \param self
 * \param M
 * \param x: int[M]
 * \param alphas: double[M + 1][self->L]
 */
void hmm_forward(hmm_t* self, int M, int *x, double *alphas) {

    /* zero-initialize local variable length arrays */
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < self->L; j++) {
            alphas[i * self->L + j] = 0.0;
        }
    }

    for (int i = 0; i < self->L; i++) {
        alphas[1 * self->L + i] = self->A_start[i] * self->O[i * self->D + x[0]];
    }

    for (int i = 1; i < M; i++) {
        double acc = 0.0;
        for (int z = 0; z < self->L; z++) {
            for (int j = 0; j < self->L; j++) {
                double v = alphas[i * self->L  + j] * 
                            hmm_A(self, j, z) *
                            hmm_O(self, z, x[i]);

                alphas[(i + 1) * self->L + z] += v;
                acc += v;
            }
        }

        for (int z = 0; z < self->L; z++) {
            alphas[(i + 1) * self->L + z] /= acc;
        }
    }
}


/**
 * \param self
 * \param M
 * \param x: int[M]
 * \param alphas: double[M + 1][self->L]
 */
void hmm_backward(hmm_t* self, int M, int *x, double *betas) {
    double alphas[M + 1][self->L];

    /* zero-initialize local variable length arrays */
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < self->L; j++) {
            betas[M * self->L + j] = 0.0;
        }
    }

    for (int i = 0; i < self->L; i++) {
        betas[M * self->L + i] = 1.0;
    }

    for (int i = M - 1; i >= 0; i--) {
        double acc = 0.0;
        for (int z = 0; z < self->L; z++) {
            for (int j = 0; j < self->L; j++) {
                double v = betas[(i + 1) * self->L + j] *
                            hmm_A(self, z, j) *
                            hmm_O(self, j, x[i]);
                betas[i * self->L + z] += v;
                acc += v;
            }
        }

        for (int z = 0; z < self->L; z++) {
            betas[i * self->L + z] /= acc;
        }
    }
}

void hmm_unsupervised_learning(hmm_t *self, int N, int *Ms, int **Xs, int n_iters) {
    for (int epoch = 0; epoch < n_iters; epoch++) {
        // draw_progress_bar(epoch, n_iters);
        
        double A_num[self->L][self->L];
        double A_den[self->L][self->L];
        double O_num[self->L][self->D];
        double O_den[self->L][self->D];
        memset(&A_num, 0, self->L * self->L * sizeof(double));
        memset(&A_den, 0, self->L * self->L *  sizeof(double));
        memset(&O_num, 0, self->L * self->D *  sizeof(double));
        memset(&O_den, 0, self->L * self->D *  sizeof(double));

        for (int n = 0; n < 3; n++) {

            int *x = Xs[n];
            int M = Ms[n];
            if (M < 2) continue;
            hmm_printA(self);
            hmm_printO(self);

            double alphas[M + 1][self->L];
            double betas[M + 1][self->L];
            memset(&alphas, 0, (M + 1) * self->L * sizeof(double));
            memset(&betas, 0, (M + 1) * self->L * sizeof(double));
            hmm_forward(self, M, x, (double*) alphas);
            hmm_backward(self, M, x, (double*) betas);

            double prob_ones[M][self->L];
            double prob_twos[M][self->L][self->L];
            memset(&prob_ones, 0, M * self->L * sizeof(double));
            memset(&prob_twos, 0, M * self->L * self->L * sizeof(double));

            for (int i = 1; i < M + 1; i++) {
                for (int z = 0; z < self->L; z++) {
                    double n = alphas[i][z] * betas[i][z];
                    double d = 0.0;
                    for (int j = 0; j < self->L; j++) {
                        d += alphas[i][j] * betas[i][j];
                    }
                    prob_ones[i - 1][z] =  n / d;
                }
            }
            
            for (int i = 1; i < M; i++) {
                for (int a = 0; a < self->L; a++) {
                    for (int b = 0; b < self->L; b++) {

                        double n = alphas[i][a] *
                                    hmm_A(self, a, b) *
                                    hmm_O(self, b, x[i]) *
                                    betas[i+1][b];

                        double d = 0.0;
                        for (int a_ = 0; a_ < self->L; a_++) {
                            for (int b_ = 0; b_ < self->L; b_++) {
                                d += alphas[i][a_] *
                                     hmm_A(self, a_, b_) *
                                     hmm_O(self, b_, x[i]) *
                                     betas[i+1][b_];
                            }
                        }

                        prob_twos[i - 1][a][b] = n / d;
                    }
                }
            }

            for (int a = 0; a < self->L; a++) { 
                for (int b = 0; b < self->L; b++) {
                    for (int i = 0; i < M; i++) {
                        A_num[a][b] += prob_twos[i][a][b];
                        A_den[a][b] += prob_ones[i][a];  
                    }
                }
            }

            for (int z = 0; z < self->L; z++) {
                for (int w = 0; w < self->D; w++) {
                    for (int i = 0; i < M; i++) {
                        if (x[i] == w) {
                            O_num[z][w] += prob_ones[i][z];
                        }
                        O_den[z][w] += prob_ones[i][z];
                    }
                }
            }

            for (int i = 0; i < self->L; i++) {
                for (int j = 0; j < self->L; j++) {
                    self->A[i * self->L + j] = A_num[i][j] / A_den[i][j];
                }
            }     

            for (int i = 0; i < self->L; i++) {
                for (int j = 0; j < self->D; j++) {
                    self->O[i * self->D + j] = O_num[i][j] / O_den[i][j];
                }
            }    
        }
    }

    printf("\n");
}

/*
 * usage: n_states n_epochs n_unique n_train [M xi0 xi1 xi2 ... xiM] ...-
 */
int main(int argc, char const *argv[]) {
    int n_states = atoi(argv[1]);
    int n_iters = atoi(argv[2]);
    int n_unique = atoi(argv[3]);
    int N = atoi(argv[4]);
    int Ms[N];
    int *Xs[N];
    
    int argi = 5;
    for (int n = 0; n < N; n++) {
        Ms[n] = atoi(argv[argi++]);
        Xs[n] = calloc(Ms[n], sizeof(int));
        for (int i = 0; i < Ms[n]; i++) {
            Xs[n][i] = atoi(argv[argi++]);
        }
    }

    double *A = (double *) calloc(n_states * n_states, sizeof(double));
    double *O = (double *) calloc(n_states * n_unique, sizeof(double));

    srand(2020);

    hmm_t *hmm = hmm_create(n_states, n_unique, A, O);
    hmm_unsupervised_learning(hmm, N, Ms, Xs, n_iters);

    printf("A\n");
    hmm_printA(hmm);

    printf("O\n");
    hmm_printO(hmm);

    
    free(A);
    free(O);
    free(hmm->A_start);
    free(hmm);

    return 0;
}
