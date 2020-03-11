#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/**
 * Hidden 
 * 
 */
typedef struct hmm {
    int L;
    int D;
    double *A;
    double *O;
    double *A_start;
} hmm_t;

double hmm_A(hmm_t *hmm, int i, int j) {
    assert(i >= 0 && i < hmm->L);
    assert(j >= 0 && j < hmm->L);
    return hmm->A[i * hmm->L + j];
} 

double hmm_O(hmm_t *hmm, int i, int j) {
    assert(i >= 0 && i < hmm->L);
    assert(j >= 0 && j < hmm->D);
    return hmm->O[i * hmm->D + j];
} 

void hmm_init_A(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        for (int j = 0; j < hmm->L; j++) {
            hmm->A[i * hmm->L + j] = (double) rand() / RAND_MAX;
        }
    }

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

void hmm_normalize_A(hmm_t *hmm) {
   
}

void hmm_init_O(hmm_t *hmm) {
    for (int i = 0; i < hmm->L; i++) {
        for (int j = 0; j < hmm->D; j++) {
            hmm->O[i * hmm->D + j] = (double) rand() / RAND_MAX;
        }
    }

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

void draw_progress_bar(epoch, n_epoch) {
    double progress = (double) (epoch) / n_epoch;
    printf("\r[");
    for (int i = 0; i < 50; i++) {
        if (i < (int)(50 * progress)) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%%", progress * 100);
    fflush(stdout);
}

hmm_t* hmm_create(int L, int D, double *A, double *O) {
    hmm_t *hmm = malloc(sizeof(hmm_t));
    hmm->L = L;
    hmm->D = D;
    hmm->A = A;
    hmm->O = O;

    hmm_init_A(hmm);
    hmm_init_O(hmm);

    hmm->A_start = malloc(sizeof(double) * L);
    for (int i = 0; i < L; i++) {
        hmm->A_start[i] = 1.0 / L;
    }
    return hmm;
}

void hmm_forward(hmm_t* self, int M, int *x, double *alphas) {

    for (int curr = 0; curr < self->L; curr++) {
        alphas[1 * self->L + curr] = self->A_start[curr] * hmm_O(self, curr, x[0]);
    }

    for (int t = 1; t < M; t++) {
        double norm = 0.0;
        for (int curr = 0; curr < self->L; curr++) {
            double prob = 0.0;
            double prev_norm = 0.0;
            for (int prev = 0; prev < self->L; prev++) {
                prob += alphas[t * self->L + prev] * hmm_A(self, prev, curr) * hmm_O(self, curr, x[t]);
            }
            alphas[(t + 1) * self->L + curr] = prob;
            norm += prob;
        }
        
        for (int curr = 0; curr < self->L; curr++) {
            assert(norm > 0.0);
            alphas[(t + 1) * self->L + curr] /= norm;
        }
    }
}

void hmm_backward(hmm_t* self, int M, int *x, double *betas) {
    for (int curr = 0; curr < self->L; curr++) {
        betas[M * self->L + curr] = 1.0;
    }

    for (int t = M; t >= 1; t--) {
        double norm = 0.0;
        for (int curr = 0; curr < self->L; curr++) {
            double prob = 0.0;
            for (int nxt = 0; nxt < self->L; nxt++) {
                if (t == 1) {
                    prob += betas[(t) * self->L + nxt] * self->A_start[nxt] * hmm_O(self, nxt, x[t]);
                } else {
                    prob += betas[(t) * self->L + nxt] * hmm_A(self, curr, nxt) * hmm_O(self, nxt, x[t]);
                }
            }
            betas[(t - 1) * self->L + curr] = prob;
            norm += prob;
        }

        for (int curr = 0; curr < self->L; curr++) {
            betas[(t - 1) * self->L + curr] /= norm;
        }
    }
}

void hmm_unsupervised_learning(hmm_t *self, int N, int *Ms, int **Xs, int n_iters) {

    for (int iteration = 1; iteration <= n_iters; iteration++) {
        if (iteration % 10 == 0) {
            draw_progress_bar(iteration, n_iters);
        }
        
        double A_num[self->L][self->L];
        double O_num[self->L][self->D];
        double A_den[self->L];
        double O_den[self->L];
        memset(&A_num, 0, self->L * self->L * sizeof(double));
        memset(&A_den, 0, self->L * self->D * sizeof(double));
        memset(&O_num, 0, self->L * sizeof(double));
        memset(&O_den, 0, self->L * sizeof(double));

        for (int n = 0; n < N; n++) {
            int M = Ms[n];
            int *x = Xs[n];
            double alphas[M + 1][self->L];
            double betas[M + 1][self->L];
            memset(&alphas, 0, (M + 1) * self->L * sizeof(double));
            memset(&betas, 0, (M + 1) * self->L * sizeof(double));
            hmm_forward(self, M, x, (double*) &alphas);
            hmm_backward(self, M, x, (double*) &betas);
        
            /* Compute A_den */
            for (int t = 1; t < M + 1; t++) {
                double P_curr[self->L];
                double P_curr_norm = 0.0;
                memset(&P_curr, 0, self->L * sizeof(double));

                for (int curr = 0; curr < self->L; curr++) {
                    P_curr[curr] = alphas[t][curr] * betas[t][curr];
                    P_curr_norm += P_curr[curr];
                }

                for (int curr = 0; curr < self->L; curr++) {
                    P_curr[curr] /= P_curr_norm;
                }

                for (int curr = 0; curr < self->L; curr++) {
                    if (t != M) {
                        A_den[curr] += P_curr[curr];
                    }
                    O_den[curr] += P_curr[curr];
                    O_num[curr][x[t - 1]] += P_curr[curr];
                }
            }

            /* Compute A_num */
            for (int t = 1; t < M; t++) {
                double P_curr_nxt[self->L][self->L];
                double P_curr_nxt_norm = 0.0;
                memset(&P_curr_nxt, 0, self->L * self->L * sizeof(double));

                for (int curr = 0; curr < self->L; curr++) {
                    for (int nxt = 0; nxt < self->L; nxt++) {
                        P_curr_nxt[curr][nxt] = alphas[t][curr] *
                                                hmm_A(self, curr, nxt) *
                                                hmm_O(self, nxt, x[t]) *
                                                betas[t + 1][nxt];
                        P_curr_nxt_norm += P_curr_nxt[curr][nxt];
                    }
                }

                for (int curr = 0; curr < self->L; curr++) {
                    for (int nxt = 0; nxt < self->L; nxt++) {
                        P_curr_nxt[curr][nxt] /= P_curr_nxt_norm;
                        A_num[curr][nxt] += P_curr_nxt[curr][nxt];
                    }
                }
            }
        }

        /* Update A */
        for (int curr = 0; curr < self->L; curr++) {
            for (int nxt = 0; nxt < self->L; nxt++) {
                self->A[curr * self->L + nxt] = A_num[curr][nxt] / A_den[curr];
            }
        }

        /* Update O */
        for (int curr = 0; curr < self->L; curr++) {
            for (int nxt = 0; nxt < self->D; nxt++) {
                self->O[curr * self->D + nxt] = O_num[curr][nxt] / O_den[curr];
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

    double A[n_states][n_states];
    double O[n_states][n_unique];
    memset(&A, 0, n_states * n_states * sizeof(double));
    memset(&O, 0, n_states * n_unique * sizeof(double));

    srand(time(NULL));

    hmm_t *hmm = hmm_create(n_states, n_unique, (double *) A, (double *) O);
    hmm_unsupervised_learning(hmm, N, Ms, Xs, n_iters);

    FILE *A_out = fopen("./A.csv", "w+");
    assert(A_out != NULL);
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_states; j++) {
            if (j != 0) fprintf(A_out, ", ");
            fprintf(A_out, "%f", A[i][j]);
        }
        fprintf(A_out, "\n");        
    }
    fclose(A_out);

    FILE *O_out = fopen("./O.csv", "w+");
    assert(O_out != NULL);
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_unique; j++) {
            if (j != 0) fprintf(O_out, ", ");
            fprintf(O_out, "%f ", O[i][j]);
        }  
        fprintf(O_out, "\n");              
    }
    fclose(O_out);

    free(hmm->A_start);
    free(hmm);

    return 0;
}
