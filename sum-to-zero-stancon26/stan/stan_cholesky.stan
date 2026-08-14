data {
  int<lower=0> K;
}
transformed data {
 int K_choose_2 = choose(K, 2);
}
parameters {
  // y is a vector K-choose-2 unconstrained parameters
  vector[K_choose_2] y;
}
transformed parameters {
  // L is a Cholesky factor of a K x K correlation matrix
  cholesky_factor_corr[K] L = identity_matrix(K);
  real log_det_jacobian = 0;
  {
    int counter = 1;
    real sum_sqs;
    vector[K_choose_2] z = tanh(y);
    log_det_jacobian += sum(log1m(square(z)));
    
    for (i in 2 : K) {
      L[i, 1] = z[counter];
      counter += 1;
      sum_sqs = square(L[i, 1]);
      for (j in 2 : (i - 1)) {
        log_det_jacobian += 0.5 * log1m(sum_sqs);
        L[i, j] = z[counter] * sqrt(1 - sum_sqs);
        counter += 1;
        sum_sqs = sum_sqs + square(L[i, j]);
      }
      L[i, i] = sqrt(1 - sum_sqs);
    }
  }
}