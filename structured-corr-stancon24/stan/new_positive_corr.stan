functions{
  vector lb_ub_lp (vector y, real lb, real ub) {
    int N = num_elements(y);
    vector[N] x = log_inv_logit(y);
    target += -abs(y) - 2 * log1p_exp(-abs(y));
   // vector[N] x;
    
    // for (n in 1:N) {
    //    if (y[n] >= 0) {
    //      x[n] = -log1p_exp(-y[n]);
    //    } else {
    //      x[n] = y[n] - log1p_exp(y[n]);
    //    }
    //   // target += -abs(y[n]) - 2 * log1p_exp(-abs(y[n]));
    //   target += -log_sum_exp({y[n], -y[n], log2()});
    // }
  
   real length;
    
   if (lb == -1 && ub == 1) {
       length = log2();
    } else if (ub == 1) {
      length = log1m(lb);
    } else {
      length = log(ub - lb);
    }
    
    target += N * length;

    return lb + exp(length + x);
  }

    real lb_ub_lp (real y, real lb, real ub) {
     real x = log_inv_logit(y);
     target += -abs(y) - 2 * log1p_exp(-abs(y));
    
  //  real x = y >= 0 ? -log1p_exp(-y) : y - log1p_exp(y);
  //  target += -abs(y) - 2 * log1p_exp(-abs(y));
  //  target += -log_sum_exp({y, -y, log2()});

     real length;
 
   if (lb == -1 && ub == 1) {
       length = log2();
    } else if (ub == 1) {
      length = log1m(lb);
    } else {
      length = log(ub - lb);
    }
     target += length;
     return lb + exp(length + x);
  }

    matrix cholesky_corr_constrain_lp (vector col_one_raw, vector off_raw,
                                           real lb, real ub) {
    int K = num_elements(col_one_raw) + 1;
    matrix[K, K] L = identity_matrix(K);
    vector[K] D;
    D[1] = 1;
    int cnt = 1;
    L[2, 1] = lb_ub_lp(col_one_raw[1], lb, ub);
    D[2] = 1 - L[2, 1]^2;
    L[3:K, 1] = lb_ub_lp(col_one_raw[2:K - 1], lb, ub);

    for (i in 3:K) {
       D[i] = 1 - L[i, 1]^2; 
       L[i, 2] = 1 - L[i, 1]^2;
       real l_ij_old = L[i, 2];
      for (j in 2:i - 1) {
        real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);
        real low = max({-sqrt(l_ij_old) * D[j], lb - b1});
        real up = min({sqrt(l_ij_old) * D[j], ub - b1});
        real x = lb_ub_lp(off_raw[cnt], low, up);
         
        L[i, j] = x / D[j]; 

        target += -0.5 * log(D[j]);
        l_ij_old -= D[j] * L[i, j]^2;
        cnt += 1;
      }
        D[i] = l_ij_old;
      }
        return diag_post_multiply(L, sqrt(D));
  }
matrix cholesky_corr_constrain_lp (vector col_one_raw, vector off_raw,
                                           real lb, real ub) {
    int K = num_elements(col_one_raw) + 1;
    matrix[K, K] L = rep_matrix(0, K, K);
    L[1, 1] = 1;
    L[2, 1] = lb_ub_lp(col_one_raw[1], lb, ub);
    L[2, 2] = sqrt(1 - L[2, 1]^2);
    L[3:K, 1] = lb_ub_lp(col_one_raw[2:K - 1], lb,  ub);

    int cnt = 1;

    for (i in 3:K) {
       real l_ij_old = log1m(L[i, 1]^2);
      for (j in 2:i - 1) {
        real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
        real stick_length = exp(0.5 * l_ij_old);
        real low = max({-stick_length, (lb - b1) / L[j, j]});
        real up = min({stick_length, (ub - b1) /  L[j, j] });

          L[i, j] = lb_ub_lp(off_raw[cnt], low, up);
          l_ij_old = log_diff_exp(l_ij_old, 2 * log(abs(L[i, j])));
          cnt += 1;
        }
         L[i, i] = exp(0.5 * l_ij_old);
      }
        return L;
  }
}
data {
  int<lower=2> K; // dimension of correlation matrix
  real<lower=0> eta;
  real<lower=-1> lb;
  real<upper=1> ub;
}
transformed data {
  int k_choose_2 = (K * (K - 1)) %/% 2;
  int km1_choose_2 = ((K - 1) * (K - 2)) %/% 2;
}
parameters {
  vector[K - 1] col_one_raw;
  vector[km1_choose_2] off_raw;
}
transformed parameters {
  matrix[K, K] L_Omega = cholesky_corr_constrain_lp(col_one_raw, off_raw, lb, ub);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
    matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}