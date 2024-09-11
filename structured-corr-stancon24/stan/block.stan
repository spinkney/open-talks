functions{
  vector lb_ub_lp (vector y, real lb, real ub) {
    target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
    
    return lb + (ub - lb) * inv_logit(y);
  }
  
    real lb_ub_lp (real y, real lb, real ub) {
    target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
    
    return lb + (ub - lb) * inv_logit(y);
  }
  // matrix cholesky_corr_constrain_lp (vector col_one_raw, vector off_raw,
  //                                     matrix block_index, real lb, real ub) {
  //   int K = num_elements(col_one_raw) + 1;
  //   matrix[K, K] L = rep_matrix(0, K, K);
  //   L[1, 1] = 1;
  //   L[2, 1] = lb_ub_lp(col_one_raw[1], lb, ub);
  //   L[2, 2] = sqrt(1 - L[2, 1]^2);
  //   L[3:K, 1] = lb_ub_lp(col_one_raw[2:K - 1], lb,  ub);
  //   int block_cnt = 0;
  //   int cnt = 1;
  //   real L_cache;
  //   real x;
  // 
  //   for (i in 3:K) {
  //      real l_ij_old = log1m(L[i, 1]^2);
  //     for (j in 2:i - 1) {
  //       real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
  //       real stick_length = exp(0.5 * l_ij_old);
  //       real low = max({-stick_length, (lb - b1) / L[j, j]});
  //       real up = min({stick_length, (ub - b1) / L[j, j] });
  //       
  //      if (block_index[i, j] == 1) {
  //           if (block_cnt == 0) {
  //             L[i, j] = lb_ub_lp(off_raw[cnt], low, up) ;
  //             x =  L[i, j] * L[j, j] + b1;
  //             cnt += 1;
  //             block_cnt += 1;
  //           } else {
  //              L[i, j] =  (x - b1) / L[j, j] ;
  //              target += -log(L[j, j]);
  //           }
  //         } else {
  //           L[i, j] = lb_ub_lp(off_raw[cnt], low, up);
  //           cnt += 1;
  //         }
  // 
  //        // L[i, j] = lb_ub_lp(off_raw[cnt], low, up);
  //         l_ij_old = log_diff_exp(l_ij_old, 2 * log(abs(L[i, j])));
  //         //cnt += 1;
  //       }
  //        L[i, i] = exp(0.5 * l_ij_old);
  //     }
  //       return L;
  // }
  // 
  
  matrix cholesky_corr_constrain_lp(int K, vector raw,
   int N_blocks,
   array[ , ] int res_index, array[] int res_id, 
   vector lb, vector ub) {
   matrix[K, K] L = rep_matrix(0, K, K);
    int cnt = 1;
    int N_res = num_elements(res_id);
    vector[N_blocks] x_cache;
    array[N_blocks] int res_id_cnt = ones_int_array(N_blocks);
    int res_row = 1;
  
    L[1, 1] = 1;
    L[2, 1] = lb_ub_lp(raw[cnt], lb[cnt], ub[cnt]);
    L[2, 2] = sqrt(1 - L[2, 1]^2);
    cnt += 1;

    if (res_index[res_row, 1] == 2) {
      x_cache[res_id[res_row]] = L[2, 1];
      res_id_cnt[res_id[res_row]] += 1;
      res_row += 1;
    }
    
    for (i in 3:K) {
       if (res_index[res_row, 1] == i) {
         if (res_id_cnt[res_id[res_row]] == 1) {
            L[i, 1] = lb_ub_lp(raw[cnt], lb[cnt], ub[cnt]);
            x_cache[res_id[res_row]] = L[i, 1];
            res_id_cnt[res_id[res_row]] += 1;
            cnt += 1;
        } else {
         L[i, 1] = x_cache[res_id[res_row]];
        }
        res_row += 1;
      } else {
       L[i, 1] = lb_ub_lp(raw[cnt], lb[cnt], ub[cnt]);
       cnt += 1;
      }
       
    L[i, 2] = sqrt(1 - L[i, 1]^2);
   // real l_ij_old = L[i, 2];
    real l_ij_old = log1m(L[i, 1]^2);
      for (j in 2:i - 1) {
       // real l_ij_old_x_l_jj = l_ij_old * L[j, j];
        real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
       // real low = max({b1 - l_ij_old_x_l_jj, lb});
      //  real up = min({b1 + l_ij_old_x_l_jj, ub});
      real stick_length = exp(0.5 * l_ij_old);
        real low = max({-stick_length, (lb[cnt] - b1) / L[j, j]});
        real up = min({stick_length, (ub[cnt] - b1) /  L[j, j] });
          
      if (res_index[res_row, 1] == i && res_index[res_row, 2] == j) {
         if (res_id_cnt[res_id[res_row]] == 1) {
            L[i, j] = lb_ub_lp(raw[cnt], low, up);
           // x_cache[res_id[res_row]] = lb_ub_lp(raw[cnt], low, up);
          //  L[i, j] = (x_cache[res_id[res_row]] - b1) / L[j, j];
           
            x_cache[res_id[res_row]] = L[i, j] * L[j,j] + b1;
            res_id_cnt[res_id[res_row]] += 1;
            cnt += 1;
          //  target += -log(L[j, j]);
         } else {
            L[i, j] = (x_cache[res_id[res_row]] - b1) / L[j, j];
            target += -log(L[j, j]);
        }
        res_row = res_row == N_res ? N_res : res_row + 1;
      } else {
         L[i, j] = lb_ub_lp(raw[cnt], low, up);
       // L[i, j] = (lb_ub_lp(raw[cnt], low, up) - b1) / L[j, j];
        cnt += 1;
      //  target += -log(L[j, j]);
      }
      
       //   l_ij_old *= sqrt(1 - (L[i, j] / l_ij_old)^ 2);
             l_ij_old = log_diff_exp(l_ij_old, 2 * log(abs(L[i, j])));
      
        }
       // L[i, i] = l_ij_old;
        L[i, i] = exp(0.5 * l_ij_old);
      }
        return L;
  }
}
data {
  int<lower=2> K; // dimension of correlation matrix
  real<lower=0> eta;
  vector[(K * (K - 1)) %/% 2] lb;
  vector[(K * (K - 1)) %/% 2] ub;
  int<lower=0> free_params;
  int<lower=0> N_blocks;
  int<lower=0> N_res;
  array[N_res, 2] int res_index;
  array[N_res] int res_id;
}
transformed data {
  int k_choose_2 = (K * (K - 1)) %/% 2;
}
parameters {
  vector[free_params] raw;
}
transformed parameters {
  matrix[K, K] L_Omega = cholesky_corr_constrain_lp(K,
    raw,
  N_blocks, res_index, res_id,
  lb, ub);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
    matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}