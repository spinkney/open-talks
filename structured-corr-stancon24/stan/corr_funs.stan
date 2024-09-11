functions{
  // vector lb_ub(vector y, real lb, real ub) {
  // //  target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
  //   
  //   return lb + (ub - lb) * inv_logit(y);
  // }
  
    real lb_ub(real y, real lb, real ub) {
   // target += log(ub - lb) + log_inv_logit(y) + log1m_inv_logit(y);
    
    return lb + (ub - lb) * inv_logit(y);
  }
  matrix cholesky_corr_constrain (int K, vector raw,
   int N_blocks,
   array[ , ] int res_index, array[] int res_id, 
   real lb, real ub) {
   matrix[K, K] L = rep_matrix(0, K, K);
    int cnt = 1;
    int N_res = num_elements(res_id);
    vector[N_blocks] x_cache;
    array[N_blocks] int res_id_cnt = ones_int_array(N_blocks);
    int res_row = 1;
  
    L[1, 1] = 1;
    L[2, 1] = lb_ub(raw[cnt], lb, ub);
    L[2, 2] = sqrt(1 - L[2, 1]^2);
    cnt += 1;

    if (res_index[res_row, 1] == 2) {
      x_cache[res_id[res_row]] = L[2, 1];
      res_id_cnt[res_id[res_row]] += 1;
      res_row += 1;
    }
    
    for (i in 3:K) {
       if (res_index[res_row, 1] == i) {
          print("res_row[", i, ", ", 1, "] = ", res_row);
         if (res_id_cnt[res_id[res_row]] == 1) {
            L[i, 1] = lb_ub(raw[cnt], lb, ub);
           // print("cnt[", i, ", ", 1, "] = ", cnt);
            x_cache[res_id[res_row]] = L[i, 1];
            res_id_cnt[res_id[res_row]] += 1;
            cnt += 1;
        } else {
         L[i, 1] = x_cache[res_id[res_row]];
        }
        res_row += 1;
      } else {
       L[i, 1] = lb_ub(raw[cnt], lb, ub);
       // print("cnt[", i, ", ", 1, "] = ", cnt);
       cnt += 1;
      }
       
    L[i, 2] = sqrt(1 - L[i, 1]^2);
    real l_ij_old = L[i, 2];
      for (j in 2:i - 1) {
        real l_ij_old_x_l_jj = l_ij_old * L[j, j];
        real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
          
          // how to derive the bounds
          // we know that the correlation value C is bound by
          // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
          // Now we want our bounds to be enforced too so
          // max(lb, b1 - Ljj * Lij_old) <= C <= min(ub, b1 + Ljj * Lij_old)
          // We have the Lij_new = (C - b1) / Ljj
          // To get the bounds on Lij_new is
          // (bound - b1) / Ljj 
          
          real low = max({b1 - l_ij_old_x_l_jj, lb});
          real up = min({b1 + l_ij_old_x_l_jj, ub});
         // real x = lb_ub_lp(off_raw[cnt], low, up);
        //  L[i, j] = x / L[j, j]; 
          
      if (res_index[res_row, 1] == i && res_index[res_row, 2] == j) {
                   print("res_row[", i,  ", ",j, "] = ", res_row);
         if (res_id_cnt[res_id[res_row]] == 1) {
            x_cache[res_id[res_row]] = lb_ub(raw[cnt], low, up);
            L[i, j] = (x_cache[res_id[res_row]] - b1) / L[j, j];
         //   print("cnt[", i,  ", ",j, "] = ", cnt);
            res_id_cnt[res_id[res_row]] += 1;
            cnt += 1;
         } else {
            L[i, j] = (x_cache[res_id[res_row]] - b1) / L[j, j];
        }
        
        res_row = res_row == N_res ? N_res : res_row + 1;

      } else {
             //  print("cnt[", i,  ", ",j, "] = ", cnt);
        L[i, j] = (lb_ub(raw[cnt], low, up) - b1) / L[j, j];
        cnt += 1;
      }
          

        //  target += -log(L[j, j]);
          l_ij_old *= sqrt(1 - (L[i, j] / l_ij_old)^ 2);
        }
        L[i, i] = l_ij_old;
      }
        return L;
  }

}
data {
  int<lower=2> K; // dimension of correlation matrix
  real<lower=0> eta;
  real<lower=-1> lb;
  real<upper=1> ub;
  int<lower=0> res_params;
  int<lower=0> N_blocks;
  int<lower=0> N_res;
  array[N_res, 2] int res_index;
  array[N_res] int res_id;
}
transformed data {
  int k_choose_2 = (K * (K - 1)) %/% 2;
}
parameters {
  vector[k_choose_2 - res_params] raw;
}
transformed parameters {
}
model {
    raw ~ std_normal();
//  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
//    matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}