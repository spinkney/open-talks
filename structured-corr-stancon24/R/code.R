library(cmdstanr)

mod_stan_positive_corr <- cmdstan_model("structured-corr-stancon24/stan/stan_positive_corr.stan")

mod_stan_positive_out <- mod_stan_positive_corr$sample(
  data = list(K = 5),
  parallel_chains = 4,
  iter_sampling = 10000
)

print(mod_stan_positive_out$summary("Omega"), n = 45)


mod_new_positive_corr <- cmdstan_model("structured-corr-stancon24/stan/new_positive_corr.stan")

mod_new_positive_out <- mod_new_positive_corr$sample(
  data = list(K = 5,
              eta = 4,
              lb = -1,
              ub = 1),
  parallel_chains = 4,
  init = 0.75,
  adapt_delta = 0.9,
  seed = 989839,
  iter_sampling = 10000
)

print(mod_new_positive_out$summary("Omega"), n = 45)


library(cmdstanr)

mod_blocks <- cmdstan_model("structured-corr-stancon24/stan/block_clean.stan")

mod_blocks$format()

K <- 6
res_mat <- matrix(
  c( NA, 0, 0, 0, 0, 0,
      1, NA, 0, 0, 0, 0,
      1, 1, NA, 0, 0, 0,
      1, 1, 0, NA, 0, 0,
      2, 3, 4, 4, NA, 0,
      3, 2, 4, 4, 0, NA), byrow = T, K, K)

ub <- c(rep(1/sqrt(2), 5), 1, rep(0.7, 8), 1)
lb <- -ub

N_blocks <- max(res_mat, na.rm = T)

which(res_mat == 1, arr.ind = TRUE)

blocks <- lapply(1:N_blocks, function(x) {
  y <- cbind(which(res_mat == x, arr.ind = TRUE), res_id = x)
})

block_mat <- do.call(rbind, blocks)
N_res <- nrow(block_mat)
block_dt <- data.table::data.table(block_mat)
block_dt <- block_dt[order(row, col), .(row, col, res_id)]

open_spots <- length(which(res_mat[lower.tri(res_mat)] == 0))
res_params <- ((K * (K - 1)) /2) - block_dt[, max(res_id)] - open_spots
res_index <- lapply(seq_len(nrow(block_dt)), function(i) t(as.matrix(block_dt[i, .(row, col)] )))
res_id <- unlist(block_dt[, res_id])

mod_block_out <- mod_blocks$sample(
  data = list(
    K = K, 
    eta = 4,
    lb = lb,
    ub = ub,
    free_params = (K * (K - 1)) / 2 - res_params,
    N_blocks = N_blocks,
    N_res = N_res,
    res_index = as.matrix(block_dt[, .(row, col)]),
    res_id = res_id
  ),
  parallel_chains = 4,
  iter_sampling = 100
)

matrix(mod_block_out$summary("Omega")$mean, K, K)

print(mod_block_out$summary("Omega"), n =36)

###
mod_funs <- cmdstan_model("structured-corr-stancon24/stan/both.stan")
mod_funs$expose_functions(global = TRUE)


L <- cholesky_corr_constrain(K = K, 
                        raw = runif(free_params, -2, 2), 
                        N_blocks = N_blocks, 
                        res_index = res_index, 
                        res_id = res_id, 
                        lb = -1, 
                        ub = 1) 
tcrossprod(L)


mod_both <- cmdstan_model("structured-corr-stancon24/stan/both_clean.stan")
mod_both$format()
K <- 6
new_res_mat <- matrix(
  c( NA, 0, 0, 0, 0, 0,
     0, NA, 0, 0, 0, 0,
     NA, NA, NA, 0, 0, 0,
     NA, NA, 0, NA, 0, 0,
     NA, NA,  NA, NA, NA, 0,
     NA, NA, NA, NA, NA, NA), byrow = T, K, K)

known_mat <- matrix(NA, K, K)
open_vals <- which(new_res_mat[lower.tri(new_res_mat)] == 0)
known_mat[lower.tri(known_mat)][open_vals] <- 0

known_index <- which(!is.na(known_mat), arr.ind = TRUE)
known_index_dt <- data.table::data.table(known_index)
known_index_dt <- known_index_dt[order(row, col), .(row, col)]


# ub <- c(rep(1/sqrt(2), 5), 1, rep(0.7, 8), 1)
# lb <- -ub

N_blocks <- max(new_res_mat, na.rm = T)
blocks <- lapply(1:N_blocks, function(x) {
  y <- cbind(which(new_res_mat == x, arr.ind = TRUE), res_id = x)
})

block_mat <- do.call(rbind, blocks)
N_res <- nrow(block_mat)
block_dt <- data.table::data.table(block_mat)
block_dt <- block_dt[order(row, col), .(row, col, res_id)]

open_spots <- length(which(new_res_mat[lower.tri(new_res_mat)] == 0))
res_params <- ((K * (K - 1)) /2) - open_spots - (nrow(block_dt) - block_dt[, max(res_id)])

res_index <- lapply(seq_len(nrow(block_dt)), function(i) t(as.matrix(block_dt[i, .(row, col)] )))
res_id <- unlist(block_dt[, res_id])

lb <- rep(-1, 15)
ub <- rep(1, 15)

 lb[3] <- -0.8
 ub[3] <- 0
 
 lb[7] <- 0.3
 ub[7] <- 0.7

mod_both_out <- mod_both$sample(
  data = list(
    K = K, 
    eta = 5,
    lb = lb,
    ub =  ub,
    free_params = choose(K, 2) - 2,
    N_blocks = N_blocks,
    N_res = N_res,
    res_index = as.matrix(block_dt[, .(row, col)]),
    res_id = res_id,
    N_known = nrow(known_index),
    known_index = as.matrix(known_index_dt),
    known_vals = rep(0, nrow(known_index))
  ),
  init = 0.75,
  parallel_chains = 4,
  adapt_delta = 0.9,
  iter_sampling = 1000,
  seed = 2903423
)
matrix(mod_both_out$summary("Omega")$mean, K, K)
Omega <- mod_both_out$draws("Omega") |>
  posterior::summarize_draws( .num_args = list(digits = 4, notation = "dec")) 
print(Omega,  n = 36)

raw <- runif((K * (K - 1)) / 2 - res_params - open_spots, -2, 2)

ub <- rep(0.4, 15)
lb <- -ub

L <- cholesky_corr_constrain(K = K, 
                             raw = raw, 
                             N_blocks = N_blocks, 
                             res_index =  lapply(seq_len(nrow(block_dt)), function(i) t(as.matrix(block_dt[i, .(row, col)] ))), 
                             res_id = res_id, 
                             known_index = lapply(seq_len(nrow(known_index_dt)), function(i) t(as.matrix(known_index_dt[i, .(row, col)] ))),
                             known_vals = rep(0, nrow(known_index)),
                             lb = lb, 
                             ub = ub) 
zapsmall(tcrossprod(L))
