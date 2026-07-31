data {
  int<lower=1> N;
  int<lower=2> J;
  array[N] int<lower=1, upper=J> group;
  vector[N] x;
  real<lower=0> alpha_prior_sd;
}
parameters {
  real alpha_s2z;
  sum_to_zero_vector[J] a_s2z;
  real<lower=0> tau;
  real<lower=0> sigma_x;
}
model {
  tau ~ normal(0, 5);
  sigma_x ~ normal(0, 2);
  alpha_s2z ~ normal(0, hypot(alpha_prior_sd, tau / sqrt(J)));
  a_s2z ~ normal(0, tau);
  // a_s2z lives in J - 1 subspace
  // but the normal divides by J instead of J-1
  // add back one log(tau)
  target += log(tau);
  x ~ normal(alpha_s2z + a_s2z[group], sigma_x);
}
generated quantities {
  real mean_a_bayes;
  real alpha_bayes;
  vector[J] a_bayes;
  {
    real var_mean_a = square(tau) / J;
    real var_alpha = square(alpha_prior_sd);
    real conditional_weight = var_mean_a / (var_alpha + var_mean_a);
    real conditional_sd = sqrt(
      var_alpha * var_mean_a / (var_alpha + var_mean_a)
    );

    mean_a_bayes = normal_rng(
      conditional_weight * alpha_s2z,
      conditional_sd
    );
  }
  alpha_bayes = alpha_s2z - mean_a_bayes;
  a_bayes = a_s2z + mean_a_bayes;
}
