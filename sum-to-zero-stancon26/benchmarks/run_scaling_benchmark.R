#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
})

project_dir <- normalizePath(getwd())
model_paths <- c(
  conventional = file.path(
    project_dir,
    "stan",
    "varying_intercept_bayes_diff_vars.stan"
  ),
  s2z = file.path(
    project_dir,
    "stan",
    "varying_intercept_diff_vars.stan"
  )
)
if (!all(file.exists(model_paths))) {
  stop("Run this script from the sum-to-zero-stancon26 directory.")
}

result_dir <- file.path(project_dir, "benchmarks")
cmdstan_output_dir <- tempfile("s2z-scaling-cmdstan-")
dir.create(cmdstan_output_dir, recursive = TRUE)

settings <- list(
  J = c(10L, 20L, 50L, 100L),
  group_size_pattern = c(8L, 13L, 21L, 34L, 55L, 89L),
  alpha = 5,
  tau = 2.5,
  sigma_x = 1,
  simulation_seed = 260810L,
  sampler_seed = 260811L,
  chains = 4L,
  parallel_chains = 4L,
  iter_warmup = 1000L,
  iter_sampling = 5000L,
  adapt_delta = 0.8,
  max_treedepth = 15L
)

# Simulate one 100-group data set and use the first J groups in each fit.
# Cycling the original group sizes keeps their distribution approximately
# fixed while both model dimension and sample size increase with J.
set.seed(settings$simulation_seed)
J_max <- max(settings$J)
group_sizes <- rep(settings$group_size_pattern, length.out = J_max)
a_true <- rnorm(J_max, 0, settings$tau)
x_by_group <- lapply(seq_len(J_max), function(j) {
  rnorm(
    group_sizes[j],
    settings$alpha + a_true[j],
    settings$sigma_x
  )
})

make_data <- function(J) {
  sizes <- group_sizes[seq_len(J)]
  list(
    N = sum(sizes),
    J = J,
    group = rep(seq_len(J), sizes),
    x = unlist(x_by_group[seq_len(J)], use.names = FALSE)
  )
}

models <- lapply(model_paths, cmdstan_model)

summarize_fit <- function(fit, J, parameterization) {
  diagnostics <- fit$diagnostic_summary()
  sampler_diagnostics <- as_draws_matrix(fit$sampler_diagnostics())
  total_leapfrog <- sum(sampler_diagnostics[, "n_leapfrog__"])

  effect_variables <- if (parameterization == "conventional") {
    paste0("a_bayes[", seq_len(J), "]")
  } else {
    paste0("a_bayes_recovered[", seq_len(J), "]")
  }
  scale_variables <- paste0("a_prior_sd[", seq_len(J), "]")
  variables <- c(effect_variables, scale_variables)
  summary <- fit$summary(variables = variables)

  expected <- data.frame(
    variable = variables,
    family = rep(c("local_effect", "local_prior_sd"), each = J),
    group = rep(seq_len(J), 2L),
    stringsAsFactors = FALSE
  )
  parameter_result <- merge(
    expected,
    summary[c("variable", "ess_bulk", "rhat")],
    by = "variable",
    sort = FALSE
  )
  parameter_result <- parameter_result[
    match(variables, parameter_result$variable),
  ]
  parameter_result$J <- J
  parameter_result$parameterization <- parameterization
  parameter_result$total_postwarmup_leapfrog <- total_leapfrog
  parameter_result$ess_per_grad <-
    parameter_result$ess_bulk / total_leapfrog

  diagnostic_result <- data.frame(
    J = J,
    N = sum(group_sizes[seq_len(J)]),
    parameterization = parameterization,
    chains = settings$chains,
    iter_warmup = settings$iter_warmup,
    iter_sampling = settings$iter_sampling,
    postwarmup_draws = settings$chains * settings$iter_sampling,
    adapt_delta = settings$adapt_delta,
    max_treedepth = settings$max_treedepth,
    divergences = sum(diagnostics$num_divergent),
    max_treedepth_hits = sum(diagnostics$num_max_treedepth),
    min_ebfmi = min(diagnostics$ebfmi),
    median_ebfmi = median(diagnostics$ebfmi),
    total_postwarmup_leapfrog = total_leapfrog,
    mean_leapfrog_per_draw = total_leapfrog /
      (settings$chains * settings$iter_sampling),
    max_rhat_checked = max(parameter_result$rhat),
    min_bulk_ess_checked = min(parameter_result$ess_bulk),
    stringsAsFactors = FALSE
  )

  list(diagnostics = diagnostic_result, parameters = parameter_result)
}

diagnostic_rows <- list()
parameter_rows <- list()
fit_index <- 0L

for (J in settings$J) {
  stan_data <- make_data(J)
  for (parameterization in names(models)) {
    fit_index <- fit_index + 1L
    message(sprintf(
      "[%d/8] Sampling %s model at J=%d (N=%d)",
      fit_index,
      parameterization,
      J,
      stan_data$N
    ))
    fit_started <- Sys.time()
    fit <- models[[parameterization]]$sample(
      data = stan_data,
      seed = settings$sampler_seed + J,
      chains = settings$chains,
      parallel_chains = settings$parallel_chains,
      iter_warmup = settings$iter_warmup,
      iter_sampling = settings$iter_sampling,
      adapt_delta = settings$adapt_delta,
      max_treedepth = settings$max_treedepth,
      refresh = 500,
      output_dir = cmdstan_output_dir,
      output_basename = sprintf(
        "scaling_%s_J%03d",
        parameterization,
        J
      )
    )
    result <- summarize_fit(fit, J, parameterization)
    result$diagnostics$elapsed_seconds <- as.numeric(difftime(
      Sys.time(),
      fit_started,
      units = "secs"
    ))
    diagnostic_rows[[fit_index]] <- result$diagnostics
    parameter_rows[[fit_index]] <- result$parameters
  }
}

diagnostics <- do.call(rbind, diagnostic_rows)
parameters <- do.call(rbind, parameter_rows)

stopifnot(
  sum(diagnostics$divergences) == 0,
  sum(diagnostics$max_treedepth_hits) == 0,
  min(diagnostics$min_ebfmi) > 0.3,
  max(diagnostics$max_rhat_checked) < 1.01
)

quantile_type8 <- function(x, probability) {
  unname(quantile(x, probability, type = 8))
}

summary_keys <- unique(parameters[c(
  "J",
  "parameterization",
  "family"
)])
summary_rows <- lapply(seq_len(nrow(summary_keys)), function(i) {
  key <- summary_keys[i, ]
  keep <- parameters$J == key$J &
    parameters$parameterization == key$parameterization &
    parameters$family == key$family
  x <- parameters[keep, ]
  data.frame(
    J = key$J,
    parameterization = key$parameterization,
    family = key$family,
    n_parameters = nrow(x),
    ess_bulk_q25 = quantile_type8(x$ess_bulk, 0.25),
    ess_bulk_median = median(x$ess_bulk),
    ess_bulk_q75 = quantile_type8(x$ess_bulk, 0.75),
    ess_per_grad_q25 = quantile_type8(x$ess_per_grad, 0.25),
    ess_per_grad_median = median(x$ess_per_grad),
    ess_per_grad_q75 = quantile_type8(x$ess_per_grad, 0.75),
    stringsAsFactors = FALSE
  )
})
efficiency_summary <- do.call(rbind, summary_rows)
efficiency_summary <- efficiency_summary[order(
  efficiency_summary$family,
  efficiency_summary$J,
  efficiency_summary$parameterization
), ]

write.csv(
  diagnostics,
  file.path(result_dir, "scaling_diagnostics.csv"),
  row.names = FALSE
)
write.csv(
  efficiency_summary,
  file.path(result_dir, "scaling_efficiency.csv"),
  row.names = FALSE
)

message("\nFit diagnostics:")
print(diagnostics)
message("\nEfficiency summaries:")
print(efficiency_summary)
