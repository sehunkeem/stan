functions {
  real site_log_lambda(real t, real offset, real a) {
    if (t - offset <=0)
      return negative_infinity();
    else
      return a;
  }
}

data {
  int<lower=0> end_time;
  int<lower=0> total_number_of_sites;
  // vector<lower=0>[end_time] num_sites;
  int<lower=0> Y[end_time];
  vector<lower=0>[total_number_of_sites] site_offsets;
  vector<lower=0>[total_number_of_sites] site_closing_times;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  // real<lower=0> lambda;
  real alpha;
}

transformed parameters {
  vector<lower=0>[end_time] lambda_vector; {
    for (j in 1:end_time) {
      lambda_vector[j] = 0;
      for (i in 1:total_number_of_sites) {
        lambda_vector[j] += exp( site_log_lambda(j, site_offsets[i], alpha) );
      }
    }
  }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // lambda ~ gamma(1, 2);
  alpha ~ normal(0, 5);


  for (t in 1:end_time){
    // target += poisson_lpmf(Y[t] | full_lambda[t]);
    // print("Time = ", t);
    // print("Lambda = ", lambda);
    // print("Transformed lambda = ", lambda_vector[t]);
    Y[t] ~ poisson(lambda_vector[t]);
  }
}
