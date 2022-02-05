functions {
  real site_lambda(real t, real offset, real a, real b) {
    if (t - offset <=0)
      return negative_infinity();
    else
      return a + b * log(t - offset);
  }
}

data {
  int<lower=0> end_time;
  int<lower=0> total_number_of_sites;
  // vector<lower=0>[end_time] num_sites;
  int<lower=0> Y[end_time];
  vector<lower=0>[total_number_of_sites] site_offsets;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  // real<lower=0> lambda;
  real alpha;
  real beta;
}

transformed parameters {
  vector<lower=0>[end_time] lambda_vector; {
    for (j in 1:end_time) {
      lambda_vector[j] = 0;
      for (i in 1:total_number_of_sites) {
        lambda_vector[j] += exp( site_lambda(j, site_offsets[i], alpha, beta) );
      }
    }
  }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // lambda ~ gamma(1, 2);
  alpha ~ normal(1, 0.1);
  beta ~ normal(-0.2, 0.1);

  for (t in 1:end_time){
    // target += poisson_lpmf(Y[t] | full_lambda[t]);
    // print("Time = ", t);
    // print("Lambda = ", lambda);
    // print("Transformed lambda = ", lambda_vector[t]);
    Y[t] ~ poisson(lambda_vector[t]);
  }
}
