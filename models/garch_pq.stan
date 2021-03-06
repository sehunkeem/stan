data {
  int<lower=0> T;
  array[T] real r;
  real<lower=0> sigma1;
}
parameters {
  real mu;
  real<lower=0> alpha0;
  real<lower=0, upper=1> alpha1;
  real<lower=0, upper=(1-alpha1)> beta1;
}
transformed parameters {
  array[T] real<lower=0> sigma;
  sigma[1] = sigma1;
  for (t in 2:T) {
    sigma[t] = sqrt(alpha0
                     + alpha1 * pow(r[t - 1] - mu, 2)
                     + beta1 * pow(sigma[t - 1], 2));
  }
}
model {
  r ~ normal(mu, sigma);
}