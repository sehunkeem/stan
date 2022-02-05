functions {
  vector gp_f(int N,
                      real[] x,
                     real alpha, 
                     real rho,
                     vector eta) {
    vector[N] f;
    {
      matrix[N, N] K;
      matrix[N, N] L_K;
      K =  cov_exp_quad(x, alpha, rho)
                         + diag_matrix(rep_vector(1e-6, N));
      L_K = cholesky_decompose(K);
      f = L_K * eta;
    }
    return f;
  }
}

data {
  int<lower=1> J; // Number of countries 
  int<lower=1> maxN; // max number of observations per country
  array[J] int<lower=1> N ; // number of observations per country
  array[J,maxN] int<lower=0> y; // Poisson observations (dummy-filled ragged array)
  array[J,maxN] real x; // Input  (dummy-filled ragged array)
  int<lower=1> n_tilde;

}

transformed data {
  int<lower=1> eta_start[J] ;
  int<lower=1> eta_end[J] ;
  eta_start[1] = 1 ;
  eta_end[1] = N[1] ;
  for(j in 2:J){
    eta_start[j] = eta_end[j-1] +1 ;
    eta_end[j] = eta_start[j] -1 + N[j] ;
}
  array[J,n_tilde] real x_tilde;
  for (j in 1:J) {
    x_tilde[j,1:n_tilde] = linspaced_array(n_tilde, 1, max(x[j,1:N[j]])+10);
  }
}

parameters {
  vector[eta_end[J]+J*n_tilde] eta; // std_normal() noise for GP
  vector[J] country_a;
  real<lower=0> country_rho[J];   //length scale
  real<lower=0> country_alpha[J]; //signal standard deviation
}


transformed parameters {  
  matrix[J, maxN+n_tilde] log_f;
  for (j in 1:J) {
    log_f[j, 1:maxN+n_tilde] = rep_row_vector(0,maxN+n_tilde);
    log_f[j,1:n_tilde+N[j]] = gp_f(
        N[j]+n_tilde, 
        append_array(x_tilde[j,1:n_tilde], x[j,1:N[j]]),
        country_alpha[j],
        country_rho[j], 
        eta[eta_start[j]+(j-1)*n_tilde:eta_end[j]+j*n_tilde])';
}
}

model {
    
  // Priors (on population-level params)
  target += inv_gamma_lpdf(country_rho | 2, .5);
  target += normal_lpdf(country_alpha | 0, 1) + log(2);
  target += normal_lpdf(country_a | 0, 3) + log(2);

  target += normal_lpdf(eta | 0, 1);
    
  // observations as poisson   
  for (j in 1:J) {
    target += poisson_lpmf(
      y[j,1:N[j]] | exp(country_a[j] + log_f[j, n_tilde+1:n_tilde+N[j]]'));
  }                        
}

generated quantities {  
  // so we can viz the country functions:
  matrix[J, n_tilde] a_plus_f ;
  for (j in 1:J) {
    a_plus_f[j,1:n_tilde] = exp(country_a[j] + log_f[j,1:n_tilde]);
  }
    
  int<lower=0> count = 0;
  vector[eta_end[J]] log_lik;
  for (j in 1:J) {
      for (n in 1:N[j]){
          count += 1;
          log_lik[count] = poisson_lpmf(y[j,n] | exp(country_a[j] + log_f[j,n_tilde+n]));
      }
  }                    
}
