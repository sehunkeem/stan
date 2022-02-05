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

  // Per-subject parameters (non-centered parameterization)
  matrix[4,J] country_gp_par_scaled_deviations ;   //non-centered std of length scale

  vector[eta_end[J]+J*n_tilde] eta; // std_normal() noise for GP

  real a_m;
  real<lower=0> a_s;
    
  real b_m;
  real<lower=0> b_s;
  
  real<lower=0> alpha_m;
  real<lower=0> rho_m;
  
  real<lower=0> alpha_s;
  real<lower=0> rho_s;
}

transformed parameters {
  // Non-centered parameterization of per-subject parameters
  vector[J] country_rho = exp(log(rho_m) + rho_s * country_gp_par_scaled_deviations[1]') ;   //length scale
  vector[J] country_alpha = exp(log(alpha_m) + alpha_s * country_gp_par_scaled_deviations[2]'); //signal standard deviation
  vector[J] country_a = a_m + a_s * country_gp_par_scaled_deviations[3]' ; // GP constant
  vector[J] country_b = b_m + b_s * country_gp_par_scaled_deviations[4]' ; // slope parameter

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
  target += inv_gamma_lpdf(rho_m | 2, 0.5); 
  target += normal_lpdf(alpha_m | 0, 4) + log(2);
  target += normal_lpdf(a_m | 0, 1) + log(2);
  target += normal_lpdf(b_m | 0, 1) + log(2);
    
  target += normal_lpdf(rho_s   | 0, 4) + log(2);
  target += normal_lpdf(alpha_s | 0, 4) + log(2);
  target += normal_lpdf(a_s | 0, 1) + log(2);
  target += normal_lpdf(b_s | 0, 1) + log(2);

  // Subject-level parameters drawn from pop-level distributions
  // (non-centered parameterizations)
  target += std_normal_lpdf(country_gp_par_scaled_deviations[1]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[2]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[3]) ;
  target += std_normal_lpdf(country_gp_par_scaled_deviations[4]) ;

  // GP noise as std_normal
  target += std_normal_lpdf(eta);


  // observations as poisson   
  for (j in 1:J) {
    target += poisson_lpmf(
      y[j,1:N[j]] | exp(country_a[j] + country_b[j] * to_vector(x[j,1:N[j]]) + log_f[j, n_tilde+1:n_tilde+N[j]]'));
  }
}
generated quantities{
  // so we can viz the country functions:
  matrix[J, n_tilde] a_plus_f ;
  for (j in 1:J) {
    a_plus_f[j,1:n_tilde] = exp(country_a[j] + country_b[j] * to_row_vector(x_tilde[j,1:n_tilde]) + log_f[j,1:n_tilde]);
  }

  int<lower=0> count = 0;
  vector[eta_end[J]] log_lik;
  for (j in 1:J) {
      for (n in 1:N[j]){
          count += 1;
          log_lik[count] = poisson_lpmf(y[j,n] | exp(country_a[j] + country_b[j] * x[j,n]+ 
                                                     log_f[j,n_tilde+n]));
      }
  }
}