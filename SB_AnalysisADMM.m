% Stanislas Dumas
% Started on 10/04/21
% 4YP 20/21 academic year
% Oxford University
% Supervisor: Kostas Margellos 
% Running the Scenario Based analysis for the Game

%% Identifying support Constraints  COULD BE WRONG CHANGE DATA
% Error tolerance
th_sup = 10^(-5);
inp = [];
for i = 1:M
    idx1 = (i-1)*n + nx*(N+1)+1
    idx2 = (i-1)*n + nx*(N+1)+N*nu
    inp = [inp; s_k(idx1:idx2)];
end

sup_constr = [];

for i = 1:M
    for  index = 1:N*nu 
        if inp( ((i-1)*N*nu) + index) <= (min_input{i}(index)+th_sup)    
            sup_constr = [sup_constr;i, index, realisation_max(index)];
        elseif inp(((i-1)*N*nu) + index) >= (max_input{i}(index)-th_sup)
            sup_constr = [sup_constr;i, index, realisation_min(index)]; 
        end
    end
end

n_star = size(sup_constr,1);

%% Evaluating Risk  % % % MUST CHANGE; test with M = 1000 and M = 10000 
[eps, t_out] = epsilon(M*N*nu, M_sample, Bet);
eps_priori = eps(M*N*nu+1);
epsDegPriori = epsilonDegen(M*N*nu, M_sample, Bet);

% evaluating epsilon A -posteriori
[eps, t_out] = epsilon(n_star, M_sample, Bet);
eps_n_star = eps(n_star+1);
epsDeg = epsilonDegen(n_star, M_sample, Bet);

%% SB - Monte Carlo

realisation_MC = cell(N_MC,1);
V_count = 0;
No_Viol = 0;
V_index = [];

for i = 1:N_MC
    realisation_MC{i} = real_Norm(M, N, nu, mu, sig, lim_low, lim_up);
    max_MC = repmat(repmat(umax,N,1) + realisation_MC{i}, 3, 1); % Repeatingit 3 times for the 3 agents 
    min_MC = repmat(repmat(umin,N,1) + realisation_MC{i}, 3, 1);
    if inp >= min_MC & inp <= max_MC
        No_Viol = No_Viol+1;
    else    
       V_count = V_count+1; 
       V_index = [V_index; i];
    end
end
V_freq = V_count / N_MC;

%% Functions
% From Garrati and Campi paper 
function [eps, t_out] = epsilon(d,N,bet)
    eps = zeros(d+1,1);
    t_out = zeros(d+1,1);
    for k = 0:d
        m = [k:1:N];
        aux1 = sum(triu(log(ones(N-k+1,1)*m),1),2);
        aux2 = sum(triu(log(ones(N-k+1,1)*(m-k)),1),2);
        coeffs = aux2-aux1;
        t1 = 0;
        t2 = 1;
        while t2-t1 > 1e-10
            t = (t1+t2)/2;
            val = 1 - bet/(N+1)*sum( exp(coeffs-(N-m')*log(t)) );
            if val >= 0
                t2 = t;
            else
                t1 = t;
            end
        end
      eps(k+1) = 1-t1;
      t_out(k+1) = t1;
    end

end

function BetFixed = BetaFixed(d, M_sample, eps) 

     i = sym('i');    % Needs to be declared this way in nested function 
    Beta = symsum(nchoosek(M_sample, i) * eps^(i) * (1-eps)^(M_sample-i), i, 0, d-1);
    BetFixed = vpa(Beta);
end

function eps = epsilonDegen(k, N, bet)
    eps = 1 - ((bet/N) * 1 / (nchoosek(N,k))) ^ (1/(N-k));
end

function realisation = real_Norm(M, N, nu, mu, sig, lower, upper)    
    DistUntrunc = makedist('Normal', mu, sig);
    Dist = truncate(DistUntrunc,lower,upper);
    realisation = random(Dist, N*nu,1);
end