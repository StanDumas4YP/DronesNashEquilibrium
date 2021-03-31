% Stanislas Dumas
% Started on 31/03/21
% 4YP 20/21 academic year
% Oxford University
% Supervisor: Kostas Margellos 
% Nash Equilibrium drone game


%% Pre-Defined Model CHECKED 
T = 0.1;

Ad = [1 0 0.09629 0 0 0.03962;
    0 1 0 0.09629 -0.03962 0;
    0 0 0.8943 0 0 0.7027;
    0 0 0 0.8943 -0.7027 0;
    0 0 0 0.1932 0.4524 0;
    0 0 -0.1932 0 0 0.4524];

Bd = [0.003709 0; 0 0.003709;0.1057 0;0 0.1057;0 -0.1932;0.1932 0];

%% Variables 
% States
M = 3;           % No of Agents
nx = size(Ad, 2);
nu = size(Bd, 2);
nz = nu;         % number of cost variables for an agent
xypos = 2;       % x and y are in positions 1 and 2, THAT MUST STAY TRUE

% Collision avoidance
delta = 0.3;
deltaw = 0.31; 

% Prediction
N = 25;                     % Prediction horizon 
n = (N+1)*nx + N*nu + N*nz;
eta = 0.01;                 % Added to leading diagonal of Quadrti cost 

% Initial and reference states
x0 = [-delta 0 0 0 0 0; delta 0 0 0 0 0; 0 sqrt(3)*delta 0 0 0 0]';
r = [2*delta -delta/2 0 0 0 0; -2*delta delta/2 0 0 0 0; 0 -delta 0 0 0 0]';

% % % Define the vector that goes into the individual cost matrices
Q_vec = [4, 4, 0, 0, 0, 0;          % Weighting of deviation from reference; dimension nx
         4, 4, 0, 0, 0, 0;
         4, 4, 0, 0, 0, 0];
S_vec = [1,1;                       % Weighting of use of inputs ; dimension nz
         1, 1;
         1, 1];

%% Useful Matrices 
% Sigma multiplier
sig_mul = 1/M * kron(ones(1,M), eye(n));

% Creating the blocks for matrix V
H = [eye(xypos), zeros(xypos,nx-xypos)];
Blocks = {};                % Initialising the variable
for i =1:M
   
    Blocks{i} = [];          % Initalising the variable
    
    for k = 2 : (N+1)
       % For each time step, part of a block is: 
       % [zeros(previous agents) zeros(previous k) H zeros(remaining k) zeros(inputs) zeros(later agents)]
       B_temp = [zeros(xypos, (i-1)*n), zeros(xypos, (k-1)*nx), H, zeros(xypos, (N+1-k)*nx),zeros(xypos,nu * N), zeros(xypos,nz * N), zeros(xypos, (M-i)*n)];   
       Blocks{i} = [Blocks{i};B_temp];
   end
end

% Assembling V
V = [];
for i = 1:M
    V_i = [];
    j_vec = 1:M;                  % Vector of indices j for j=/= i 
    j_vec(i) = [];                % removing i 
    for j = 1:(M-1)
        V_ij = Blocks{i} - Blocks{j_vec(j)};
        V_i = [V_i ; V_ij];
    end
    V = [V;V_i];
end 

%% Setting up the cost function 

Q_cost = cell(1,M);
C_cost = cell(1,M);

for i = 1:M
   % Q_aug is never defined explicitely, but it can be seen in the definition for Q_cost (same for S_aug)
   Qi = diag(Q_vec(i,:));
   Si = diag(S_vec(i,:));
   Q_cost{i} = sparse(2 * blkdiag(kron(eye(N+1),Qi), zeros(N*nu), zeros(N*nz)));
   C_cost{i} = sparse(blkdiag(zeros((N+1)*nx), zeros(N*nz), kron(eye(N),Si)));
end

c_i = {};  
c_aug = [];
for i = 1:M
    Qi = diag(Q_vec(i,:));
    c_i{i} = [repmat(-2*Qi*r(:,i),N+1,1); zeros(N*nu,1); zeros(N*nz,1)];
    c_aug = [c_aug; c_i{i}];
end

% Making Q positive defintie
Q_add = diag(repmat(eta,n,1));
for i =1:M
    Q_cost{i} = sparse(Q_cost{i} + Q_add);
end

%% Setting up individual constraints 
% written in an OSQP format
% For a single drone
% Linear dynamics 
% Note: this assumes all drones have the same dynamics
Ax = kron(speye(N+1), -speye(nx)) + kron(sparse(diag(ones(N, 1), -1)), Ad);                            
Bu = kron([sparse(1, N); speye(N)], Bd);
Aeq = [Ax, Bu, zeros((N+1)*nx, N*nz)];

% Initial constraint 

leq_i = {};
for i = 1:M    
    leq = [-x0(:,i); zeros(N*nx, 1)];                % Column with dimension (N+1)nx x 1
    leq_i{i} = leq;
end
ueq_i = leq_i;                                       % It is an equality constraint 

% Input constraint

Ainputs = [zeros(N*nu,(N+1)*nx),eye(N*nu),zeros(N*nu,N*nz)];          % Matrix to apply this to inputs only
umin = ones(nu,1)*-1;                                % input limits (-1 and 1)
umax = ones(nu,1)*1;
min_input = repmat(umin,N,1);
max_input = repmat(umax,N,1);

% Z (cost function variable) constraints; The constraint is separated in two parts 
A_z = [zeros(2*N*nu, (N+1)*nx), [eye(N*nu); -1*eye(N*nu)], [-1*eye(N*nz); -1*eye(N*nz)]];
l_z = [-inf * ones(N*nu,1); -inf * ones(N*nu,1)];
u_z = [zeros(N*nu,1); zeros(N*nu,1)]; 

% OSQP constraints
A_i ={};
l_i = {};
u_i = {};

for i = 1:M 
    A_i{i} = [Aeq; Ainputs; A_z];
    l_i{i} = [leq_i{i}; min_input; l_z];
    u_i{i} = [ueq_i{i}; max_input; u_z];
end

% Augmenting individual constraints for M drones
A_aug = [];             % Initialising variables
l_aug = [];
u_aug = [];

for i = 1:M
    A_aug = blkdiag(A_aug, A_i{i});
    l_aug = [l_aug; l_i{i}];
    u_aug = [u_aug; u_i{i}];
end

% Coupling constraints defined in loop

%% Defining the additional parameters 

[F_Nm, F_Np] = F_Nash(M, Q_cost, C_cost, c_aug);

F_Nmf = full(F_Nm);
F_Nmsvd= svd(F_Nmf);          
alpha = min(F_Nmsvd);

% Determining L_F
L_F = norm(F_Nmf,2);

% Definition of tau moved to the algo

%% Setup and intialisation                 
% Initalising z(0)  
 z_0 = zeros(n,1) ;                 % Placeholder for intiialisation problem 

% Setting up the projection problem 
% Running the initialisation problem 
init = {};                  % The initialisation problem (single use)
proj = {};                  % The projection problem (For the algorithm)
P_proj = 2 * eye(n);        % Factor 2 essential due to OSQP definition

s_k_i = {};                 % Placeholder variables
s_k = [];

for i = 1:M         
    % Setting up the projection problem
    q = -2* z_0;
    proj{i} = osqp;
    proj{i}.setup(P_proj, q, A_i{i}, l_i{i}, u_i{i},'warm_start', true,'verbose',false);

    % Setting up the initialisation problem   
    P_init = 2*((Q_cost{i}-Q_add)/2 + C_cost{i}); % % % NO FACTOR 2 HERE % % % 
    q_init = c_i{i};
    init{i} = osqp;
    init{i}.setup(P_init, q_init, A_i{i}, l_i{i}, u_i{i},'warm_start', true,'verbose',false);
    
    % Initialising s_0(i)
    sol_init = init{i}.solve;
    s_k_i{i} = sol_init.x;
    s_k = [s_k; s_k_i{i}];
end

% Convergence problem set up in loop 

%% The Algorithm 

Nash = false;               % Nash condition has not been reached
k_n = 0;                    % Iteration of Nash seeking algorithm (not to be confused with timestep k)
LinVec = 0:10000:200000;     % Perform linearistaion only for these values of k_n (to not do too often)
LinVec = 0;
A_cc = 0;                   % Placeholder; to have an A_ccOld at start

% Thresholds
th_delta = delta * 10^(-10);
th_obj = -10^(-3);


% Finding a Nash equilibrium 
while Nash == false
    
    %% Updating the linearisation of CA condition  
    if ismember(k_n,LinVec)
        k_n
        sbar = s_k;                                         % Reference trajectory 

        % Coupling constraints 
       [A_cc, b_cc] = colcon(sbar, V, N, M, xypos, delta);
       m_cc = length(b_cc);
       b_ccw = repmat(-deltaw, m_cc,1)

        % Breaking up A_cc into blocks - for s_k update
        A_cc_i = {1,M};
        for i =1:M
            A_cc_i{i} = A_cc(:,(i-1)*n+1:i*n);
        end
      
        % Determining tau - New A_cc                    
        tau_up = (-L_F^2 ...
            + sqrt(L_F^4 + 4 * alpha^2 * norm(A_cc)^2))...
            /(2 * alpha * norm(A_cc)^2); 
        tau = 0.999 * tau_up;    % Ensuring it is below tau max


        % Defining the constraints for the linprog problem - lp stands for linprog
        % % % Must later be updated to be more in line with global constraints % % % 
        % Inequality constraint - Collision and z 
            % Cannot use A_z as it was not coded the same way
        Alp = [A_cc; kron(eye(M),A_z)];
        blp = [b_cc;repmat(u_z,M,1)];
        % Equality constraints - the model
        Aeqlp = kron(eye(M),Aeq);
        beqlp =[];
        for i = 1:M
            beqlp = [beqlp; leq_i{i}];
        end
        % Bounded constraints: the inputs, rest is ± inf
        lblp = repmat([repmat(-inf,nx*(N+1),1);repmat(-1,N*nu,1);repmat(-inf,N*nz,1)],3,1);
        ublp = -lblp;
        options = optimset('linprog');
        options.Display = 'off';
    
        % initalising parameters for Nash 
        lambdak = 1 * ones(m_cc,1);         % Definition of lambda must improve
    end
   
   %% Updating strategies
   %Compute sigma(u(k_n)) ; Reset at each step
    sigmak = sig_mul*s_k;

    % Compute s(k_n+1)
    s_kplus1_i = {};
    s_kplus1 =[];
    for i = 1:M
        z_kplus1 = (s_k_i{i} - tau*(gradsCost(M, Q_cost{i}, C_cost{i}, c_i{i}, s_k_i{i}, sigmak) + (A_cc_i{i})' * lambdak));

        q_k_i = -2*z_kplus1;

        proj{i}.update('q', q_k_i)
        sol = proj{i}.solve;
        s_kplus1_i{i} = sol.x;
        s_kplus1 = [s_kplus1; s_kplus1_i{i}];
         
    end

    % Update the dual variable
    % Use b_ccw until condition reached
    % Then use b_cc to reach Nash 
    
    if A_cc*s_k<=b_cc 
        % Replacing lambdak if a condition is reached 
        lambdak = max(0, (lambdak - tau * (b_cc - 2* A_cc * s_kplus1 + A_cc * s_k) ));
    else
        lambdak = max(0, (lambdak - tau * (b_ccw - 2* A_cc * s_kplus1 + A_cc * s_k) ));
    end 
    % Updating s_k
    s_k_i = s_kplus1_i;
    s_k = s_kplus1 ;   

  %%  Convergence check; Only run is cc met
    if A_cc*s_k<=b_cc 

        q_conv = F_Nm * s_k + F_Np;
        s_VI = linprog(q_conv,Alp, blp, Aeqlp, beqlp,  lblp, ublp, options);
        obj = q_conv' * s_VI;
        bm = (F_Nm * s_k + F_Np)' * s_k;    % Establishing the benchmark

        if obj-bm >= (0+ th_obj)
%             Fin = true
%             k2 = 0
            Nash = true      
        end

    end

    %% Increase k count 
    k_n = k_n+1;
    
end


%% Functions 
function [F_Nm,F_Np] = F_Nash(M,Q_cost,C_cost,c)
% This assumes a unique cost function for all agents 

% Follows eq (29) 
Q_FNm = [];
C_FNm1 = [];
C_FNm2 = [];
for i = 1:M
    Q_FNm = blkdiag(Q_FNm, Q_cost{i});
    C_FNm1 = [C_FNm1; repmat(C_cost{i},1,M)];
    C_FNm2 = blkdiag(C_FNm2, C_cost{i});
end
F_Nm = Q_FNm + 1/M * C_FNm1 + 1/M * C_FNm2;

% % % For a test, assuming unique cost fuction % % %
% F_Nm2 = kron(eye(M),Q_cost{1}) ...
%     + 1/M * kron(ones(M), C_cost{1})...   
%     + 1/M * kron(eye(M), C_cost{1}');

% Note: 1_M * 1_M ' = ones(M) 

F_Np = c;       
% Defining F_np seems unnecessary, but it is to keep F_Nash in a single place   

end

% Could gradxCost be merged with F_Nash 
function GsC = gradsCost(M, Q, C, c, s_i, sigma)
    % Using eq(11)
    GsC = 1/2 * 2 * Q * s_i + (C*sigma+c)+ 1/M * C * s_i;  
end

function [A_cc, b_cc] = colcon(sbar, V, N, M, xypos, delta)

    pbar = V*sbar;
    ij = M*(M-1);               % Number of constraints per timestep
    
    Eta_cc = [];                % Initialising variables (consider preallocating for speed later
    b_cc = [];

    for i = 1:ij
        pbarij = pbar((i-1)*N*xypos+1:i*N*xypos,1) ;    % Difference computed for the last N (not N+1) timesteps 
        for k = 2:N+1
            pbarijk = pbarij((k-2)*xypos+1:(k-1)*xypos,1);
            norpijk = norm(pbarijk);
            eta = pbarijk / norpijk;
           
            b_cc = [b_cc; norpijk - delta - eta'* pbarijk];
            Eta_cc((i-1)*N+(k-1), (i-1)*N*xypos+xypos*(k-2)+1:(i-1)*N*xypos+xypos*(k-1)) = eta';
        end
    end
    
    A_cc = -Eta_cc*V;
    
end

function plotpath(M, s_k, r, x0, Blocks, N, xypos)

    % Obtaining Positions
    posmat = {};
    for i = 1:M
        posvec = Blocks{i}*s_k;
        posmat{i} = [];
        for k = 1:(N)
           posk = [posvec(xypos*(k-1)+1), posvec(xypos*k)];
           posmat{i} = [posmat{i}; posk] ;
        end
    end
    % extract x and y data to plot - (x and y must be in positions 1 and 2)
    X = [];
    Y = [];
    for i=1:M
       X = [X ; [x0(1,i), posmat{i}(:,1)']];    % Adding in x0 and the path 
       Y = [Y ; [x0(2,i), posmat{i}(:,2)']]; 
    end
    
    hold on
    % Plot reference points
    for i = 1:M
        plot(r(1,i),r(2,i),'o')
        plot(x0(1,i),x0(2,i),'*') 
    end
    % Plot path
    for i = 1:M
        plot(X(i,:),Y(i,:),'r')
    end
end



    
