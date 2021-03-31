% Stanislas Dumas
% Started on 31/03/21
% 4YP 20/21 academic year
% Oxford University
% Supervisor: Kostas Margellos 
% Centralised solution to the optimal problem 


%% Pre-Defined Model 
% (From Aren's work) 
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

% Prediction
N = 25;                     % Prediction horizon 
n = (N+1)*nx + N*nu + N*nz;
eta = 0.01;                 % Added to leading diagonal of Quadrti cost 


% Initial and reference states
x0 = [-delta 0 0 0 0 0; delta 0 0 0 0 0; 0 sqrt(3)*delta 0 0 0 0]';
r = [2*delta -delta/2 0 0 0 0; -2*delta delta/2 0 0 0 0; 0 -delta 0 0 0 0]';

% Define the vector that goes into the individual cost matrices
Q_vec = [4, 4, 0, 0, 0, 0;          % Weighting of deviation from reference; dimension nx
         4, 4, 0, 0, 0, 0;
         4, 4, 0, 0, 0, 0];
S_vec = [1,1;                       % Weighting of use of inputs ; dimension nz
         1, 1;
         1, 1];

%% Useful Matrices  
% Sigma mulriplier
sig_mul = 1/M * kron(ones(1,M), eye(n));
sig_mulz = 1/M * kron(ones(1,M),  blkdiag(zeros(n-(N*nz)) ,eye(N*nz)));

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

%% Setting up the cost function elements (from the game)  
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

%% Setting up individual constraints 
% written in an OSQP format
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

%% Setup and intialisation 

% Collision Avoidance
s_k = [];
for i = 1:M         
    % Setting up the initialisation problem   
    P_init = 2*((Q_cost{i})/2 + C_cost{i}); % % % NO FACTOR 2 HERE % % % 
    q_init = c_i{i};
    init{i} = osqp;
    init{i}.setup(P_init, q_init, A_i{i}, l_i{i}, u_i{i},'warm_start', true,'verbose',false, 'eps_rel', 10^(-12), 'polish', true);
    
    % Initialising s_0(i)
    sol_init = init{i}.solve;
    s_k_i{i} = sol_init.x;
    s_k = [s_k; s_k_i{i}];
end

% Limits 
sbar = s_k; 
% plotpath(M, sbar, r, x0, Blocks, N, xypos)
[A_cc, b_cc] = colcon(sbar, V, N, M, xypos, delta);
m_cc = size(A_cc,1);
l_cc = -inf * ones(m_cc,1); 

% Complete Constraints - Adding A_cc
A_aug = [A_aug; A_cc];
u_aug = [u_aug; b_cc];
l_aug = [l_aug; l_cc];

% Optimisation problem
P_agg = [];
for i = 1:M
    Q_layer = [zeros(n, n* (i-1)), Q_cost{i}, zeros(n, n * (M-i))];
    C_layer = 2 * C_cost{i} * sig_mulz;
    
    P_agg = [P_agg; Q_layer + C_layer];
end
q_agg = c_aug;

%% Solving the problem 
% No adaptive linearisation
prob = osqp;
prob.setup(P_agg, q_agg, A_aug, l_aug, u_aug, 'warm_start', true,'verbose',true, 'polish', true, 'eps_rel', 10^(-5));

sol = prob.solve;
s_k = sol.x;

plotpath(M, s_k, r, x0, Blocks, N, xypos)

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



    
