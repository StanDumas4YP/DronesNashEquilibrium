% Stanislas Dumas
% Started on 31/03/21
% 4YP 20/21 academic year
% Oxford University
% Supervisor: Kostas Margellos 
% ADMM like algorithm for the GNEP

%% Pre-Defined Model
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
sizA = size(Ad);
sizB = size(Bd);
nx = sizA(2);    % number of states for an agent
nu = sizB(2);    % number of inputs for an agent
nz = nu;         % number of cost variables for an agent
xypos = 2;       % x and y are in positions 1 and 2, THAT MUST STAY TRUE

% Collision avoidance
delta = 0.3;

% ADMM parameters
rho = 1;

% Prediction
N = 25;         % Prediction horizon 
n = (N+1)*nx + N*nu + N*nz;
nw = n;

% Initial and reference states
x0 = [-delta 0 0 0 0 0; delta 0 0 0 0 0; 0 sqrt(3)*delta 0 0 0 0]';
r = [2*delta -delta/2 0 0 0 0; -2*delta delta/2 0 0 0 0; 0 -delta 0 0 0 0]';

Q_vec = [4, 4, 0, 0, 0, 0;          % Weighting of deviation from reference; dimension nx
         4, 4, 0, 0, 0, 0;
         4, 4, 0, 0, 0, 0];
S_vec = [[1,1];                       % Weighting of use of inputs ; dimension nz
         [1, 1];
         [1, 1]];
     
%% Useful Matrices

sig_mul = 1/M * kron(ones(1,M), eye(n));

% extracting positon vector from states (names for posM and posMN taken from Aren)
posM = blkdiag(eye(xypos), zeros(nx-xypos, nx-xypos)); 

% Extracting position vector from strategy vector
pos_sel = [eye(xypos), zeros(xypos, nx-xypos)];
posMN = [kron(eye(N+1), pos_sel), zeros((N+1)*xypos, N*nu), zeros((N+1)*xypos, N*nz)];
posZ = [zeros(N*nz, (N+1)*nx), zeros(N*nz, (N)*nu) ,eye(N*nz)];
posMNZ = [posMN; posZ];

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

%% Setting up the cost function elements 
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
for i = 1:M
    Qi = diag(Q_vec(i,:));
    c_i{i} = [repmat(-2*Qi*r(:,i),N+1,1); zeros(N*nu,1); zeros(N*nz,1)];
end

%% Setting up individual constraints; 
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

%% Setting up Collision constraint
init = {};
for i = 1:M 
    % Initialisation problem 
    P_init = 2*(Q_cost{i}/2 + C_cost{i});
    q_init = c_i{i};
    init{i} = osqp;
    init{i}.setup(P_init, q_init, A_i{i}, l_i{i}, u_i{i},'warm_start', true,'verbose',false);
    
    % Initialising s_0(i) - for A_cc
    sol_init = init{i}.solve;
    s_k(n*(i-1)+1:n*i,1) = sol_init.x;
end 

sbar = s_k;
[A_cc, b_cc] = colcon(sbar, V, N, M, xypos, delta);
m_cc = size(A_cc,1);
l_cc = -inf * ones(m_cc,1);

%% Initilaising optimisation problems 
% Prediction
pred = cell(1,M);
for i = 1:M 
    % Costs
    P_pred = 2 * (1/2 * Q_cost{i} + 1/M * C_cost{i} + eye(n)*M*rho/2);
    q_pred = c_i{i}; % Placeholder
    
    pred{i} = osqp;
    pred{i}.setup(P_pred, q_pred, A_i{i}, l_i{i}, u_i{i}, 'warm_start', true, 'verbose', false)   
end

% Coordination
coor = cell(1,M);
for i = 1:M 
    % Costs
    P_coor = 2 * kron(eye(M), kron(eye(n), rho/2));
    q_coor = zeros(M*n,1); % Placeholder
    
    coor{i} = osqp;
    coor{i}.setup(P_coor, q_coor, A_cc, l_cc, b_cc, 'warm_start', true, 'verbose', false)   
end

%% Setting up the convergence check 
% Inequality constraint
% Cannot use A_z as it was not coded the same way
Alp = [A_cc; kron(eye(M),[zeros(N*nu*2, (N+1)*nx),[eye(N*nu);-eye(N*nu)],[-eye(N*nu);-eye(N*nu)]])];
blp = [b_cc;repmat(zeros(2*N*nu,1),M,1)];
% Equality constraints: the model
Aeqlp = kron(eye(M),Aeq);
beqlp =[];
for i = 1:M
    beqlp = [beqlp; leq_i{i}];
end
% Bounded constraints: the inputs, rest is ± inf
lblp = repmat([repmat(-inf,nx*(N+1),1);repmat(-1,N*nu,1);repmat(-inf,N*nz,1)],3,1);
ublp = -lblp;

% Cost function element
c_aug = [];
for i = 1:M
   c_aug = [c_aug; c_i{i}];
end

% Linear problem Parameters
[F_Nm,F_Np] = F_Nashv6(M,Q_cost,C_cost,c_aug);
options = optimset('linprog');
options.Display = 'off';

%% Initialising ADMM parameters
% dual variable and to and from
w_tof = zeros(n,M,M);
gamma_tof = zeros(n,M,M);

%% ADMM Loop 

Nash = false;
k_n = 0; 
visu = 0;

% Thresholds
th_delta = delta * 10^(-10);
th_obj = -10^(-3);

% Convergence check
obj = 0; 

while Nash == false
    %% Prediction
    for i = 1:M
        qp = pred_q(i, gamma_tof, w_tof, C_cost, c_i, rho, M);
        pred{i}.update('q', qp)

        resp = pred{i}.solve;
        s(:,i) = resp.x;
        s_k((i-1)*n+1 : i*n) = s(:,i);
    end 
   
   %% Coordination
   for i = 1:M
       ind = 1:M;
       ind(i) = [];
       qc = coor_q(i, gamma_tof, s, rho, M);
       coor{i}.update('q', qc)
       
       resc = coor{i}.solve;
       for j = 1:M
           w_tof(:,i,j) = resc.x((j-1)*n+1 : j*n);
       end
   end
   
   w_k = [];
   for i = 1:M
      w_k = [w_k; w_tof(:,i,i)]; 
   end
   
    %% Convergence check
    if A_cc*s_k<=(b_cc + th_delta)

        q_conv = F_Nm * s_k + F_Np;
        s_VI = linprog(q_conv,Alp, blp, Aeqlp, beqlp,  lblp, ublp, options);
        obj = q_conv' * s_VI;
        bm = (F_Nm * s_k + F_Np)' * s_k;    % Establishing the benchmark

        if obj-bm >= (0 + th_obj)
%             Fin = true
%             k2 = 0
            Nash = true      
        end

    end
    
   %% Mediation 
    for i = 1:M
        for j = 1:M
            gamma_tof(:,i,j) = gamma_tof(:,i,j) + rho * (s(:,j) - w_tof(:,i,j));
        end
    end 
   
   k_n = k_n + 1 ;
   
   
end

%% Functions

function q = pred_q(agent, gamma_tof, w_tof, C_cost, c_i, rho, M)
    ind = 1:M;
    ind(agent) = [];
    q = 0;
    
    % J element
    q = q + c_i{agent};
    for i = 1:M-1
       q = q + 1/M * C_cost{agent}' * w_tof(:,ind(i), ind(i));  
    end
    % Dual element
    q = q + gamma_tof(:,agent,agent); 
    % Penalty element
    q = q - rho * w_tof(:,agent,agent);
    % To/from element
    for i = 1:(M-1)
        q = q + gamma_tof(:, ind(i), agent);
        q = q - rho * w_tof(:, ind(i), agent);
    end
end

function q = coor_q(agent, gamma_tof, s, rho, M)
    ind = 1:M;
    ind(agent) = [];
    
    q = [];
    for j = 1:M
        q = [q; - gamma_tof(:,agent, j) - rho * s(:,j)];
    end

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

function [F_Nm,F_Np] = F_Nashv6(M,Q_cost,C_cost,c)
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

end % For the convergence check 

function plotpath(M, s_k, r, x0, Blocks, N, xypos, col)

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
        plot(X(i,:),Y(i,:),'Color', col)
    end
end


