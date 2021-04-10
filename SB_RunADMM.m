% Stanislas Dumas
% Started on 10/04/21
% 4YP 20/21 academic year
% Oxford University
% Supervisor: Kostas Margellos 
% Running the ADMM GNEP (for Scenario Based Analysis)

%% ADMM Loop 

Nash = false;
k_n = 0; 
visu = 0;

% Thresholds
th_delta = delta * 10^(-10);
th_obj = -10^(-5);

% Convergence check
obj = 0; 
ObjChange = [];

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
       w_k = [];
       for j = 1:M
           w_tof(:,i,j) = resc.x((j-1)*n+1 : j*n);
           w_k = [w_k; w_tof(:,i,j)];
       end
   end
   
    %% Convergence check
    if A_cc*s_k<=(b_cc + th_delta)

        q_conv = F_Nm * s_k + F_Np;
        s_VI = linprog(q_conv,Alp, blp, Aeqlp, beqlp,  lblp, ublp, options);
        obj = q_conv' * s_VI;
        bm = (F_Nm * s_k + F_Np)' * s_k;    % Establishing the benchmark

        if obj-bm >= (0 + th_obj)
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
    for i = 1:M
        q = [q; - gamma_tof(:,agent, i) - rho * s(:,i)];
    end

end
