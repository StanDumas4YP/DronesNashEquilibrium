# Multi Quadcopter Games 4YP
This repository contains the code for Stanislas Dumas' 4YP at the University of Oxford

Scripts included:

- Game_M_drones_APA.m
  - The APA GNEP solving algorithm for M drones, defined in section 5  
  - Number of drones, horizon, intital position, reference points and weighing matrices can all be changed in the 'variables' Section of the code
  - A through convergence check is included, relying on private agent information and threholds defined manually

- Game_ADMM.m
  - The ADMM GNEP solving algorithm for M drones, defined in section 7   
  - Number of drones, horizon, intital position, reference points and weighing matrices can all be changed in the 'variables' Section of the code
  - A through convergence check is included, relying on private agent information and threholds defined manually

- Centralised_Agg.m
  - A centrlaised algorithm finding the optimal solution for the sum of the individual agent cost function
  - Number of drones, horizon, intital position, reference points and weighing matrices can all be changed in the 'variables' Section of the code

- SB_SetupADMM.m
  - Must be reun before SB_RunADMM.m
  - Generates a set of modified constraints 
  - Initialises all parameters and storing variables necessary to run the GNEP seeking ADMM algorithm
  - Workspace MUST NOT be cleared after running

- SB_RunADMM.m
  - Must be run after SB_SetupADMM.m 
  - Uses previously defined parameters to output strategies s_k and w_k using the ADMM like GNEP solving algorithm (section 7)
  - Workspace MUST NOT be cleared after running

- SB_AnalyseADMM.m 
  - Must be run after SB_RunADMM.m
  - Outputs a prioir and a posteriori certificates on the game setup and solved in SB_SetupADMM.m and SB_RunADMM.m
  - Conducts a Monte Carlo analysis to analyse the precision of said certificates
