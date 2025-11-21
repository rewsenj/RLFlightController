
To run these you will need:
MATLAB R2025a			- Version 25.1
Simulink R2025a			- Version 25.1
Control System Toolbox	        - Version 25.1
Reinforcement Learning Toolbox  - Version 25.1
Deep Learning Toolbox 		- Version 25.1

Ensure all files are in the same folder and the folder is added to path in MATLAB.
To run, open both the Simulink model, RL_SAC.slx, and MATLAB script, then run the MATLAB script: RL_SACscript.m

This should load the policy to the workspace and you can run simulations within Simulink.

To change gust magnitude(m/s) or duration(s), adjust the corresponding constant blocks. Scopes are on the right side for linear position, attitude and estimated vs true position.

To see more detail. e.g. if you turn off X in the linear position scope, press space to scale the remaining lines.