%% environment setup
open_system("RL_SAC")
% observation parameters
obsInfo = rlNumericSpec([5 1], LowerLimit=[-20 0 -3 -2 -3]', UpperLimit=[20 1 3 2 3]');
numObservations = obsInfo.Dimension(1);
% action parameters - upper and lower limits
numActions = 1;
actInfo = rlNumericSpec([numActions 1], "LowerLimit",-1, "UpperLimit", 1);
%% create environment object
% set paths
env = rlSimulinkEnv("RL_SAC","RL_SAC/Attitude Control/RL/RL Agent", obsInfo,actInfo);
% run reset fcn to randomise parameters
env.ResetFcn = @localResetFcn;
% sample time
sampleTime = 0.01; 
% sim time
simTime = 10;

env.UseFastRestart = "off"; 
%% critic network 1
statePath1 = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')
    ];
actionPath1 = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1')
    ];
commonPath1 = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1,'Name','CriticOutput')
    ];

criticNet1 = layerGraph(statePath1);
criticNet1 = addLayers(criticNet1,actionPath1);
criticNet1 = addLayers(criticNet1,commonPath1);
criticNet1 = connectLayers(criticNet1,'CriticStateFC2','add/in1');
criticNet1 = connectLayers(criticNet1,'CriticActionFC1','add/in2');

figure
plot(criticNet1)

%% critic network 2
%critic 2
statePath2 = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')
    ];
actionPath2 = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1')
    ];
commonPath2 = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1,'Name','CriticOutput')
    ];

criticNet2 = layerGraph(statePath2);
criticNet2 = addLayers(criticNet2,actionPath2);
criticNet2 = addLayers(criticNet2,commonPath2);
criticNet2 = connectLayers(criticNet2,'CriticStateFC2','add/in1');
criticNet2 = connectLayers(criticNet2,'CriticActionFC1','add/in2');

figure
plot(criticNet2)

%% create critic
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);
critic1 = rlQValueRepresentation(criticNet1,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
critic2 = rlQValueRepresentation(criticNet2,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(400, 'Name','commonFC1')
    reluLayer('Name','CommonRelu')];
meanPath = [
    fullyConnectedLayer(300,'Name','MeanFC1')
    reluLayer('Name','MeanRelu')
    fullyConnectedLayer(numActions,'Name','Mean')
    ];
stdPath = [
    fullyConnectedLayer(300,'Name','StdFC1')
    reluLayer('Name','StdRelu')
    fullyConnectedLayer(numActions,'Name','StdFC2')
    softplusLayer('Name','StandardDeviation')
    ];

concatPath = [
    concatenationLayer(1,2,'Name','GaussianParameters')
    ];

actorNetwork = layerGraph(statePath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);
actorNetwork = addLayers(actorNetwork,concatPath);
actorNetwork = connectLayers(actorNetwork,'CommonRelu','MeanFC1/in');
actorNetwork = connectLayers(actorNetwork,'CommonRelu','StdFC1/in');
actorNetwork = connectLayers(actorNetwork,'Mean','GaussianParameters/in1');
actorNetwork = connectLayers(actorNetwork,'StandardDeviation','GaussianParameters/in2');

figure
plot(actorNetwork)

actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,...
                                       'GradientThreshold',1,'L2RegularizationFactor',1e-5);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,actorOptions,...
    'Observation',{'observation'});

SampleTime = 0.05; % 0.05
agentOptions = rlSACAgentOptions;
agentOptions.SampleTime = SampleTime;
agentOptions.NumWarmStartSteps = 1000; 
agentOptions.PolicyUpdateFrequency = 2; %10
agentOptions.CriticUpdateFrequency = 8; %10
agentOptions.DiscountFactor = 0.99;
agentOptions.TargetSmoothFactor = 0.005; %1e-3
agentOptions.ExperienceBufferLength = 1e5;
agentOptions.MiniBatchSize = 128; %96
agentOptions.NumStepsToLookAhead = 3; %3
agentOptions.SaveExperienceBufferWithAgent =true;
agentOptions.ResetExperienceBufferBeforeTraining = true; % set to false if resuming

%create agent object
RLtestSAC3Agent = rlSACAgent(actor,[critic1 critic2],agentOptions);
open_system("RL_SAC/Attitude Control/RL/RL Agent")

%train
maxEpisodes = 2000;
maxSteps = ceil(simTime/sampleTime);
trainOpts = rlTrainingOptions(...
    "MaxEpisodes",maxEpisodes, ...
    "MaxStepsPerEpisode",maxSteps, ...
    "ScoreAveragingWindowLength",50, ... 
    "StopTrainingCriteria","AverageReward", ...
    "StopTrainingValue",450, ...
    "Verbose", true, ...
    "Plots","training-progress");
trainOpts.SaveAgentCriteria = "EpisodeReward";
trainOpts.SaveAgentValue = 500;



%{
trainOpts.SaveAgentCriteria = "EpisodeReward";
trainOpts.SaveAgentValue = inf;
%}

doTraining = false; % Toggle this to true for training. 

if doTraining
    % Load the agent from the previous session
    trainingStats = train(RLtestSAC3Agent,env,trainOpts);
    save('RL_SACAgent26aug2.mat', 'RLtestSAC3Agent') %% activate if want to save or copy the agent at the end training
    
else
    % Load pretrained agent for the example. 
    rng(0);
    load RLtestSAC3Agent25aug3.mat RLtestSAC3Agent
end
% save('RL_SACAgent.mat', 'RLtestSAC3Agent')
%run sim
rng(0);

simOpts = rlSimulationOptions( ...
    MaxSteps=ceil(simTime/sampleTime), ...
    StopOnError="on");
experiences = sim(env,RLtestSAC3Agent,simOpts);

%% reset funciton

function in = localResetFcn(in)
previousRngState = rng(0,"twister");
rng(previousRngState);



%{
% Randomize init speed
blk = sprintf("RLtestSAC3/Y velocity target"); 
is = 2*randn + 3;
while is <= 0 || is >= 15
    is = 10*randn + 7;
end
in = setBlockParameter(in,blk,Value=num2str(is));
blk = "RLtestSAC3/Dynamic Model/Discrete-Time Integrator3";
in = setBlockParameter(in,blk,InitialCondition=num2str([is; 0; 0]));
blk = "RLtestSAC3/Dynamic Model/Discrete-Time Integrator1";
in = setBlockParameter(in,blk,InitialCondition=num2str([is; 0; 0]));
%}

%{
% Randomize disturbance
blk = sprintf("RLtestSAC3/Constant1"); 
dis = 3 + 4*rand;
in = setBlockParameter(in,blk,Value=num2str(dis));
%}
end