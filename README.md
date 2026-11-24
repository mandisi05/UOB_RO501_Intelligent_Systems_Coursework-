# UOB_RO501_Intelligent_Systems_Coursework-
The following program is a coursework task assigned to University of Brighton students as Part of our Intelligent systems module. The assignment consisted of creating and training a free choice Neural Network ( NN ) or Fuzzy based decision-making agent. I opted for a gymnasium based reinforcement learning library using the 'Frozen-Lake' Toy Text environment agent.

# Simple function description
The Agent uses a simpleQNetwork class utilizing backpropogation to run system rewards based on the models decision-making, [-1] denoting a negative action and (falling into water), [+1] promoting positive behaviour (reaching goal), and not rewarding [0] for actions that don't achieve anything (looping indefinitely). The model uses two weight matrices passing ReLu to prevent gradient vanishing and notify the system if a neuron was inactive during a forward pass to update the backward pass values. The system then uses the imported Deque library to select a batch size [32] from the models previous attempts; preventing temporal correlation. Finally, the agent uses gymnasium's Epilson-Greedy exploration to start training at 100% random movement decaying by 1% until the system reaches 0% ensuring the agent has explored the entire map. The system then selects 5 epochs throughout the training and displays the ['human render'] agent completing the task. In this case the reward system points are [+5] for sucess and [-2] for failure. 

# Code Dependencies
The code requires the following libraries 
- [Anaconda/ Miniconda]
- [Gymnasium]
- [Numpy], [Deque], [Random], [Copy]
