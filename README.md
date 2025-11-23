# How Generalizable is Reiforcement Learning in Navigating Newly Seen Maps?

Objective: Can an RL agent trained on procedurally generated 3D ViZDoom mazes generalize navigation to newly seen maps?

Rationale: RL typically works in fixed environments. We wanted to see if we could test RL algorithms to learn in new environments. The aim to evaluate how well RL can generalize in new 3D spaces.  

The findings from this research would provide applications to:  
- new video games  
- flying drones in complex and changing spaces  
- teaching an RL agent to navigate new cellular environments (https://pmc.ncbi.nlm.nih.gov/articles/PMC9947252/)  

Goals: 
    (1) Compare value based and actor critic methods  
    (2) Training on Easy, Medium, and Hard mazes  
    (3) Training on random mazes for train and test split (maybe if we have time)  

Will test:
    (1) Quake (depending on environment complexity - prioritize learning in RL over CS set up)     
    (2) ViZDoom (Proof of Concept)  

Models:  
- DQN  
- PPO  
- A2C  