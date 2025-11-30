# How Generalizable is Reiforcement Learning in Navigating Newly Seen Maps?

Objective: RL typically works in fixed environments. We wanted to see if we could test RL algorithms to learn in changing and new environments. **The aim to evaluate how well RL can generalize in new 3D spaces.**  

The findings from this research would provide applications to:  
- new video games  
- flying drones in complex and changing spaces  
- teaching an RL agent to navigate new cellular environments (https://pmc.ncbi.nlm.nih.gov/articles/PMC9947252/)  

Goals:  
    (1) Compare value based and actor critic methods  
    (2) Training on Easy, Medium, and Hard spaces (i.e. mazes)  
    (3) Training on random mazes for train and test split (maybe if we have time)  

Will test generalizability in two video game environments:  
    (1) Quake (creating new RL benchamrk)     
    (2) ViZDoom (RL modeling on existing video game benchmark)  

Models:  
**Value-based**   
- DQN  
**Actor Critic**  
- PPO    
- A2C   

11/29/2025
Set up Frog Fly environment and model. Experimented with PPO, SAC, and IMPALA. Also, tried adding a transformer policy to PPO to test if the model can handle partial observability. Please see code in /frog_fly

Next, would like to test transformer policy with SAC.
Tiana is starting to write paper on 11/30/25