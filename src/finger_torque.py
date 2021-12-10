import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from torch import linalg as LA

class FingerTorque(object):
  
  INTERMD1_LINK = "finger_intermd1"
  INTERMD2_LINK = "finger_intermd2"  
  TIP_LINK = "finger_tip"
  TENDON_ANCHOR1_LINK = "tendon_site_tip1"
  TENDON_ANCHOR2_LINK = "tendon_site_tip2"
  JOINT1 = "finger1"
  JOINT2 = "finger2"
  JOINT3 = "finger3"
  JOINT1_SPRING_REF_ANGLE = -1.029142421
  JOINT2_SPRING_REF_ANGLE = -1.029142421
  JOINT3_SPRING_REF_ANGLE = -1.029142421
  JOINT1_SPRING_CONSTANT = 0.0128
  JOINT2_SPRING_CONSTANT = 0.0128
  JOINT3_SPRING_CONSTANT = 0.0128  
  
  def __init__(self, 
               gym, 
               prefix,
               rb_states,
               dof_states,
               envs,
               actors):
    self.gym = gym
    self.prefix = prefix
    self.rb_states = rb_states
    self.dof_states = dof_states
    self.envs = envs
    self.actors = actors
 
    self.bodies_per_env = gym.get_env_rigid_body_count(envs[0])
    self.finger_intermd1_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.INTERMD1_LINK, gymapi.DOMAIN_SIM)
    self.finger_intermd2_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.INTERMD2_LINK, gymapi.DOMAIN_SIM)
    self.finger_tip_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.TIP_LINK, gymapi.DOMAIN_SIM)
    self.tendon_anchor1_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.TENDON_ANCHOR1_LINK, gymapi.DOMAIN_SIM)
    self.tendon_anchor2_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.TENDON_ANCHOR2_LINK, gymapi.DOMAIN_SIM)

    self.finger_intermd1_pos = rb_states[self.finger_intermd1_idx::self.bodies_per_env, 0:3]
    self.finger_intermd2_pos = rb_states[self.finger_intermd2_idx::self.bodies_per_env, 0:3]
    self.finger_tip_pos = rb_states[self.finger_tip_idx::self.bodies_per_env, 0:3]
    self.tendon_anchor1_pos = rb_states[self.tendon_anchor1_idx::self.bodies_per_env, 0:3]
    self.tendon_anchor2_pos = rb_states[self.tendon_anchor2_idx::self.bodies_per_env, 0:3]    
    
    self.dof_per_env = gym.get_env_dof_count(envs[0])
    self.num_dofs = len(self.envs)*self.dof_per_env
    self.finger_dof1_idx = gym.find_actor_dof_index(envs[0], actors[0], prefix+self.JOINT1, gymapi.DOMAIN_SIM)
    self.finger_dof2_idx = gym.find_actor_dof_index(envs[0], actors[0], prefix+self.JOINT2, gymapi.DOMAIN_SIM)
    self.finger_dof3_idx = gym.find_actor_dof_index(envs[0], actors[0], prefix+self.JOINT3, gymapi.DOMAIN_SIM)   

    self.finger_dof1_pos = self.dof_states[self.finger_dof1_idx::self.dof_per_env,0]
    self.finger_dof2_pos = self.dof_states[self.finger_dof2_idx::self.dof_per_env,0]
    self.finger_dof3_pos = self.dof_states[self.finger_dof3_idx::self.dof_per_env,0]  
    
  def compute_tendon_torques(self, tendon_ctl):
    assert(len(tendon_ctl.shape) <= 1 or 
          (len(tendon_ctl.shape) == 2 and tendon_ctl.shape[1] == 1))
    
    tendon_torques = torch.zeros(self.num_dofs, dtype=torch.float32, device="cuda:0")
    
    # Compute lever arms, (n_actors, 3)
    intermd1_levers = self.tendon_anchor2_pos - self.finger_intermd1_pos
    intermd2_levers = self.tendon_anchor2_pos - self.finger_intermd2_pos
    tip_levers = self.tendon_anchor2_pos - self.finger_tip_pos    
    
    # Compute force, (n_actors, 3)      
    tendon_force_dirs = self.tendon_anchor1_pos - self.tendon_anchor2_pos
    tendon_force_norms = LA.vector_norm(tendon_force_dirs, dim=1)
    tendon_forces = torch.div(tendon_force_dirs, tendon_force_norms[:, None])
    torch.mul(tendon_ctl.reshape((-1,1)), tendon_forces, out=tendon_forces)

    # Compute torque, (n_actors, 3)
    tendon_intermd1_torque = torch.cross(intermd1_levers, tendon_forces, dim=1)
    tendon_intermd2_torque = torch.cross(intermd2_levers, tendon_forces, dim=1)
    tendon_tip_torque = torch.cross(tip_levers, tendon_forces, dim=1)     
    
    tendon_torques[self.finger_dof1_idx::self.dof_per_env] = tendon_intermd1_torque[:,2]
    tendon_torques[self.finger_dof2_idx::self.dof_per_env] = tendon_intermd2_torque[:,2]
    tendon_torques[self.finger_dof3_idx::self.dof_per_env] = tendon_tip_torque[:,2]  
    
    return tendon_torques
    
  def compute_spring_torques(self):
    spring_torques = torch.zeros(self.num_dofs, dtype=torch.float32, device="cuda:0")        
    
    finger_dof1_spring_torque = -1*self.JOINT1_SPRING_CONSTANT*(self.finger_dof1_pos - self.JOINT1_SPRING_REF_ANGLE)
    finger_dof2_spring_torque = -1*self.JOINT2_SPRING_CONSTANT*(self.finger_dof2_pos - self.JOINT2_SPRING_REF_ANGLE)
    finger_dof3_spring_torque = -1*self.JOINT3_SPRING_CONSTANT*(self.finger_dof3_pos - self.JOINT3_SPRING_REF_ANGLE)
    
    spring_torques[self.finger_dof1_idx::self.dof_per_env] = finger_dof1_spring_torque
    spring_torques[self.finger_dof2_idx::self.dof_per_env] = finger_dof2_spring_torque
    spring_torques[self.finger_dof3_idx::self.dof_per_env] = finger_dof3_spring_torque
    
    return spring_torques  
    
    
    
