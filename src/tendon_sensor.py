import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from torch import linalg as LA

class TendonLinkParams(object):

  def __init__(self, anchor_x, anchor_y, lever_x, lever_y, theta0):
    self.anchor_x = anchor_x
    self.anchor_y = anchor_y
    self.lever_x = lever_x
    self.lever_y = lever_y
    self.theta0 = theta0


class TendonSensor(object):

  BASE_TENDON_LINK = "tendon_site_base"
  INTERMD1_TENDON_LINK = "tendon_site_intermd1"
  INTERMD2_TENDON_LINK = "tendon_site_intermd2"
  TIP_TENDON_LINK = "tendon_site_tip1"
  BASE_LINK = "finger_base"
  INTERMD1_LINK = "finger_intermd1"
  INTERMD2_LINK = "finger_intermd2"  
  TIP_LINK = "finger_tip"
  JOINT1 = "finger1"
  JOINT2 = "finger2"
  JOINT3 = "finger3"
  TLP1 = TendonLinkParams(0.002238, 0.021763, 0.032257, 0.005655, 0.7854)
  TLP2 = TendonLinkParams(0.009884, 0.006905, 0.027257, 0.006905, 0.0)
  TLP3 = TendonLinkParams(0.009884, 0.006905, 0.014957, 0.006725, 0.0)    
  
  def __init__(self,
               gym,
               prefix,               
               rb_states,
               dof_states,
               envs,
               actors,
               dt):
      
    self.gym = gym
    self.prefix = prefix
    self.rb_states = rb_states
    self.dof_states = dof_states
    self.envs = envs
    self.actors = actors
    self.dt = dt
    
    self.bodies_per_env = gym.get_env_rigid_body_count(envs[0])
    self.tendon_site_base_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.BASE_TENDON_LINK, gymapi.DOMAIN_SIM)    
    self.tendon_site_intermd1_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.INTERMD1_TENDON_LINK, gymapi.DOMAIN_SIM)       
    self.tendon_site_intermd2_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.INTERMD2_TENDON_LINK, gymapi.DOMAIN_SIM)     
    self.tendon_site_tip_idx = gym.find_actor_rigid_body_index(envs[0], actors[0], prefix+self.TIP_TENDON_LINK, gymapi.DOMAIN_SIM)  

    self.dof_per_env = gym.get_env_dof_count(envs[0])
    self.num_dofs = len(self.envs)*self.dof_per_env
    self.finger_dof1_idx = gym.find_actor_dof_index(envs[0], actors[0], prefix+self.JOINT1, gymapi.DOMAIN_SIM)
    self.finger_dof2_idx = gym.find_actor_dof_index(envs[0], actors[0], prefix+self.JOINT2, gymapi.DOMAIN_SIM)
    self.finger_dof3_idx = gym.find_actor_dof_index(envs[0], actors[0], prefix+self.JOINT3, gymapi.DOMAIN_SIM)   
  
    self.tendon_site_base_pos = rb_states[self.tendon_site_base_idx::self.bodies_per_env, 0:3]
    self.tendon_site_intermd1_pos = rb_states[self.tendon_site_intermd1_idx::self.bodies_per_env, 0:3]  
    self.tendon_site_intermd2_pos = rb_states[self.tendon_site_intermd2_idx::self.bodies_per_env, 0:3]
    self.tendon_site_tip_pos = rb_states[self.tendon_site_tip_idx::self.bodies_per_env, 0:3]

    self.finger_dof1_pos = self.dof_states[self.finger_dof1_idx::self.dof_per_env,0]
    self.finger_dof2_pos = self.dof_states[self.finger_dof2_idx::self.dof_per_env,0]
    self.finger_dof3_pos = self.dof_states[self.finger_dof3_idx::self.dof_per_env,0]  
    
    self.prev_finger_dof1_pos = self.finger_dof1_pos.clone()
    self.prev_finger_dof2_pos = self.finger_dof2_pos.clone()
    self.prev_finger_dof3_pos = self.finger_dof3_pos.clone()        
      
    self.tendon_length_dev_rb = None
    self.tendon_length_dev_analytic = None
  
  def compute_tendon_length_dv_rb(self):
    tendon_length1 = LA.vector_norm(self.tendon_site_intermd1_pos - self.tendon_site_base_pos, dim=1)
    tendon_length2 = LA.vector_norm(self.tendon_site_intermd2_pos - self.tendon_site_intermd1_pos, dim=1)
    tendon_length3 = LA.vector_norm(self.tendon_site_tip_pos - self.tendon_site_intermd2_pos, dim=1)        
    
    tendon_length = tendon_length1 + tendon_length2 + tendon_length3
    
    if self.tendon_length_dev_rb is None:
      self.tendon_length_dev_rb = tendon_length
    
    return tendon_length - self.tendon_length_dev_rb
    
    
  def compute_tendon_length_dv_analytic(self):
    tendon_length1 = self.compute_tendon_length_analytic(self.finger_dof1_pos,
                                                         self.TLP1)
    tendon_length2 = self.compute_tendon_length_analytic(self.finger_dof2_pos,
                                                         self.TLP2)
    tendon_length3 = self.compute_tendon_length_analytic(self.finger_dof3_pos,
                                                         self.TLP3)
    
    tendon_length = tendon_length1 + tendon_length2 + tendon_length3
                                                     
    if self.tendon_length_dev_analytic is None:
      self.tendon_length_dev_analytic = tendon_length
      
    return tendon_length - self.tendon_length_dev_analytic
                                                             
  def compute_tendon_length_analytic(self, dof_pos, tlp):
    dof_cos = torch.cos(dof_pos-tlp.theta0)
    dof_sin = torch.sin(dof_pos-tlp.theta0)
    
    length = (tlp.anchor_x**2)+(tlp.anchor_y**2)+(tlp.lever_x**2)+(tlp.lever_y**2)
    length = length - 2*(tlp.anchor_y*tlp.lever_x+tlp.anchor_x*tlp.lever_y)*dof_sin
    length = length + 2*(tlp.anchor_x*tlp.lever_x-tlp.anchor_y*tlp.lever_y)*dof_cos
    
    return torch.sqrt(length)
  
  def compute_tendon_velocity(self):
    # When sim_params.solver_type = 1, dof velocity does not seem to be correct,
    # for example when applying return spring forces but joint limits prevent
    # a link from moving, the reported velocity will be non-zero. Instead we use
    # finite differencing, which matches the sim computed dof velocity when
    # sim_params.solver_type = 0
    finger_dof1_vel = (self.finger_dof1_pos - self.prev_finger_dof1_pos)/self.dt
    finger_dof2_vel = (self.finger_dof2_pos - self.prev_finger_dof2_pos)/self.dt
    finger_dof3_vel = (self.finger_dof3_pos - self.prev_finger_dof3_pos)/self.dt        
    
    self.prev_finger_dof1_pos = self.finger_dof1_pos.clone()
    self.prev_finger_dof2_pos = self.finger_dof2_pos.clone()
    self.prev_finger_dof3_pos = self.finger_dof3_pos.clone()
    
    velocity1 = self.compute_tendon_velocity_analytic(self.finger_dof1_pos,
                                                      finger_dof1_vel,
                                                      self.TLP1)
    velocity2 = self.compute_tendon_velocity_analytic(self.finger_dof2_pos,
                                                      finger_dof2_vel,
                                                      self.TLP2)  
    velocity3 = self.compute_tendon_velocity_analytic(self.finger_dof3_pos,
                                                      finger_dof3_vel,
                                                      self.TLP3)
                                                 
    return velocity1 + velocity2 + velocity3 
                                                                                                   
  def compute_tendon_velocity_analytic(self, dof_pos, dof_vel, tlp):
    dof_cos = torch.cos(dof_pos-tlp.theta0)
    dof_sin = torch.sin(dof_pos-tlp.theta0)
    length = self.compute_tendon_length_analytic(dof_pos, tlp)
    
    velocity = (tlp.anchor_x*tlp.lever_x-tlp.anchor_y*tlp.lever_y)*dof_sin
    velocity = velocity + (tlp.anchor_y*tlp.lever_x+tlp.anchor_x*tlp.lever_y)*dof_cos
    velocity = torch.mul(velocity, -1*dof_vel)
    velocity = torch.div(velocity, length)
    
    return velocity


    
