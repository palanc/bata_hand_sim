"""

BATA Hand Sim
-------------
- Simulates the bata hand model

"""

import rospy
import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from bata_hand_sim.finger_torque import FingerTorque
from bata_hand_sim.tendon_sensor import TendonSensor

class BATAHandSim(object):
  asset_root = "/home/patrick/bata_hand_ws/devel/share/bata_hand_description"
  hand_asset_file = "robots/bata_hand_isaac.urdf"
  object_asset_file = "robots/cylinder.urdf"
  goal_asset_file = "robots/short_cylinder.urdf"
  
  def __init__(self, 
               gym,
               num_envs,
               brake_configs,
               args,
               headless=False):
               
    self.gym = gym
    self.num_envs = num_envs
    self.brake_configs = brake_configs
    if self.brake_configs is not None and len(self.brake_configs.shape) == 1:
      self.brake_configs = self.brake_configs[None]
      
    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    self.sim = gym.create_sim(args.compute_device_id, 
                              args.graphics_device_id,
                              args.physics_engine,
                              sim_params)
    self.dt = sim_params.dt
                                      
    if self.sim is None:
      print("*** Failed to create sim")
      quit()    
    
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 0.001
    plane_params.dynamic_friction = 0.001
    plane_params.restitution = 0
    gym.add_ground(self.sim, plane_params)    
    
    if not headless:
      # create viewer
      self.viewer = gym.create_viewer(self.sim, gymapi.CameraProperties())
      if self.viewer is None:
        print("*** Failed to create viewer")
        quit()
    else:
      self.viewer = None    

    # load assets
    hand_asset_options = gymapi.AssetOptions()
    hand_asset_options.fix_base_link = True
    hand_asset_options.flip_visual_attachments = False
    hand_asset_options.use_mesh_materials = True
    hand_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    hand_asset_options.override_inertia = True
    hand_asset_options.override_com = True
    hand_asset_options.thickness = 0.0001    
    self.hand_asset = gym.load_asset(self.sim, 
                                     self.asset_root, 
                                     self.hand_asset_file, 
                                     hand_asset_options) 

    object_asset_options = gymapi.AssetOptions()
    object_asset_options.fix_base_link = False
    object_asset_options.flip_visual_attachments = False
    object_asset_options.override_inertia = True
    object_asset_options.override_com = True
    object_asset_options.thickness = 0.0001
    self.object_asset = gym.load_asset(self.sim, 
                                       self.asset_root, 
                                       self.object_asset_file, 
                                       object_asset_options)                               
                                       
    goal_asset_options = gymapi.AssetOptions()
    goal_asset_options.fix_base_link = True
    self.goal_asset = gym.load_asset(self.sim, 
                                     self.asset_root, 
                                     self.goal_asset_file, 
                                     goal_asset_options)      

    self.envs_per_row = int(math.sqrt(self.num_envs)) if(self.brake_configs is None) else self.brake_configs.shape[0]
    self.env_spacing = 0.3
    self.env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
    self.env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)                                                                                   
        
    # position the camera
    self.cam_pos = gymapi.Vec3(0.15,0.1,0.55)
    self.cam_target = gymapi.Vec3(0.2, 0.1, 0)
    
    self.envs = []
    self.actors = []
    self.actors_per_env = 3
    self.max_body_count = (gym.get_asset_rigid_body_count(self.hand_asset) +
                           gym.get_asset_rigid_body_count(self.object_asset) +
                           gym.get_asset_rigid_body_count(self.goal_asset))
    self.max_shape_count = (gym.get_asset_rigid_shape_count(self.hand_asset) +
                            gym.get_asset_rigid_shape_count(self.object_asset) +
                            gym.get_asset_rigid_shape_count(self.goal_asset))    
    
    self.hand_pose = gymapi.Transform()
    self.hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    self.hand_pose.r = gymapi.Quat(0,0,0,1)
    
    self.object_pose = gymapi.Transform()
    #self.object_pose.p = gymapi.Vec3(0.17, 0.045, 0.0)
    self.object_pose.p = gymapi.Vec3(0.172, 0.039, 0.0)
    self.object_pose.r = gymapi.Quat(0,0,0,1)    
    
    self.init_goal_pose = gymapi.Transform()
    self.init_goal_pose.p = gymapi.Vec3(0.17, -0.07, 0.0)
    self.init_goal_pose.r = gymapi.Quat(0,0,0,1)     
            
    for i in range(self.num_envs):
      # create env
      env = gym.create_env(self.sim, self.env_lower, self.env_upper, self.envs_per_row)
      self.envs.append(env)     

      gym.begin_aggregate(env, 
                          self.max_body_count, 
                          self.max_shape_count, 
                          True)
      hand_actor_handle = gym.create_actor(env, 
                                           self.hand_asset, 
                                           self.hand_pose, 
                                           "actor"+str(self.actors_per_env*i), 
                                           self.actors_per_env*i, 
                                           0)
      object_actor_handle = gym.create_actor(env, 
                                             self.object_asset, 
                                             self.object_pose, 
                                             "actor"+str(self.actors_per_env*i+1), 
                                             self.actors_per_env*i)        
      goal_actor_handle = gym.create_actor(env, 
                                           self.goal_asset, 
                                           self.init_goal_pose, 
                                           "actor"+str(self.actors_per_env*i+2), 
                                           self.actors_per_env*i+2)        
      gym.end_aggregate(env)
      
      gym.enable_actor_dof_force_sensors(env, hand_actor_handle)
      
      dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
      dof_states[0][0] = 0.0
      dof_states[1][0] = 0.298
      dof_states[2][0] = 0.35
      dof_states[3][0] = 0.8#0.9
      dof_states[4][0] = 0.29
      dof_states[5][0] = 0.361            
      gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)  
      
      dof_props = gym.get_actor_dof_properties(env, 
                                               hand_actor_handle)
      if self.brake_configs is None:
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["friction"].fill(0.1)
      else:
        brake_config_idx = i%(self.brake_configs.shape[0])
        dof_props["driveMode"][self.brake_configs[brake_config_idx]==1] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][self.brake_configs[brake_config_idx]==1] = 1000.0
        dof_props["damping"][self.brake_configs[brake_config_idx]==1] = 1.0
                
        dof_props["driveMode"][self.brake_configs[brake_config_idx]==0] = gymapi.DOF_MODE_EFFORT
        dof_props["friction"][self.brake_configs[brake_config_idx]==0] = 0.0325#0.01
        if self.brake_configs[brake_config_idx,2] == 0:
          dof_props["friction"][2] = 0.01
        if self.brake_configs[brake_config_idx,5] == 0:
          dof_props["friction"][5] = 0.01          
      
      gym.set_actor_dof_properties(env, 
                                   hand_actor_handle, 
                                   dof_props)  
      
      hand_rs_props = gym.get_actor_rigid_shape_properties(env, 
                                                           hand_actor_handle)
      #TODO: Filter for just finger tip shape
      for j in range(len(hand_rs_props)):
        # For some reason, can't allow finger tips to self collide with rest of robot
        if j == 4 or j == 8:
          hand_rs_props[j].filter = 1
        hand_rs_props[j].friction = 10.0
        hand_rs_props[j].rolling_friction = 10.0
        hand_rs_props[j].torsion_friction = 10.0                
      gym.set_actor_rigid_shape_properties(env, 
                                           hand_actor_handle, 
                                           hand_rs_props)
        
      cylinder_rs_props = gym.get_actor_rigid_shape_properties(env, 
                                                               object_actor_handle)
      for j in range(len(cylinder_rs_props)):
        cylinder_rs_props[j].friction = 0.1
        cylinder_rs_props[j].rolling_friction = 10.0
        cylinder_rs_props[j].torsion_friction = 10.0          
      gym.set_actor_rigid_shape_properties(env, 
                                           object_actor_handle, 
                                           cylinder_rs_props)        
      self.actors.append(hand_actor_handle)
      self.actors.append(object_actor_handle)  
      self.actors.append(goal_actor_handle)        

    gym.prepare_sim(self.sim)

    self.num_dofs = gym.get_sim_dof_count(self.sim)
    self.dof_per_env = gym.get_env_dof_count(self.envs[0])

    self._root_tensor = gym.acquire_actor_root_state_tensor(self.sim)
    self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

    self._rb_states = gym.acquire_rigid_body_state_tensor(self.sim)
    self.rb_states = gymtorch.wrap_tensor(self._rb_states)

    self._dof_states = gym.acquire_dof_state_tensor(self.sim)
    self.dof_states = gymtorch.wrap_tensor(self._dof_states)

    self._net_cf = gym.acquire_net_contact_force_tensor(self.sim)
    self.net_cf = gymtorch.wrap_tensor(self._net_cf)

    self.cylinder_idx = gym.get_actor_index(self.envs[0], self.actors[1], gymapi.DOMAIN_SIM)
    self.cylinder_pos = self.root_tensor[self.cylinder_idx::self.actors_per_env,:]   

    self.goal_idx = gym.get_actor_index(self.envs[0], self.actors[2], gymapi.DOMAIN_SIM)
    self.goal_pos = self.root_tensor[self.goal_idx::self.actors_per_env, :]

    self.r_finger_torque = FingerTorque(gym,
                                        "r_",
                                        self.rb_states,
                                        self.dof_states,
                                        self.envs,
                                        self.actors,
                                        self.device)                                                 

    self.l_finger_torque = FingerTorque(gym,
                                        "l_",
                                        self.rb_states,
                                        self.dof_states,
                                        self.envs,
                                        self.actors,
                                        self.device) 
    self.tendon_sensor = TendonSensor()                                                           

    self.l_tip_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], 
                                                               self.actors[0], 
                                                               'l_'+self.tendon_sensor.TIP_LINK, 
                                                               gymapi.DOMAIN_SIM)
    self.r_tip_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], 
                                                               self.actors[0], 
                                                               'r_'+self.tendon_sensor.TIP_LINK, 
                                                               gymapi.DOMAIN_SIM)

    bodies_per_env = gym.get_env_rigid_body_count(self.envs[0])
    self.l_tip_contact = self.net_cf[self.l_tip_body_idx::bodies_per_env]
    self.r_tip_contact = self.net_cf[self.r_tip_body_idx::bodies_per_env]               

    if self.brake_configs is not None:
      # Tell braked joints to stay where they are
      pos_targets = torch.tensor([dof_states[0][0],
                                  dof_states[1][0],
                                  dof_states[2][0],
                                  dof_states[3][0],
                                  dof_states[4][0],
                                  dof_states[5][0]], dtype=torch.float32, device=self.device).repeat(self.num_envs)
      self.gym.set_dof_position_target_tensor(self.sim,
                                              gymtorch.unwrap_tensor(pos_targets))
      
    # Populate initial state of sim
    self.gym.simulate(self.sim)
    self.gym.fetch_results(self.sim, True)
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_dof_state_tensor(self.sim)    
    self.gym.refresh_net_contact_force_tensor(self.sim)
    
    self.prev_finger_dofs = torch.zeros((self.num_envs,6),
                                        dtype=torch.float32, 
                                        device=self.device)
    self.prev_finger_dofs[:,0] = self.l_finger_torque.finger_dof1_pos[:]
    self.prev_finger_dofs[:,1] = self.l_finger_torque.finger_dof2_pos[:]
    self.prev_finger_dofs[:,2] = self.l_finger_torque.finger_dof3_pos[:]
    self.prev_finger_dofs[:,3] = self.r_finger_torque.finger_dof1_pos[:]
    self.prev_finger_dofs[:,4] = self.r_finger_torque.finger_dof2_pos[:]
    self.prev_finger_dofs[:,5] = self.r_finger_torque.finger_dof3_pos[:]                                          
    
    if self.viewer is not None:
      self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
      self.gym.step_graphics(self.sim)
      self.gym.draw_viewer(self.viewer, self.sim, True)
    self.gym.sync_frame_time(self.sim)      
      
  def set_goal_pose(self, goal_x, goal_y):
    self.goal_pos[:,0] = goal_x
    self.goal_pos[:,1] = goal_y
    self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)      

  # Get the state of the sim
  # Returns tensor of size (n_envs, 6+6+3+4+3+3=25)
  # First 6 columns represent the robot's joint positions
  # Next 6 columns represent the robot's joint velocities
  # Next 3 columns are (x,y,z) position of cylinder
  # Next 4 represent the quaternion orientation of the cylinder
  # Next 3 represent the linear velocity of cylinder
  # Next 3 represent the angular velocity of cylinder
  def get_sim_state(self):                  
    return torch.cat([self.l_finger_torque.finger_dof1_pos[:,None],
                      self.l_finger_torque.finger_dof2_pos[:,None],
                      self.l_finger_torque.finger_dof3_pos[:,None],
                      self.r_finger_torque.finger_dof1_pos[:,None],
                      self.r_finger_torque.finger_dof2_pos[:,None],
                      self.r_finger_torque.finger_dof3_pos[:,None],
                      (self.l_finger_torque.finger_dof1_pos[:] - self.prev_finger_dofs[:,0])[:,None]/self.dt,
                      (self.l_finger_torque.finger_dof2_pos[:] - self.prev_finger_dofs[:,1])[:,None]/self.dt,
                      (self.l_finger_torque.finger_dof3_pos[:] - self.prev_finger_dofs[:,2])[:,None]/self.dt,
                      (self.r_finger_torque.finger_dof1_pos[:] - self.prev_finger_dofs[:,3])[:,None]/self.dt,
                      (self.r_finger_torque.finger_dof2_pos[:] - self.prev_finger_dofs[:,4])[:,None]/self.dt,
                      (self.r_finger_torque.finger_dof3_pos[:] - self.prev_finger_dofs[:,5])[:,None]/self.dt,
                      self.cylinder_pos], 
                     dim=1)      
                     
  # Set the state of the sim
  # state should be a tensor of size (n_envs, 6+6+3+4+3+3=25) or( 1,25)
  # See get_sim_state() for format of state
  def set_sim_state(self, state):                    
    if(len(state.shape) == 1):
      state = state[None] # Add another axis
    assert(state.shape[1] == 25)
  
    self.prev_finger_dofs[:,:] = state[:,0:6] - self.dt*state[:,6:12]
  
    self.dof_states[self.l_finger_torque.finger_dof1_idx::self.dof_per_env,0] = state[:,0]
    self.dof_states[self.l_finger_torque.finger_dof2_idx::self.dof_per_env,0] = state[:,1]
    self.dof_states[self.l_finger_torque.finger_dof3_idx::self.dof_per_env,0] = state[:,2]        
    self.dof_states[self.r_finger_torque.finger_dof1_idx::self.dof_per_env,0] = state[:,3]
    self.dof_states[self.r_finger_torque.finger_dof2_idx::self.dof_per_env,0] = state[:,4]
    self.dof_states[self.r_finger_torque.finger_dof3_idx::self.dof_per_env,0] = state[:,5]
    self.dof_states[self.l_finger_torque.finger_dof1_idx::self.dof_per_env,1] = state[:,6]
    self.dof_states[self.l_finger_torque.finger_dof2_idx::self.dof_per_env,1] = state[:,7]
    self.dof_states[self.l_finger_torque.finger_dof3_idx::self.dof_per_env,1] = state[:,8]        
    self.dof_states[self.r_finger_torque.finger_dof1_idx::self.dof_per_env,1] = state[:,9]
    self.dof_states[self.r_finger_torque.finger_dof2_idx::self.dof_per_env,1] = state[:,10]
    self.dof_states[self.r_finger_torque.finger_dof3_idx::self.dof_per_env,1] = state[:,11]      
    
    self.root_tensor[self.cylinder_idx::self.actors_per_env,:] = state[:,12:]

    if self.brake_configs is not None:
      # Tell braked joints to stay where they are
      pos_targets = torch.zeros((self.num_envs, 6),dtype=torch.float32,device=self.device)
      pos_targets[:,:] = state[:,0:6]
      pos_targets = pos_targets.flatten()
      self.gym.set_dof_position_target_tensor(self.sim,
                                              gymtorch.unwrap_tensor(pos_targets))
                                              
    self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)
    self.gym.set_dof_state_tensor(self.sim, self._dof_states)    
    
  # state should be a tensor of size (n_envs, 6+6+3+4+3+3=25) or( 1,25)
  # See get_sim_state() for format of state
  # action should be a tenor of size (n_envs, 2)
  def step(self, state, action, sync=False, render=False):
    #print(self.dof_states[0:6,0])
    # Add check that passed state is equal to current state?
    
    l_tendon_torques = -1*self.l_finger_torque.compute_tendon_torques(action[:,0])
    l_spring_torques = self.l_finger_torque.compute_spring_torques()
        
    r_tendon_torques = self.r_finger_torque.compute_tendon_torques(action[:,1])
    r_spring_torques = self.r_finger_torque.compute_spring_torques()    

    intrinsic_torques = (r_tendon_torques + r_spring_torques +
                         l_tendon_torques + l_spring_torques)

    self.gym.set_dof_actuation_force_tensor(self.sim, 
                                            gymtorch.unwrap_tensor(intrinsic_torques))
    
    self.prev_finger_dofs[:,0] = self.l_finger_torque.finger_dof1_pos[:]
    self.prev_finger_dofs[:,1] = self.l_finger_torque.finger_dof2_pos[:]
    self.prev_finger_dofs[:,2] = self.l_finger_torque.finger_dof3_pos[:]
    self.prev_finger_dofs[:,3] = self.r_finger_torque.finger_dof1_pos[:]
    self.prev_finger_dofs[:,4] = self.r_finger_torque.finger_dof2_pos[:]
    self.prev_finger_dofs[:,5] = self.r_finger_torque.finger_dof3_pos[:]    
                                            
    # step the physics
    self.gym.simulate(self.sim)
    self.gym.fetch_results(self.sim, True)

    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_dof_state_tensor(self.sim)    
    self.gym.refresh_net_contact_force_tensor(self.sim)
    
    if (self.viewer is not None) and render:
      # update the viewer
      self.gym.step_graphics(self.sim)
      self.gym.draw_viewer(self.viewer, self.sim, True)
    
    if sync:
      self.gym.sync_frame_time(self.sim)
    
    return self.get_sim_state()  
    
def main():    
  # initialize gym
  gym = gymapi.acquire_gym()

  args = gymutil.parse_arguments(
    description="BATA Hand Sim")

  if args.pipeline != 'cpu':
    print('ERROR:Please run with [--pipeline cpu], gpu pipeline has problems with setting state')
    return

  # set up the env grid
  num_envs = 300
  brake_configs = torch.tensor([[0,0,0,0,0,0]], dtype=torch.int32, device='cpu')
  device = args.sim_device if args.use_gpu_pipeline else 'cpu'
  
  headless = False
  bhs = BATAHandSim(gym, num_envs, brake_configs, args, headless=headless)
  cur_state = bhs.get_sim_state()
  cur_action = torch.zeros((1,2),dtype=torch.float32,device=device)
  while True:
    cur_state = bhs.step(cur_state, cur_action, True, True)    
if __name__ == '__main__':
  main()

                                                                                                    
