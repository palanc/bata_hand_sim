"""

Model Test
------------
- Simulates a model and provides a keyboard interface for applying force to
- simulated tendons
"""
import rospy
import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from torch import linalg as LA
from bata_hand_sim.finger_torque import FingerTorque
from bata_hand_sim.tendon_sensor import TendonSensor

class BATAHandSim(object):
  asset_root = "/home/patrick/bata_hand_ws/devel/share/bata_hand_description"
  hand_asset_file = "robots/bata_hand.urdf"
  object_asset_file = "robots/cylinder.urdf"
  goal_asset_file = "robots/short_cylinder.urdf"
  
  def __init__(self, 
               gym,
               num_envs,
               args,
               headless=False):
  
    self.gym = gym
    self.num_envs = num_envs
    
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
    hand_asset_options.vhacd_enabled = True
    hand_asset_options.vhacd_params = gymapi.VhacdParams()
    #hand_asset_options.vhacd_params.alpha = 0.05
    #hand_asset_options.vhacd_params.beta = 0.05
    #hand_asset_options.vhacd_params.convex_hull_downsampling = 4
    #hand_asset_options.vhacd_params.max_num_vertices_per_ch = 64
    #hand_asset_options.vhacd_params.mode = 0
    #hand_asset_options.vhacd_params.pca = 1
    #hand_asset_options.vhacd_params.resolution = 5000000
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
    object_asset_options.vhacd_enabled = True
    object_asset_options.vhacd_params = gymapi.VhacdParams()
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

    self.envs_per_row = 6
    self.env_spacing = 0.5
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
                            
    # add actor
    self.hand_pose = gymapi.Transform()
    self.hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    self.hand_pose.r = gymapi.Quat(0,0,0,1)
    
    self.object_pose = gymapi.Transform()
    #self.object_pose.p = gymapi.Vec3(0.17, 0.07, 0.0)
    #self.object_pose.p = gymapi.Vec3(0.17, 0.00, 0.0)
    self.object_pose.p = gymapi.Vec3(0.17, 0.045, 0.0)
    self.object_pose.r = gymapi.Quat(0,0,0,1)    
    
    self.goal_pose = gymapi.Transform()
    self.goal_pose.p = gymapi.Vec3(0.17, -0.07, 0.0)
    self.goal_pose.r = gymapi.Quat(0,0,0,1)                             
                                                  
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
                                           self.goal_pose, 
                                           "actor"+str(self.actors_per_env*i+2), 
                                           self.actors_per_env*i+2)        
      gym.end_aggregate(env)
      
      dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
      #dof_states[0][0] = 0.78
      #dof_states[3][0] = 1.35
      
      dof_states[0][0] = 0.0
      dof_states[1][0] = 0.298
      dof_states[2][0] = 0.35
      dof_states[3][0] = 0.9
      dof_states[4][0] = 0.29
      dof_states[5][0] = 0.361      
      
      gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)
      
      self.actors.append(hand_actor_handle)
      self.actors.append(object_actor_handle)  
      self.actors.append(goal_actor_handle)                   

    for i in range(self.num_envs):
      dof_props = gym.get_actor_dof_properties(self.envs[i], 
                                               self.actors[self.actors_per_env*i])
      dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
      dof_props["stiffness"].fill(0.0)
      dof_props["damping"].fill(0.0)
      dof_props["friction"].fill(1.0)
      gym.set_actor_dof_properties(self.envs[i], 
                                   self.actors[self.actors_per_env*i], 
                                   dof_props)  
      
      hand_rs_props = gym.get_actor_rigid_shape_properties(self.envs[i], 
                                                           self.actors[self.actors_per_env*i+0])
      #TODO: Filter for just finger tip shape
      #TODO: Does this actually affect anything?
      for j in range(len(hand_rs_props)):
        hand_rs_props[j].friction = 10.0
        hand_rs_props[j].rolling_friction = 10.0
        hand_rs_props[j].torsion_friction = 10.0                
        
      cylinder_rs_props = gym.get_actor_rigid_shape_properties(self.envs[i], 
                                                               self.actors[self.actors_per_env*i+1])
      for j in range(len(cylinder_rs_props)):
        cylinder_rs_props[j].friction = 0.1
        cylinder_rs_props[j].rolling_friction = 10.0
        cylinder_rs_props[j].torsion_friction = 10.0          
      gym.set_actor_rigid_shape_properties(self.envs[i], self.actors[3*i+1], cylinder_rs_props)
  
  
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

    self.r_finger_torque = FingerTorque(gym,
                                        "r_",
                                        self.rb_states,
                                        self.dof_states,
                                        self.envs,
                                        self.actors,
                                        self.device)
                                   
    self.r_tendon_sensor = TendonSensor(gym,
                                        "r_",
                                        self.rb_states,
                                        self.dof_states,
                                        self.envs, 
                                        self.actors, 
                                        sim_params.dt,
                                        self.device)                

    self.l_finger_torque = FingerTorque(gym,
                                        "l_",
                                        self.rb_states,
                                        self.dof_states,
                                        self.envs,
                                        self.actors,
                                        self.device)
                                       
    self.l_tendon_sensor = TendonSensor(gym,
                                        "l_",
                                        self.rb_states,
                                        self.dof_states,
                                        self.envs, 
                                        self.actors, 
                                        sim_params.dt,
                                        self.device)                

    self.l_tip_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], 
                                                               self.actors[0], 
                                                               'l_'+self.l_tendon_sensor.TIP_LINK, 
                                                               gymapi.DOMAIN_SIM)
    self.r_tip_body_idx = self.gym.find_actor_rigid_body_index(self.envs[0], 
                                                               self.actors[0], 
                                                               'r_'+self.r_tendon_sensor.TIP_LINK, 
                                                               gymapi.DOMAIN_SIM)
                                                                                                                                                                                            
    self.l_tip_contact = self.net_cf[self.l_tip_body_idx::self.l_tendon_sensor.bodies_per_env]
    self.r_tip_contact = self.net_cf[self.r_tip_body_idx::self.r_tendon_sensor.bodies_per_env]
    
    if self.viewer is not None:
      gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
      gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "l_inc_force")
      gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "l_dec_force")
      gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "r_inc_force")
      gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "r_dec_force")

    self.force_del = 0.2
    self.l_force_sp = 0.0
    self.r_force_sp = 0.0

    # Populate initial state of sim
    self.gym.simulate(self.sim)
    self.gym.fetch_results(self.sim, True)
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_dof_state_tensor(self.sim)    
    self.gym.refresh_net_contact_force_tensor(self.sim)
    if self.viewer is not None:
      self.gym.step_graphics(self.sim)
      self.gym.draw_viewer(self.viewer, self.sim, True)
    self.gym.sync_frame_time(self.sim)

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
                      self.l_finger_torque.finger_dof1_vel[:,None],
                      self.l_finger_torque.finger_dof2_vel[:,None],
                      self.l_finger_torque.finger_dof3_vel[:,None],
                      self.r_finger_torque.finger_dof1_vel[:,None],
                      self.r_finger_torque.finger_dof2_vel[:,None],
                      self.r_finger_torque.finger_dof3_vel[:,None],                      
                      self.cylinder_pos], dim=1)
  
  # Set the state of the sim
  # state should be a tensor of size (n_envs, 6+6+3+4+3+3=25) or( 1,25)
  # See get_sim_state() for format of state
  def set_sim_state(self, state):                    
    if(len(state.shape) == 1):
      state = state[None] # Add another axis
    assert(state.shape[1] == 25)
  
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
    
    self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)
    self.gym.set_dof_state_tensor(self.sim, self._dof_states)

  # state should be a tensor of size (n_envs, 6+6+3+4+3+3=25) or( 1,25)
  # See get_sim_state() for format of state
  # action should be a tenor of size (n_envs, 2)
  def step(self, state, action, sync=False, render=False):
    # Add check that passed state is equal to current state?
    
    l_tendon_torques = -1*self.l_finger_torque.compute_tendon_torques(action[:,0])
    l_spring_torques = self.l_finger_torque.compute_spring_torques()
        
    r_tendon_torques = self.r_finger_torque.compute_tendon_torques(action[:,1])
    r_spring_torques = self.r_finger_torque.compute_spring_torques()    

    intrinsic_torques = (r_tendon_torques + r_spring_torques +
                         l_tendon_torques + l_spring_torques)


    self.gym.set_dof_actuation_force_tensor(self.sim, 
                                            gymtorch.unwrap_tensor(intrinsic_torques))
                                            
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
  
         
  def simulate(self):
    assert(self.viewer is not None)
    while not self.gym.query_viewer_has_closed(self.viewer):

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
    
        # Check for keyboard input
        for evt in self.gym.query_viewer_action_events(self.viewer):
          if evt.action == "l_inc_force" and evt.value > 0:
            self.l_force_sp = self.l_force_sp+self.force_del
          elif evt.action == "l_dec_force" and evt.value > 0:
            self.l_force_sp = self.l_force_sp-self.force_del
          elif evt.action == "r_inc_force" and evt.value > 0:
            self.r_force_sp = self.r_force_sp+self.force_del
          elif evt.action == "r_dec_force" and evt.value > 0:
            self.r_force_sp = self.r_force_sp-self.force_del        
            
        r_tendon_force = torch.tensor([self.r_force_sp], dtype=torch.float32, device=self.device)     
        r_tendon_torques = self.r_finger_torque.compute_tendon_torques(r_tendon_force)
        r_spring_torques = self.r_finger_torque.compute_spring_torques()
        r_tendon_lengths_rb = self.r_tendon_sensor.compute_tendon_length_dv_rb()
        r_tendon_lengths_analytic = self.r_tendon_sensor.compute_tendon_length_dv_analytic()
        r_tendon_velocities_analytic = self.r_tendon_sensor.compute_tendon_velocity()

        l_tendon_force = torch.tensor([self.l_force_sp], dtype=torch.float32, device=self.device)
        l_tendon_torques = -1*self.l_finger_torque.compute_tendon_torques(l_tendon_force)
        l_spring_torques = self.l_finger_torque.compute_spring_torques()
        l_tendon_lengths_rb = self.l_tendon_sensor.compute_tendon_length_dv_rb()
        l_tendon_lengths_analytic = self.l_tendon_sensor.compute_tendon_length_dv_analytic()
        l_tendon_velocities_analytic = self.l_tendon_sensor.compute_tendon_velocity()

        #print('Force set to %f %f'%(self.l_force_sp,self.r_force_sp))
        #print("Torques")
        #print(l_tendon_torques[0:self.dof_per_env])
        #print(l_spring_torques[0:self.dof_per_env])
        #print("RB: %f"%l_tendon_lengths_rb[0].item())
        #print("Analytic: %f"%l_tendon_lengths_analytic[0].item())
        #print("Diff: %f"%(l_tendon_lengths_rb[0]-l_tendon_lengths_analytic[0]).item())
        #print("Vel: %f"%(l_tendon_velocities_analytic[0].item()))
        #print("Contact: %f %f"%(torch.norm(self.l_tip_contact[0]).item(),
        #                        torch.norm(self.r_tip_contact[0]).item()))
        
        intrinsic_torques = r_tendon_torques + r_spring_torques
        intrinsic_torques += l_tendon_torques + l_spring_torques

        self.gym.set_dof_actuation_force_tensor(self.sim, 
                                                gymtorch.unwrap_tensor(intrinsic_torques))

        # update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

    print("Done")
    self.gym.destroy_viewer(self.viewer)
    self.gym.destroy_sim(self.sim)
    
if __name__ == '__main__':

  # initialize gym
  gym = gymapi.acquire_gym()

  # set up the env grid
  num_envs = 3
  
  bhs = BATAHandSim(gym, num_envs)
  bhs.simulate()
