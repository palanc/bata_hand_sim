import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from torch import linalg as LA
import time

class BATAHandSim(object):
  asset_root = "/home/patrick/bata_hand_ws/devel/share/bata_hand_description"
  hand_asset_file = "robots/bata_hand_isaac.urdf"
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

    self.sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
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
    #hand_asset_options.vhacd_enabled = True
    #hand_asset_options.vhacd_params = gymapi.VhacdParams()
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

    self.envs_per_row = 6
    self.env_spacing = 0.5
    self.env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
    self.env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

    # position the camera
    self.cam_pos = gymapi.Vec3(0.15,0.1,0.55)
    self.cam_target = gymapi.Vec3(0.2, 0.1, 0)

    self.envs = []
    self.actors = []
                            
    # add actor
    self.hand_pose = gymapi.Transform()
    self.hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    self.hand_pose.r = gymapi.Quat(0,0,0,1)                           
                                                  
    for i in range(self.num_envs):
      # create env
      env = gym.create_env(self.sim, self.env_lower, self.env_upper, self.envs_per_row)
      self.envs.append(env)     

      hand_actor_handle = gym.create_actor(env, 
                                           self.hand_asset, 
                                           self.hand_pose, 
                                           "actor"+str(i), 
                                           i, 
                                           0)
      
      self.actors.append(hand_actor_handle)
                
    '''
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
    '''  
  
  
    gym.prepare_sim(self.sim)

    
    self.num_dofs = gym.get_sim_dof_count(self.sim)
    self.dof_per_env = gym.get_env_dof_count(self.envs[0])

    self._dof_states = gym.acquire_dof_state_tensor(self.sim)
    self.dof_states = gymtorch.wrap_tensor(self._dof_states)

    '''
    self._root_tensor = gym.acquire_actor_root_state_tensor(self.sim)
    self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

    self._rb_states = gym.acquire_rigid_body_state_tensor(self.sim)
    self.rb_states = gymtorch.wrap_tensor(self._rb_states)

    self._net_cf = gym.acquire_net_contact_force_tensor(self.sim)
    self.net_cf = gymtorch.wrap_tensor(self._net_cf)
    '''

    if self.viewer is not None:
      gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
    
  def simulate(self):
    assert(self.viewer is not None)
    zeros = torch.zeros((6*self.num_envs,2),dtype=torch.float32, device=self.device)
    zeros[::6,0] = 0.7854
    zeros[3::6,0] = 0.7854    
    steps = 1
    
    start = time.time()
    while not self.gym.query_viewer_has_closed(self.viewer):
        
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)
        if steps % 60 == 0:
          self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(zeros))
          #self.gym.set_dof_state_tensor(self.sim,self._dof_states)
          print("Elapsed: %f"%(time.time()-start))
          start = time.time()

        # update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)
        steps += 1

    print("Done")
    self.gym.destroy_viewer(self.viewer)
    self.gym.destroy_sim(self.sim)    
  
    
if __name__ == '__main__':
  # initialize gym
  gym = gymapi.acquire_gym()

# parse arguments
  args = gymutil.parse_arguments(
    description="State test")
  
  # set up the env grid
  num_envs = 100
  headless = False
  bhs = BATAHandSim(gym, num_envs, args, headless=headless)
  bhs.simulate()

