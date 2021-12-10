"""

Model Test
------------
- Simulates a model and provides a keyboard interface for applying force to
- simulated tendons
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from torch import linalg as LA
from finger_torque import FingerTorque
from tendon_sensor import TendonSensor

asset_root = "/home/patrick/bata_hand_ws/devel/share/bata_hand_description"#"../../assets"
hand_asset_file = "robots/bata_hand.urdf"
object_asset_file = "robots/cylinder.urdf"
goal_asset_file = "robots/short_cylinder.urdf"
force_del = 0.2

args = gymutil.parse_arguments(
    description="Simulate: Simulate a model")

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.physx.contact_offset = 0.001
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 0.001
plane_params.dynamic_friction = 0.001
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "l_inc_force")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "l_dec_force")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "r_inc_force")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "r_dec_force")

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
print("Loading asset '%s' from '%s'" % (hand_asset_file, asset_root))
hand_asset = gym.load_asset(sim, asset_root, hand_asset_file, hand_asset_options)

object_asset_options = gymapi.AssetOptions()
object_asset_options.fix_base_link = False
object_asset_options.flip_visual_attachments = False
object_asset_options.override_inertia = True
object_asset_options.override_com = True
object_asset_options.thickness = 0.0001
#object_asset_options.vhacd_enabled = True
#object_asset_options.vhacd_params = gymapi.VhacdParams()
object_asset = gym.load_asset(sim, asset_root, object_asset_file, object_asset_options)

goal_asset_options = gymapi.AssetOptions()
goal_asset_options.fix_base_link = True
goal_asset = gym.load_asset(sim, asset_root, goal_asset_file, goal_asset_options)

# set up the env grid
num_envs = 3
num_per_row = 6
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0.15,0.1,0.55)
cam_target = gymapi.Vec3(0.2, 0.1, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    hand_pose.r = gymapi.Quat(0,0,0,1)
    hand_actor_handle = gym.create_actor(env, hand_asset, hand_pose, "actor"+str(3*i), 3*i, 0)
    actor_handles.append(hand_actor_handle)
    
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(0.17, 0.07, 0.0)
    object_pose.r = gymapi.Quat(0,0,0,1)
    object_actor_handle = gym.create_actor(env, object_asset, object_pose, "actor"+str(3*i+1), 3*i)
    actor_handles.append(object_actor_handle)    
    
    goal_pose = gymapi.Transform()
    goal_pose.p = gymapi.Vec3(0.17, -0.07, 0.0)
    goal_pose.r = gymapi.Quat(0,0,0,1)
    goal_actor_handle = gym.create_actor(env, goal_asset, goal_pose, "actor"+str(3*i+2), 3*i+2)
    actor_handles.append(goal_actor_handle) 

assert(num_envs ==len(envs))

for i in range(num_envs):
  dof_props = gym.get_actor_dof_properties(envs[i], actor_handles[3*i])
  dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
  dof_props["stiffness"].fill(0.0)
  dof_props["damping"].fill(0.0)
  #dof_props["damping"].fill(1.0)
  dof_props["friction"].fill(1.0)
  gym.set_actor_dof_properties(envs[i], actor_handles[3*i], dof_props)  
  
  rs_props = gym.get_actor_rigid_shape_properties(envs[i], actor_handles[3*i+1])
  for j in range(len(rs_props)):
    rs_props[j].friction = 0.1
  gym.set_actor_rigid_shape_properties(envs[i], actor_handles[3*i+1], rs_props)
  
  
gym.prepare_sim(sim)

l_force_sp = 0.0
r_force_sp = 0.0
num_dofs = gym.get_sim_dof_count(sim)
dof_per_env = gym.get_env_dof_count(envs[0])

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

r_finger_torque = FingerTorque(gym,
                               "r_",
                               rb_states,
                               dof_states,
                               envs,
                               actor_handles)
                                   
r_tendon_sensor = TendonSensor(gym,
                               "r_",
                               rb_states,
                               dof_states,
                               envs, 
                               actor_handles, 
                               sim_params.dt)                

l_finger_torque = FingerTorque(gym,
                               "l_",
                               rb_states,
                               dof_states,
                               envs,
                               actor_handles)
                                   
l_tendon_sensor = TendonSensor(gym,
                               "l_",
                               rb_states,
                               dof_states,
                               envs, 
                               actor_handles, 
                               sim_params.dt)                

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    # Check for keyboard input
    for evt in gym.query_viewer_action_events(viewer):
      if evt.action == "l_inc_force" and evt.value > 0:
        l_force_sp = l_force_sp+force_del
      elif evt.action == "l_dec_force" and evt.value > 0:
        l_force_sp = l_force_sp-force_del
      elif evt.action == "r_inc_force" and evt.value > 0:
        r_force_sp = r_force_sp+force_del
      elif evt.action == "r_dec_force" and evt.value > 0:
        r_force_sp = r_force_sp-force_del        
        
    r_tendon_torques = r_finger_torque.compute_tendon_torques(torch.tensor(r_force_sp, dtype=torch.float32, device="cuda:0"))
    r_spring_torques = r_finger_torque.compute_spring_torques()
    r_tendon_lengths_rb = r_tendon_sensor.compute_tendon_length_dv_rb()
    r_tendon_lengths_analytic = r_tendon_sensor.compute_tendon_length_dv_analytic()
    r_tendon_velocities_analytic = r_tendon_sensor.compute_tendon_velocity()

    l_tendon_torques = -1*l_finger_torque.compute_tendon_torques(torch.tensor(l_force_sp, dtype=torch.float32, device="cuda:0"))
    l_spring_torques = l_finger_torque.compute_spring_torques()
    l_tendon_lengths_rb = l_tendon_sensor.compute_tendon_length_dv_rb()
    l_tendon_lengths_analytic = l_tendon_sensor.compute_tendon_length_dv_analytic()
    l_tendon_velocities_analytic = l_tendon_sensor.compute_tendon_velocity()

    print('Force set to %f %f'%(l_force_sp,r_force_sp))
    print("Torques")
    print(l_tendon_torques[0:dof_per_env])
    print(l_spring_torques[0:dof_per_env])
    print("RB: %f"%l_tendon_lengths_rb[0].item())
    print("Analytic: %f"%l_tendon_lengths_analytic[0].item())
    print("Diff: %f"%(l_tendon_lengths_rb[0]-l_tendon_lengths_analytic[0]).item())
    print("Vel: %f"%(l_tendon_velocities_analytic[0].item()))
    
    intrinsic_torques = r_tendon_torques + r_spring_torques
    intrinsic_torques += l_tendon_torques + l_spring_torques
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(intrinsic_torques))

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
