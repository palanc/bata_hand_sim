#!/usr/bin/env python

import rospy
from isaacgym import gymapi, gymutil, gymtorch
from bata_hand_sim.bata_hand_sim import BATAHandSim
import torch

def main():
  
  rospy.init_node('simulate')
    
  # initialize gym
  gym = gymapi.acquire_gym()

  real_robot_arg = {}
  real_robot_arg["name"] = '--real_robot'
  real_robot_arg["type"] = int
  real_robot_arg["default"] = 0
  real_robot_arg["help"] = "Use sim or real robot"

  args = gymutil.parse_arguments(
    description="Simulate",
    custom_parameters=[real_robot_arg])
  device = args.sim_device if args.use_gpu_pipeline else 'cpu'
  #if args.pipeline != 'cpu':
  #  print('ERROR:Please run with [--pipeline cpu], gpu pipeline has problems with setting state')
  #  return

  # set up the env grid
  num_envs = 18
  headless = False
  brake_configs = torch.tensor([[0,1,1,0,1,1],
                                [0,1,1,1,0,1],
                                [0,1,1,1,1,0],
                                [1,0,1,0,1,1],
                                [1,0,1,1,0,1],
                                [1,0,1,1,1,0],
                                [1,1,0,0,1,1],
                                [1,1,0,1,0,1],
                                [1,1,0,1,1,0]], dtype=torch.int32, device='cpu')
  bhs = BATAHandSim(gym, num_envs, brake_configs, args, headless=headless)

  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_W, "l_inc_force")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_S, "l_dec_force")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_UP, "r_inc_force")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_DOWN, "r_dec_force")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_0, "toggle_brake_0")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_1, "toggle_brake_1")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_2, "toggle_brake_2")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_3, "toggle_brake_3")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_4, "toggle_brake_4")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_5, "toggle_brake_5")  
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_6, "toggle_brake_6")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_7, "toggle_brake_7")
  gym.subscribe_viewer_keyboard_event(bhs.viewer, gymapi.KEY_8, "toggle_brake_8") 

  force_del = 0.2
  l_force_sp = 0.0
  r_force_sp = 0.0
 
  cur_state = bhs.get_sim_state()
  cur_action = torch.zeros((1,2),dtype=torch.float32,device=device)
 
  while True:
    # Check for keyboard input
    for evt in gym.query_viewer_action_events(bhs.viewer):
      if evt.action == "l_inc_force" and evt.value > 0:
        l_force_sp = l_force_sp+force_del
        print("%f %f"%(l_force_sp, r_force_sp))
      elif evt.action == "l_dec_force" and evt.value > 0:
        l_force_sp = l_force_sp-force_del
        print("%f %f"%(l_force_sp, r_force_sp))        
      elif evt.action == "r_inc_force" and evt.value > 0:
        r_force_sp = r_force_sp+force_del
        print("%f %f"%(l_force_sp, r_force_sp))        
      elif evt.action == "r_dec_force" and evt.value > 0:
        r_force_sp = r_force_sp-force_del
        print("%f %f"%(l_force_sp, r_force_sp))
      elif "toggle_brake_" in evt.action and evt.value > 0:  
        gym.viewer_camera_look_at(bhs.viewer, 
                                  None, 
                                  bhs.cam_pos+gymapi.Vec3(int(evt.action[-1])*0.6,0.0,0.0), 
                                  bhs.cam_target+gymapi.Vec3(int(evt.action[-1])*0.6,0.0,0.0)) 
     
    cur_action[0,0] = l_force_sp
    cur_action[0,1] = r_force_sp
    cur_state = bhs.step(cur_state, cur_action, True, True)
  
if __name__ == '__main__':
  main()
