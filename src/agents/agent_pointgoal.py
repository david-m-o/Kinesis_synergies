# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import os
import torch
import numpy as np
import logging
from typing import Tuple

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_im import AgentIM
from src.learning.learning_utils import to_test, to_cpu
from src.env.myolegs_pointgoal import MyoLegsPointGoal

logger = logging.getLogger(__name__)

class AgentPointGoal(AgentIM):
    """
    AgentPointGoal is a specialized reinforcement learning agent for humanoid environments,
    adapting AgentIM for a point goal task, in accordance with the MyoLegsPointGoal environment.
    """

    def __init__(self, cfg, dtype, device, training = True, checkpoint_epoch = 0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def setup_env(self):
        self.env = MyoLegsPointGoal(self.cfg)
        logger.info("PointGoal environment initialized.")

    def eval_policy(self, epoch = 0, dump = False, runs = 100):
        logger.info("Starting policy evaluation on target goal reaching.")
        self.env.start_eval(im_eval=True)

        to_test(*self.sample_modules)

        success_list = []
        last_reward_list = []
        frame_coverage_list = []


        with to_cpu(*self.sample_modules), torch.no_grad():
            for i in range(runs):
                result, last_reward, frame_coverage = self.eval_single_thread()
                print(f"Episode {i} result: {result}")
                success_list.append(result)
                last_reward_list.append(last_reward)
                frame_coverage_list.append(frame_coverage)


        success_rate = np.mean(success_list)
        mean_frame_coverage = np.mean(frame_coverage_list)
        logger.info(f"Policy evaluation success rate: {success_rate}")
        logger.info(f"Policy evaluation frame coverage: {mean_frame_coverage}")
        
        save_data = {
                    "success_list": success_list,
                    "pca_type": self.cfg.run.pca_type,
                    "csi_pca_components": self.cfg.run.csi_pca_components,
                }
        

        np.save(f"/home/david/LAB/Kinesis/csi/sucess_rate/results_{self.cfg.run.algorithm}_{self.cfg.run.pca_type}_{self.cfg.run.csi_pca_components}.npy", save_data)

        #ADDED BY DAVID
        if self.env.recording_biomechanics:
            breakpoint()
            print("Saving recorded biomechanics data.")
            #ADDED BY DAVID 
            muscle_controls = np.array(self.env.muscle_controls)
            policy_outputs = np.array(self.env.policy_outputs)

            np.save('/home/david/LAB/Kinesis/csi/recordings/muscle_controls_POINTGOAL.npy', muscle_controls) 
            np.save('/home/david/LAB/Kinesis/csi/recordings/policy_outputs_POINTGOAL.npy', policy_outputs) 

            

        return success_rate, last_reward_list
    
    def eval_policy_csi(self, epoch = 0, dump = False, runs = 100):
        logger.info(f"Starting policy evaluation on target goal reaching with CSI ({self.cfg.run.algorithm}).")
        self.env.start_eval(im_eval=True)

        to_test(*self.sample_modules)


        for num_components in range(1, 81, 1):
            self.env.cfg.run.csi_pca_components = num_components
            for trial in range(3):
                print(f"Running trial {trial} with {self.env.cfg.run.csi_pca_components} components ({self.cfg.run.pca_type}, {self.cfg.run.algorithm}).")
                
                success_list = []
                last_reward_list = []
                frame_coverage_list = []
                
                #print(f"Number of components = {self.env.cfg.run.csi_pca_components}")
                with to_cpu(*self.sample_modules), torch.no_grad():
                    for i in range(runs):
                        result, last_reward, frame_coverage = self.eval_single_thread()
                        print(f"Episode {i} result: {result}")
                        success_list.append(result)
                        last_reward_list.append(last_reward)
                        frame_coverage_list.append(frame_coverage)
                        

                success_rate = np.mean(success_list)
                mean_frame_coverage = np.mean(frame_coverage_list)
                logger.info(f"Policy evaluation with CSI ({self.cfg.run.algorithm}) success rate: {success_rate}")
                logger.info(f"Policy evaluation with CSI ({self.cfg.run.algorithm}) Mean frame coverage: {mean_frame_coverage}")
                
                save_data = {
                    "success_list": success_list,
                    "pca_type": self.cfg.run.pca_type,
                    "csi_pca_components": self.cfg.run.csi_pca_components,
                }

                #np.save(f"/home/david/LAB/Kinesis/data/csi/results_{self.cfg.run.csi_motion_category}_{self.cfg.run.csi_pca_components}.npy", save_data) # TODO: CHECK THIS PATH
                np.save(f"/home/david/LAB/Kinesis/csi/sucess_rate_csi_{self.cfg.run.algorithm}/results_{self.cfg.run.algorithm}_{self.cfg.run.pca_type}_{self.env.cfg.run.csi_pca_components}_components_trial_{trial}.npy", save_data)

    

    
    def eval_single_thread(self) -> Tuple[bool, float]:
        """
        Evaluates the policy in a single thread by running an episode.

        Returns:
            bool: True if the episode terminated successfully, False otherwise.
        """
        with to_cpu(*self.sample_modules), torch.no_grad():
            obs_dict, info = self.env.reset()
            state = self.preprocess_obs(obs_dict)
            for t in range(10000):
                actions = self.policy_net.select_action(
                    torch.from_numpy(state).to(self.dtype), True
                )[0].numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(
                    self.preprocess_actions(actions)
                )
                next_state = self.preprocess_obs(next_obs)
                done = terminated or truncated

                if done:                      
                    return not terminated, reward, self.env.frame_coverage
                state = next_state

        # If the loop exits without termination, consider it a failure
        return False, reward, self.env.frame_coverage