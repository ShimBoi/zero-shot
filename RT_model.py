import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from einops import rearrange, repeat
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *
from libero.lifelong.models.modules.transformer_modules import *


import tensorflow as tf
import tensorflow_probability as tfp  # Add this line

import rlds
from PIL import Image
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from collections import defaultdict
import matplotlib.pyplot as plt


class MyRT1XPolicy(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        self.policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path="rt_1_x_tf_trained_for_002272480_step",
            load_specs_from_pbtxt=True,
            use_tf_function=True)
        
    def get_action(self, data):
        data_tensor = tf.convert_to_tensor(data)
        time_step = ts.transition(data_tensor, reward=np.zeros((), dtype=np.float32))
        policy_state = self.policy.get_initial_state(batch_size=1)
        action_step = self.policy.action(time_step, policy_state)
        action = action_step.action.numpy()
        return action
     



