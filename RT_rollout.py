from RT_model import MyRT1XPolicy

from hydra import compose, initialize

from libero.libero import benchmark, get_libero_path
import hydra
import pprint
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from omegaconf import OmegaConf
import yaml
from easydict import EasyDict
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset, get_dataset)
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)
hydra.core.global_hydra.GlobalHydra.instance().clear()

import torch

### load the default hydra config
initialize(config_path="../libero/configs")
hydra_cfg = compose(config_name="config")
yaml_config = OmegaConf.to_yaml(hydra_cfg)
cfg = EasyDict(yaml.safe_load(yaml_config))

pp = pprint.PrettyPrinter(indent=2)
# pp.pprint(cfg.policy)

# prepare lifelong learning
cfg.folder = get_libero_path("datasets")
cfg.bddl_folder = get_libero_path("bddl_files")
cfg.init_states_folder = get_libero_path("init_states")
cfg.eval.num_procs = 1
cfg.eval.n_eval = 5

# cfg.train.n_epochs = 25

# pp.pprint(f"Note that the number of epochs used in this example is intentionally reduced to 5.")

task_order = cfg.data.task_order_index # can be from {0 .. 21}, default to 0, which is [task 0, 1, 2 ...]
cfg.benchmark_name = "libero_spatial" # can be from {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
benchmark = get_benchmark(cfg.benchmark_name)(task_order)

# # prepare datasets from the benchmark
datasets = []
descriptions = []
shape_meta = None
n_tasks = benchmark.n_tasks

for i in range(n_tasks):
    try:
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i==0),
                seq_len=cfg.data.seq_len,
        )
        # add language to the vision dataset, hence we call vl_dataset
        descriptions.append(benchmark.get_task(i).language)
        datasets.append(task_i_dataset)
    except:
        print("Skipping", i);
        continue

# kept task n = 3
print(len(datasets))
task_embs = get_task_embs(cfg, descriptions)    # shape = (1, 768)
benchmark.set_task_embs(task_embs)


datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
n_demos = [data.n_demos for data in datasets]
n_sequences = [data.total_num_sequences for data in datasets]

cfg.policy.policy_type = "MyRT1XPolicy"
# create_experiment_dir(cfg)
cfg.shape_meta = shape_meta
print(cfg.lifelong.algo, cfg.policy.policy_type)
# print(type(cfg))
# algo = safe_device(nn.Sequential(n_tasks, cfg), cfg.device)


#########################################################
#########################################################
#########################################################

from IPython.display import HTML
from base64 import b64encode
import imageio

from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs
import numpy as np

# You can turn on subprocess
env_num = 1
action_dim = 7

# If it's packnet, the weights need to be processed first
task_id = 0
task = benchmark.get_task(task_id)
task_emb = benchmark.get_task_emb(task_id)
print(task)

# # Initialize your policy
policy = MyRT1XPolicy(cfg, shape_meta)

env_args = {
    "bddl_file_name": os.path.join(
        cfg.bddl_folder, task.problem_folder, task.bddl_file
    ),
    "camera_heights": cfg.data.img_h,
    "camera_widths": cfg.data.img_w,
}

env = DummyVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
)

init_states_path = os.path.join(
    cfg.init_states_folder, task.problem_folder, task.init_states_file
)
init_states = torch.load(init_states_path)

env.reset()

init_state = init_states[0:1]
dones = [False]

obs = env.set_init_state(init_state)

# Make sure the gripper is open to make it consistent with the provided demos.
dummy_actions = np.zeros((env_num, action_dim))
for _ in range(5):
    obs, _, _, _ = env.step(dummy_actions)

steps = 0

obs_tensors = [[]] * env_num
while steps < cfg.eval.max_steps:
    steps += 1
    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
    
    # Use your policy to get the action
    action = policy.get_action(data)

    obs, reward, done, info = env.step(action)

    for k in range(env_num):
        dones[k] = dones[k] or done[k]
        obs_tensors[k].append(obs[k]["agentview_image"])
    if all(dones):
        break

# visualize video
# obs_tensor: (env_num, T, H, W, C)

images = [img[::-1] for img in obs_tensors[0]]
fps = 30
writer  = imageio.get_writer('tmp_video.mp4', fps=fps)
for image in images:
    writer.append_data(image)
writer.close()

video_data = open("tmp_video.mp4", "rb").read()
video_tag = f'<video controls alt="test" src="data:video/mp4;base64,{b64encode(video_data).decode()}">'
HTML(data=video_tag)