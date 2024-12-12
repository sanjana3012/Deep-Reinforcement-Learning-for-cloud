# gym_cloud_env.py

import gym
import numpy as np
from gym import spaces
import random
import logging
import os
import csv
import time

class Task(object):
    """
    Represents a single task in the cloud environment.
    """
    def __init__(self, jobID, index, CPU, RAM, disk, status):
        self.parent = []
        self.child = []
        self.jobID = jobID
        self.index = index
        self.CPU = CPU  # Fraction of VM's CPU capacity
        self.RAM = RAM  # Fraction of VM's RAM capacity
        self.disk = disk
        self.status = status  # -1: rejected, 0: finished, 1: ready, 2: running
        self.original_runtime = 0  # To be set during initialization
        self.runtime = 0  # To be set during initialization
        self.ddl = 0  # To be set during initialization
        self.endtime = 0  # Step when the task will finish

class DAG(object):
    """
    Represents the task dependencies in the cloud environment.
    """
    all_tasks_loaded = False
    full_task_list = []  # holds all tasks from the file
    full_job_list = []   # list of jobs (each job is a list of tasks)

    def __init__(self, fname, num_task, env):
        self.fname = fname
        self.num_task = num_task
        self.env = env
        self.job = []
        self.task = []

    @classmethod
    def load_all_tasks(cls, fname):
        """
        Load all tasks from the file once into memory.
        """
        if cls.all_tasks_loaded:
            return  # already loaded

        cls.full_job_list = []
        current_job_tasks = []
        total_tasks = 0

        with open(fname, 'r') as f:
            for line in f:
                if line.startswith('J'):
                    # A new job line encountered
                    if len(current_job_tasks) > 0:
                        cls.full_job_list.append(current_job_tasks)
                        current_job_tasks = []
                else:
                    info = list(line.strip().split())
                    if len(info) != 6:
                        # Handle malformed lines
                        continue
                    # Assuming the file format: jobID, index, CPU, RAM, disk, status
                    # Convert CPU and RAM to float fractions
                    t = Task(
                        jobID=info[0],
                        index=info[1],
                        CPU=float(info[2]),
                        RAM=float(info[3]),
                        disk=info[4],
                        status=int(info[5]),
                          # Placeholder; to be set during task initialization
                    )
                    current_job_tasks.append(t)
                    total_tasks += 1

            # Don't forget the last job if exists
            if len(current_job_tasks) > 0:
                cls.full_job_list.append(current_job_tasks)

        cls.full_task_list = [t for job in cls.full_job_list for t in job]
        cls.all_tasks_loaded = True
        # print(f"[DEBUG] Loaded {total_tasks} tasks in total from {fname}.")

    def initTask(self):
        """
        Randomly choose `num_task` tasks from the full task list and initialize deadlines.
        """
        # print("[DEBUG] Initializing tasks with a random subset of size:", self.num_task)
        self.env.rej = 0
        chosen_tasks = random.sample(self.full_task_list, self.num_task)
        self.job=[]
        self.job = [chosen_tasks]
        # Assign deadlines relative to simulation steps
        for task in chosen_tasks:
            # Assign runtime between 5 and 50 simulation steps
            task.runtime = random.randint(5, 50)

            # Deadlines will be assigned dynamically during scheduling in the step function
            # task.ddl = self.env.current_step + task.runtime + random.randint(10, 100)

        # print("[DEBUG] Example chosen tasks indices:", [t.index for t in chosen_tasks[:5]], "...")
        self.buildDAG()

    def checkRing(self, parent, child):
        """
        Check for cyclic dependencies between tasks.
        """
        if parent.index == child.index:
            return True
        if len(child.child) == 0:
            return False
        for c in child.child:
            if self.checkRing(parent, c):
                return True
        return False

    def buildDAG(self):
        """
        Establish parent-child relationships among tasks to form a DAG.
        """
        for job in self.job:
            for task in job:
                # Randomly assign dependencies
                # Adjust the range based on the number of existing tasks
                if len(job) < 2:
                    continue
                i = random.randint(0, len(job) - 2)
                parent = job[i]
                if self.checkRing(parent, task) == False:
                    task.parent.append(parent)
                    parent.child.append(task)

    def rejTask(self, task, rej_code):
        """
        Recursively reject a task and its child tasks.
        """
        # if task.status == -1:
        #     return  # Already rejected
        task.status = -1
        self.env.rej += 1        
        self.env.total_tasks += 1
        for c in task.child:
            if rej_code == -1:
                print("Parent Task was rejected due to CPU oversize so rejecting child tasks")
            elif rej_code == -2:
                print("Parent Task was rejected due to RAM oversize so rejecting child tasks")
            elif rej_code == -3:
                print("Parent Task was rejected due to deadline so rejecting child tasks")
            else:
                print("Parent Task was rejected due to insufficient resources so rejecting child tasks")

            self.rejTask(c, rej_code)

    def hasParent(self, task):
        """
        Check if a task has any active (ready) parent tasks.
        """
        for p in task.parent:
            if p.status == 1:
                return True
        return False

    def updateStatus(self, task, rej_code):
        """
        Update the status of a task. If rejected, propagate rejection.
        """
        if task.status == -1:
            if rej_code == -1:
                print("Task rejected due to CPU oversize")
            elif rej_code == -2:
                print("Task rejected due to RAM oversize")
            elif rej_code == -3:
                print("Task rejected due to deadline")
            else:
                print("Task rejected due to insufficient resources")
            print("Going to check for child tasks to be rejected now... in the update status function")
            
            self.rejTask(task, rej_code)

    def taskQueue(self):
        """
        Generate a queue of tasks that are ready to be processed (no active parents).
        """
        self.task = []
        for job in self.job:
            for t in job:
                if t.status == 1 and not self.hasParent(t):
                    self.task.append(t)

class CloudEnv(gym.Env):
    """
    Custom Gym environment for simulating cloud task scheduling with energy efficiency considerations.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, scale='small', fname='output_5000.txt', num_task=5000, num_server=300, max_steps=10000,file_path=None):
        super(CloudEnv, self).__init__()

        self.scale = scale
        self.fname = fname
        self.num_task = num_task
        self.serverNum = num_server
        self.VMNum = 5  # Updated based on provided data (10-50 VMs per server)
        self.rej = 0
        self.total_tasks = 0
        self.episode = 0
        self.max_steps = max_steps
        self.vm_cpu_capacity = 0.5
        self.vm_ram_capacity = 1.0

        # Cumulative energy and cost metrics
        self.cumulative_energy = 0.0
        self.cumulative_power = 0.0
        self.cumulative_elec_cost = 0.0
        self.server_cpu_capacity = self.VMNum * self.vm_cpu_capacity  # CPU units per server (aligned with VM capacities)
        self.server_ram_capacity = self.VMNum * self.vm_ram_capacity  # RAM units per server (aligned with VM capacities)
        self.last_power_measurement = 0.0
        self.last_cost_measurement = 0.0
        self.episode_reward = 0.0

        # Simulation time
        self.current_step = 0

        # Determine number of farms based on scale
        if self.scale == 'small':
            # self.farmNum = max(1, self.serverNum // 30)
            self.farmNum = 1
        elif self.scale == 'medium':
            # self.farmNum = max(1, self.serverNum // 20)
            self.farmNum = 3
        elif self.scale == 'large':
            # self.farmNum = max(1, self.serverNum // 10)
            self.farmNum = 5
        else:
            raise ValueError("Scale must be 'small', 'medium', or 'large'.")

        # Load all tasks once
        DAG.load_all_tasks(self.fname)

        self.dag = DAG(self.fname, self.num_task, self)
        self.task = []

        # Farm and server configurations
        self.farmOri = []
        self.remainFarm = []
        self.FarmResources = []
        self.VMtask = []
        self.pwrPre = []
        self.pwrPFarm = []

        self.servers_per_farm = self.serverNum // self.farmNum
        self.action_space = spaces.Discrete(self.farmNum * self.servers_per_farm * self.VMNum)

        # Observation space: CPU and RAM availability for each VM
        # CPU and RAM are normalized between 0 and 1
        obs_dim = self.farmNum * self.servers_per_farm * self.VMNum * 2
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                           shape=(obs_dim,), dtype=np.float32)

        self.current_task = None

        # Tracking resource usage for metrics
        self.total_accepted_cpu = 0.0
        self.total_accepted_ram = 0.0
        self.accepted_tasks_count = 0

        # Define coefficients for penalties
        self.beta = 0.05    # Idle server penalty
        self.gamma = 0.001  # Energy consumption penalty

        # Power model parameters based on provided data
        self.P_idle = 1  # Idle power in watts (within 200-560 W range)
        # self.P_max = 6000   # Max power in watts (within 400-800 W range)

        # Electricity cost per kWh in USD (16.83 cents)
        self.electricity_price_per_kwh = 10
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.file_path=file_path
        # self.file_path = os.path.join("logs/cloud_env",self.file_name )
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)  # Ensure the directory exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Episode', 'Total_Energy_Consumed', 'Total_Power_Consumed', 'Total_Cost', 'Total Tasks', 'Total_Rejections', 'Total_Accepted_Tasks', 'Total_Reward'])

        # Initialize logging
        logging.basicConfig(filename='cloud_env.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s:%(message)s')
        self.logger = logging.getLogger()

        

        # Define VM types with varying specifications
        self.VM_TYPES = {
            'Basic': {
                'CPU_capacity': 0.2,    # CPU units
                'RAM_capacity': 0.4,    # RAM units
                'P_idle': 50,           # Idle power in Watts
                'P_active': 100         # Active power in Watts
            },
            'Standard': {
                'CPU_capacity': 0.4,
                'RAM_capacity': 1,
                'P_idle': 70,
                'P_active': 150
            },
            'High-Performance': {
                'CPU_capacity': 0.6,
                'RAM_capacity': 2,
                'P_idle': 90,
                'P_active': 200
            }
        }
        # Initialize farms and servers
        self._setFarm()

    def reset(self,seed,options=None):
        """
        Reset the environment to an initial state and return an initial observation.
        """
        # Reset random seed for variability
        # seed_val = int(random.random() * 1e6)
        # random.seed(seed_val)
        # np.random.seed(seed_val)
        if seed is not None:
            self.seed(seed)
        else:
            seed_val = int(random.random() * 1e6)
            random.seed(seed_val)
            np.random.seed(seed_val)

        self.episode += 1
        self.episode_reward = 0.0
        # Reset cumulative metrics
        self.cumulative_energy = 0.0
        self.cumulative_power = 0.0
        self.cumulative_elec_cost = 0.0
        self.current_step = 0
        self.total_tasks = 0

        # Reset rejection count
        self.rej = 0

        # Reset resource usage tracking
        self.total_accepted_cpu = 0.0
        self.total_accepted_ram = 0.0
        self.accepted_tasks_count = 0

        # Initialize tasks and DAG
        self.dag = DAG(self.fname, self.num_task, self)
        # self.dag.job=[]
        self.dag.initTask()
        self._generateQueue()

        # Reset farms and servers with heterogeneous resources
        self._setFarm()

        # Calculate initial power and cost
        initial_power, initial_cost = self._calculate_power_and_cost()
        self.last_power_measurement = initial_power
        self.last_cost_measurement = initial_cost

        # Reset cumulative energy with initial power and cost
        # Assuming the initial state has been active for one time unit
        # Adjust as necessary based on your time stepping
        self.cumulative_power += initial_power
        self.cumulative_elec_cost += initial_cost
        self.cumulative_energy += initial_power  

        # Log initial state
        self.logger.info(f"Reset: Power={initial_power}W, Cost=${initial_cost:.2f}")

        obs = self._get_obs()
        return obs,{}
    def seed(self, seed=None):
        """
        Set the random seed for the environment.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (int): The action taken by the agent.

        Returns:
            obs (np.array): The next observation.
            reward (float): The reward obtained from taking the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.
        """

        done = False
        info = {}

        # Increment simulation time
        self.current_step += 1

        # Assign the current task based on the action
        if len(self.task) > 0:
            self.current_task = self.task[0]
            # Assign deadline relative to current step, runtime, and a random buffer
            self.current_task.ddl = self.current_step + self.current_task.runtime + random.randint(10, 100)
            farm_i, server_i, vm_i = self._decode_action(action)
            rej_code = self._checkRej(farm_i, server_i, vm_i, self.current_task)

            if rej_code == -1:
                # Task rejected due to CPU oversize
                self.task.pop(0)
                self.current_task.status = -1
                self.current_task.runtime = 0
                self.dag.updateStatus(self.current_task, rej_code)
                self.logger.info(f"Step {self.current_step}: Task {self.current_task.index} rejected CPU Oversize.")
            elif rej_code == -2:
                # Task rejected due to RAM oversize
                self.task.pop(0)
                self.current_task.status = -1
                self.current_task.runtime = 0
                self.dag.updateStatus(self.current_task, rej_code)
                self.logger.info(f"Step {self.current_step}: Task {self.current_task.index} rejected RAM Oversize.")
            elif rej_code == -3:
                # Task rejected due to deadline
                self.task.pop(0)
                self.current_task.status = -1
                self.current_task.runtime = 0
                self.dag.updateStatus(self.current_task, rej_code)
                self.logger.info(f"Step {self.current_step}: Task {self.current_task.index} rejected (Deadline).")
            elif rej_code == 1:
                # Task rejected due to insufficient resources
                self.current_task.status = 1
                self.dag.updateStatus(self.current_task, rej_code)
                self.logger.info(f"Step {self.current_step}: Task {self.current_task.index} rejected (Insufficient Resources).")
            else:
                # Task accepted
                self.task.pop(0)
                self.total_tasks += 1
                self._updateServerState(farm_i, server_i, vm_i, self.current_task)
                self.current_task.status = 2  # Running
                self.current_task.endtime = self.current_step + int(self.current_task.runtime)
                self.VMtask[farm_i][server_i][vm_i].append(self.current_task)
                self.accepted_tasks_count += 1
                self.total_accepted_cpu += self.current_task.CPU
                self.total_accepted_ram += self.current_task.RAM
                self.logger.info(f"Step {self.current_step}: Task {self.current_task.index} assigned to Farm {farm_i}, Server {server_i}, VM {vm_i}.")

        # Release tasks that have completed
        self._release_tasks()

        # Generate new task queue
        self._generateQueue()

        # Calculate power and cost based on current resource usage
        current_power, current_cost = self._calculate_power_and_cost()
        # print(f"Current Power: {current_power}W, Current Cost: ${current_cost:.2f}")

        # Calculate incremental power and cost
        delta_power = current_power - self.last_power_measurement
        # print(f"Last Power: {self.last_power_measurement}W")
        # print(f"Delta Power: {delta_power}W")
        delta_cost = current_cost - self.last_cost_measurement

        # Update cumulative metrics
        self.cumulative_power += delta_power
        # print(f"Cumulative Power: {self.cumulative_power}W")
        self.cumulative_elec_cost += delta_cost
        # Accumulate energy as Power * Time (assuming time step is 1)
        self.cumulative_energy += delta_power

        # Update last measurements
        self.last_power_measurement = current_power
        self.last_cost_measurement = current_cost

        idle_servers = self.count_idle_servers()
        print(f"Idle Servers: {idle_servers}")
        print(f"Tasks at hand: {len(self.task)}")

        # Calculate reward using the refined reward function
        reward = self._calculate_reward_1(current_power, self.rej)
        self.episode_reward += reward

        # Log step metrics
        self.logger.info(f"Step {self.current_step}: Power={self.cumulative_power}W, Cost=${self.cumulative_elec_cost:.2f}, Rejections={self.rej}, Reward={reward:.4f}")

        # Check termination conditions
        # if self.current_step >= self.num_task:
        #     done = True
        if self.total_tasks >= self.num_task:
            done = True
        elif len(self.task) == 0 and self._no_running_tasks():
            done = True


        # Get next observation
        obs = self._get_obs()

        if done:
            print(f"Episode {self.episode} completed in {self.current_step} steps and energy {self.cumulative_energy}.")
            print(f"Total Tasks: {self.total_tasks} | Accepted Tasks: {self.accepted_tasks_count} | Rejections: {self.rej}")
            self.cumulative_energy=self.cumulative_power*self.current_step
            # Log episode-level metrics to CSV
            with open(self.file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.episode, self.cumulative_energy, self.cumulative_power, self.cumulative_elec_cost, self.total_tasks,self.rej, self.accepted_tasks_count, self.episode_reward])
            return obs, reward, done, False, info

        # Additional info
        info['cumulative_energy'] = self.cumulative_energy  # in Watt-steps
        info['cumulative_cost'] = self.cumulative_elec_cost  # in USD
        info['rejections'] = self.rej
        info['accepted_tasks'] = self.accepted_tasks_count

        return obs, reward, done, False, info

    def _no_running_tasks(self):
        """
        Check if there are no running tasks across all VMs.

        Returns:
            bool: True if no tasks are running, False otherwise.
        """
        for farm in self.VMtask:
            for server in farm:
                for vm in server:
                    if len(vm) > 0:
                        return False
        return True

    def _calculate_reward_1(self, energy, rejections):
        """
        Calculate the reward for the current step.

        Args:
            energy (float): Cumulative energy consumed in the episode.
            rejections (int): Number of tasks rejected in the current step.

        Returns:
            float: Calculated reward.
        """
        max_energy = 1e4    # Adjust based on observed maximums
        max_rejections = 100  # Adjust based on observed maximums

        # Normalize energy and rejections
        normalized_energy = energy / max_energy  # Scale to [0, 1]
        print(f"Normalized Energy: {normalized_energy}")
        normalized_rejections = rejections / max_rejections  # Scale to [0, 1]
        print(f"Normalized Rejections: {normalized_rejections}")

        # Define weights for normalized components
        energy_weight = 0.5    # Adjust to balance importance
        rejection_weight = 1

        # Calculate reward components
        energy_component = -energy_weight * normalized_energy
        rejection_component = -rejection_weight * normalized_rejections

        # Combine components into a total reward
        reward = energy_component + rejection_component
        print(f"Reward: {reward}")

        return reward

    def _calculate_reward_2(self, idle_servers, rejections):
        """
        Alternative reward calculation method.

        Args:
            idle_servers (int): Number of idle servers in the current step.
            rejections (int): Number of tasks rejected in the current step.

        Returns:
            float: Calculated reward.
        """
        # Define weights for rejections
        rejection_weight = -1.0     # Negative because we want to minimize rejections

        # Reward is a combination of negative idle servers and rejections
        reward = -idle_servers + (rejection_weight * rejections)
        
        return reward

    def render(self, mode='human'):
        """
        Render the environment's state.
        """
        print(f"Step: {self.current_step} | Tasks left: {len(self.task)} | "
              f"Cumulative Power: {self.cumulative_power}W | "
              f"Cumulative Electricity Cost: ${self.cumulative_elec_cost:.2f} | "
              f"Cumulative Energy Consumed: {self.cumulative_energy}W-steps | "
              f"Rejections: {self.rej} | Accepted Tasks: {self.accepted_tasks_count}")

    def close(self):
        pass

    def _is_server_highest_resources(self, farm_i, server_i, vm_i):
        """
        Check if the selected server has the highest remaining CPU among all servers.
        """
        selected_server_cpu = sum([vm[0] + vm[1] for vm in self.remainFarm[farm_i][server_i]])
    
        for fi in range(self.farmNum):
            for si in range(self.servers_per_farm):
                if fi == farm_i and si == server_i:
                    continue  # Skip the selected server
                current_server_cpu = sum([vm[0] + vm[1] for vm in self.remainFarm[fi][si]])
                if current_server_cpu > selected_server_cpu:
                    return False
        return True

    def _is_vm_highest_resources(self, farm_i, server_i, vm_i):
        """
        Check if the selected VM has the highest remaining CPU in its server.
        """
        selected_vm_resource = self.remainFarm[farm_i][server_i][vm_i][0] + self.remainFarm[farm_i][server_i][vm_i][1]
    
        for other_vm in range(self.VMNum):
            if other_vm == vm_i:
                continue
            if (self.remainFarm[farm_i][server_i][other_vm][0] + self.remainFarm[farm_i][server_i][other_vm][1]) > selected_vm_resource:
                return False
        return True

    def count_idle_vms(self):
        """
        Counts the number of idle VMs across all farms.
        An idle VM is defined as a VM with no scheduled tasks.
        """
        idle_vms = 0
        for farm in self.VMtask:
            for server in farm:
                for vm in server:
                    if len(vm) == 0:
                        idle_vms += 1
        return idle_vms

    def _generateQueue(self):
        """
        Generate a queue of tasks that are ready to be processed.
        """
        self.task = []
        self.dag.taskQueue()
        for t in self.dag.task:
            if t.status == 1 and not self._hasParent(t):
                self.task.append(t)
        # print("[DEBUG] Queue generated with", len(self.task), "tasks ready.")

    def _hasParent(self, task):
        """
        Check if a task has any active (ready) parent tasks.
        """
        for p in task.parent:
            if p.status == 1:
                return True
        return False

    def _setFarm(self):
        """
        Initialize farms, servers, and VMs with their resource capacities.
        Introduces heterogeneous resources by assigning different VM types.
        """
        # Reset farm configurations
        self.farmOri = []
        self.remainFarm = []
        self.FarmResources = []
        self.VMtask = []

        for _ in range(self.farmNum):
            farm_servers = []
            for _ in range(self.servers_per_farm):
                server_vms = []
                for _ in range(self.VMNum):
                    # Assign VM type based on predefined distribution
                    vm_type = random.choices(
                        population=list(self.VM_TYPES.keys()),
                        weights=[0.2, 0.5, 0.3],  # Example distribution: 20% Basic, 50% Standard, 30% High-Performance
                        k=1
                    )[0]
                    vm_specs = self.VM_TYPES[vm_type]
                    
                    # Initialize VM with remaining CPU and RAM based on its type
                    remaining_CPU = vm_specs['CPU_capacity']
                    remaining_RAM = vm_specs['RAM_capacity']
                    
                    # Append VM specifications to server_vms
                    server_vms.append([remaining_CPU, remaining_RAM, vm_type, vm_specs['P_idle'], vm_specs['P_active']])
                    # VM structure: [remaining_CPU, remaining_RAM, VM_type, P_idle, P_active]
                
                farm_servers.append(server_vms)
            self.remainFarm.append(farm_servers)
            # Initialize farm-level resources if needed
            # Not used in current implementation but can be extended
            self.FarmResources.append([self.server_cpu_capacity * self.servers_per_farm,
                                       self.server_ram_capacity * self.servers_per_farm])

        # Initialize task lists per VM
        self.VMtask = [[[[] for _ in range(self.VMNum)] for _ in range(self.servers_per_farm)] for _ in range(self.farmNum)]

    def _decode_action(self, action):
        """
        Decode a discrete action into farm, server, and VM indices.
        """
        farm_i = action // (self.servers_per_farm * self.VMNum)
        remainder = action % (self.servers_per_farm * self.VMNum)
        server_i = remainder // self.VMNum
        vm_i = remainder % self.VMNum
        return farm_i, server_i, vm_i

    def _get_obs(self):
        """
        Generate the observation representing the current state of the environment.
        """
        arr = []
        for farm in self.remainFarm:
            for server in farm:
                for vm in server:
                    # Normalize CPU and RAM by their capacities
                    # vm structure: [remaining_CPU, remaining_RAM, VM_type, P_idle, P_active]
                    cpu_normalized = vm[0] / self.VM_TYPES[vm[2]]['CPU_capacity']
                    ram_normalized = vm[1] / self.VM_TYPES[vm[2]]['RAM_capacity']
                    arr.extend([cpu_normalized, ram_normalized])
        return np.array(arr, dtype=np.float32)

    def _checkRej(self, farm_i, server_i, vm_j, task):
        """
        Check if a task can be assigned to a specific VM. Returns rejection code.

        Returns:
            -1: Rejected due to CPU oversize.
             -2: Rejected due to RAM oversize.
             -3: Rejected due to deadline.
             1: Rejected due to insufficient resources.
             0: Accepted.
        """
        # Extract VM type and specs
        vm_type = self.remainFarm[farm_i][server_i][vm_j][2]
        P_idle = self.remainFarm[farm_i][server_i][vm_j][3]
        P_active = self.remainFarm[farm_i][server_i][vm_j][4]
        CPU_capacity = self.VM_TYPES[vm_type]['CPU_capacity']
        RAM_capacity = self.VM_TYPES[vm_type]['RAM_capacity']

        # Check if task is oversize first
        if task.CPU > CPU_capacity:
            # Oversize task
            print(f"CPU size required by Task {task.index} is {task.CPU} which exceeds VM {vm_j}'s capacity ({CPU_capacity}).")
            return -1
        elif task.RAM > RAM_capacity:
            # Oversize task
            print(f"RAM size required by Task {task.index} is {task.RAM} which exceeds VM {vm_j}'s capacity ({RAM_capacity}).")
            return -2

        remain_cpu = self.remainFarm[farm_i][server_i][vm_j][0] - task.CPU
        remain_ram = self.remainFarm[farm_i][server_i][vm_j][1] - task.RAM

        # Check if task can finish before deadline
        if self.current_step + int(task.runtime) > task.ddl:
            return -3

        if remain_cpu < 0 or remain_ram < 0:
            return 1

        # If here, task is feasible
        return 0

    def _updateServerState(self, farm_i, server_i, vm_j, task):
        """
        Update the server's remaining resources after assigning a task.
        """
        # Deduct CPU and RAM based on task requirements
        self.remainFarm[farm_i][server_i][vm_j][0] -= task.CPU
        self.remainFarm[farm_i][server_i][vm_j][1] -= task.RAM
        # Optionally update farm-level resources if tracking

    def _release_tasks(self):
        """
        Release tasks that have completed execution based on the current simulation step.
        """
        for farm_i in range(self.farmNum):
            for server_i in range(self.servers_per_farm):
                for vm_j in range(self.VMNum):
                    finished_tasks = []
                    for task in self.VMtask[farm_i][server_i][vm_j]:
                        if task.endtime <= self.current_step:
                            task.status = 0
                            # Free resources
                            self.remainFarm[farm_i][server_i][vm_j][0] += task.CPU
                            self.remainFarm[farm_i][server_i][vm_j][1] += task.RAM
                            finished_tasks.append(task)
                            self.logger.info(f"Step {self.current_step}: Task {task.index} finished on Farm {farm_i}, Server {server_i}, VM {vm_j}.")

                    # Remove finished tasks
                    for ft in finished_tasks:
                        self.VMtask[farm_i][server_i][vm_j].remove(ft)
                        self.dag.updateStatus(ft, 0)

    def _calculate_power_and_cost(self):
        """
        Calculate total power and electricity cost based on current resource usage.

        Returns:
            total_power (float): Total power consumption across all servers.
            ep_elec_cost (float): Total electricity cost.
        """
        server_pwrs = []
        for farm_i in range(self.farmNum):
            for s_i in range(self.servers_per_farm):
                # Calculate power consumption for each VM
                # sum_power_vm=0
                for vm_j in range(self.VMNum):
                    vm = self.remainFarm[farm_i][s_i][vm_j]
                    VM_type = vm[2]
                    P_idle = vm[3]
                    P_active = vm[4]
                    CPU_capacity = self.VM_TYPES[VM_type]['CPU_capacity']
                    RAM_capacity = self.VM_TYPES[VM_type]['RAM_capacity']

                    # Calculate current utilization ratio based on CPU
                    used_cpu = CPU_capacity - vm[0]
                    Util = used_cpu / CPU_capacity

                    if Util > 0:
                        # VM is active
                        P = P_idle + (P_active * Util)
                    else:
                        # VM is idle
                        P = P_idle

                    server_pwrs.append(P)

        total_power = sum(server_pwrs)
        print(f"Total power consumption: {total_power}W")

        # Convert power to electricity cost
        # Formula: Cost = (Power (W) / 1000) * Price per kWh
        ep_elec_cost = (total_power)/1000 * self.electricity_price_per_kwh

        return total_power, ep_elec_cost

    def _calculate_idle_power(self):
        """
        Calculate the total power consumed by idle servers.
        """
        idle_power = 0.0
        for farm_i in range(self.farmNum):
            for s_i in range(self.servers_per_farm):
                # A server is idle if all its VMs have no tasks
                if all(len(vm_tasks) == 0 for vm_tasks in self.VMtask[farm_i][s_i]):
                    # Assuming P_idle for an idle server is sum of P_idle of its VMs
                    for vm_j in range(self.VMNum):
                        idle_power += self.remainFarm[farm_i][s_i][vm_j][3]  # P_idle of VM
        return idle_power

    def _get_utilization(self, farm_i, server_i, vm_j):
        """
        Get the current CPU utilization ratio of a specific VM.
        """
        vm = self.remainFarm[farm_i][server_i][vm_j]
        used_cpu = self.VM_TYPES[vm[2]]['CPU_capacity'] - vm[0]
        Ur = used_cpu / self.VM_TYPES[vm[2]]['CPU_capacity']
        return Ur

    def count_idle_servers(self):
        """
        Counts the number of idle servers across all farms.
        An idle server is defined as a server with no scheduled tasks on any of its VMs.
        """
        idle_servers = 0
        for farm in self.VMtask:
            for server in farm:
                # A server is idle if all its VMs have no tasks
                if all(len(vm_tasks) == 0 for vm_tasks in server):
                    idle_servers += 1
        return idle_servers

    def get_cloud_utilization(self):
        """
        Get the overall CPU utilization ratio of the cloud environment.
        """
        used_cpu = 0
        total_cpu = 0
        for farm_i in range(self.farmNum):
            for server_i in range(self.servers_per_farm):
                for vm_j in range(self.VMNum):
                    vm = self.remainFarm[farm_i][server_i][vm_j]
                    VM_type = vm[2]
                    CPU_capacity = self.VM_TYPES[VM_type]['CPU_capacity']
                    used_cpu += CPU_capacity - vm[0]
                    total_cpu += CPU_capacity
        Ur = used_cpu / total_cpu if total_cpu > 0 else 0
        return Ur

    def _get_overall_utilization(self, farm_i, server_i, vm_j):
        """
        Get the overall CPU utilization ratio of a specific server.
        """
        server = self.remainFarm[farm_i][server_i]
        used_cpu = sum([self.VM_TYPES[vm[2]]['CPU_capacity'] - vm[0] for vm in server])
        Ur = used_cpu / (self.server_cpu_capacity)
        return Ur

    def _get_vm_utilization(self, farm_i, server_i, vm_i):
        """
        Calculate the CPU utilization ratio of a specific VM.
        """
        vm = self.remainFarm[farm_i][server_i][vm_i]
        used_vm_cpu = self.VM_TYPES[vm[2]]['CPU_capacity'] - vm[0]
        vm_utilization = used_vm_cpu / self.VM_TYPES[vm[2]]['CPU_capacity']
        return vm_utilization
