# gym_cloud_env.py

import gym
import numpy as np
from gym import spaces
import time
import random

class Task(object):
    """
    Represents a single task in the cloud environment.
    """
    def __init__(self, jobID, index, CPU, RAM, disk, status, original_runtime):
        self.parent = []
        self.child = []
        self.jobID = jobID
        self.index = index
        self.CPU = CPU
        self.RAM = RAM
        self.disk = disk
        self.status = status  #-1: rejected, 0: finished, 1: ready, 2: running
        self.original_runtime = original_runtime  # Base runtime before adjustments
        self.runtime = original_runtime  # Adjusted runtime based on utilization
        self.ddl = time.time() + self.runtime + random.randint(1, 1000) * 100
        self.endtime = 0

class DAG(object):
    """
    Represents the task dependencies in the cloud environment.
    """
    all_tasks_loaded = False
    full_task_list = []  # holds all tasks from the file
    full_job_list = []   # list of jobs (each job is a list of tasks)

    def __init__(self, fname, num_task,env):
        self.fname = fname
        self.num_task = num_task
        self.env=env
        self.job = []
        self.task = []

    @classmethod
    def load_all_tasks(cls, fname):
        """
        Load all tasks from the file once into memory.
        """
        if cls.all_tasks_loaded:
            return  # already loaded

        # print("[DEBUG] Loading all tasks from file:", fname)
        full_job_list = []
        current_job_tasks = []
        total_tasks = 0

        with open(fname, 'r') as f:
            for line in f:
                if line.startswith('J'):
                    # A new job line encountered
                    if len(current_job_tasks) > 0:
                        full_job_list.append(current_job_tasks)
                        current_job_tasks = []
                else:
                    info = list(line.strip().split())
                    # Assuming the file format: jobID, index, CPU, RAM, disk, original_runtime
                    t = Task(
                        jobID=info[0],
                        index=info[1],
                        CPU=float(info[2]),
                        RAM=float(info[3]),
                        disk=info[4],
                        status=1,  # Ready
                        original_runtime=float(info[5])
                    )
                    current_job_tasks.append(t)
                    total_tasks += 1

            # Don't forget the last job if exists
            if len(current_job_tasks) > 0:
                full_job_list.append(current_job_tasks)

        cls.full_job_list = full_job_list
        cls.full_task_list = [t for job in full_job_list for t in job]
        cls.all_tasks_loaded = True
        # print(f"[DEBUG] Loaded {total_tasks} tasks in total from {fname}.")

    def initTask(self):
        """
        Randomly choose `num_task` tasks from the full task list.
        """
        # print("[DEBUG] Initializing tasks with a random subset of size:", self.num_task)
        chosen_tasks = random.sample(self.full_task_list, self.num_task)
        self.job = [chosen_tasks]
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
                i = random.randint(-len(job), len(job) - 1)
                if i < 0:
                    continue  # No parent
                parent = job[i]
                if self.checkRing(parent, task) == False:
                    task.parent.append(parent)
                    parent.child.append(task)

    def rejTask(self, task):
        """
        Recursively reject a task and its child tasks.
        """
        task.status = -1
        self.env.rej += 1
        self.env.total_tasks += 1
        for c in task.child:
            self.rejTask(c)

    def hasParent(self, task):
        """
        Check if a task has any active (ready) parent tasks.
        """
        for c in task.parent:
            if c.status == 1:
                return True
        return False

    def updateStatus(self, task):
        """
        Update the status of a task. If rejected, propagate rejection.
        """
        if task.status == -1:
            self.rejTask(task)

    def taskQueue(self):
        """
        Generate a queue of tasks that are ready to be processed (no active parents).
        """
        self.task = []
        for job in self.job:
            for t in job:
                if t.status == 1 and self.hasParent(t) == False:
                    self.task.append(t)

class CloudEnv(gym.Env):
    """
    Custom Gym environment for simulating cloud task scheduling with energy efficiency considerations.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, scale='small', fname='output_5000.txt', num_task=5000, num_server=300):
        super(CloudEnv, self).__init__()

        self.scale = scale
        self.fname = fname
        self.num_task = num_task
        self.severNum = num_server
        self.VMNum = 5
        self.rej = 0
        self.total_tasks=0

        # Cumulative energy and cost metrics
        self.cumulative_energy = 0.0
        self.cumulative_power = 0.0
        self.cumulative_elec_cost = 0.0
        self.last_power_measurement = 0.0
        self.last_cost_measurement = 0.0

        # Determine number of farms based on scale
        if self.scale == 'small':
            self.farmNum = max(1, self.severNum // 30)
        elif self.scale == 'medium':
            self.farmNum = max(1, self.severNum // 20)
        elif self.scale == 'large':
            self.farmNum = max(1, self.severNum // 10)

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

        self.servers_per_farm = self.severNum // self.farmNum
        self.action_space = spaces.Discrete(self.farmNum * self.servers_per_farm * self.VMNum)

        # Observation space: CPU and RAM availability for each VM
        obs_dim = self.farmNum * self.servers_per_farm * self.VMNum * 2
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                           shape=(obs_dim,), dtype=np.float32)

        self.current_task = None

        # Tracking resource usage for metrics
        self.total_accepted_cpu = 0.0
        self.total_accepted_ram = 0.0
        self.accepted_tasks_count = 0

        # Define coefficients for penalties
        # Removed self.alpha as we're eliminating resource usage penalties
        # self.alpha = 0.01  # Resource usage penalty
        self.beta = 0.05   # Idle server penalty
        self.gamma = 0.001 # Energy consumption penalty

    def reset(self):
        """
        Reset the environment to an initial state and return an initial observation.
        """
        # Reset random seed for variability
        seed_val = int(time.time() * 1000) % 10000
        # print(f"[DEBUG] Resetting environment with seed {seed_val}")
        random.seed(seed_val)
        np.random.seed(seed_val)

        # Reset cumulative metrics
        self.cumulative_energy = 0.0
        self.cumulative_power = 0.0
        self.cumulative_elec_cost = 0.0
        self.total_tasks=0

        # Reset rejection count
        self.rej = 0

        # Reset resource usage tracking
        self.total_accepted_cpu = 0.0
        self.total_accepted_ram = 0.0
        self.accepted_tasks_count = 0

        # Initialize tasks and DAG
        self.dag = DAG(self.fname, self.num_task,self)
        self.dag.initTask()
        self._generateQueue()

        # Initialize farms and servers
        self._setFarm()

        # Initialize last measurements based on initial power and cost
        initial_power, initial_cost = self._calculate_power_and_cost()
        self.last_power_measurement = initial_power
        self.last_cost_measurement = initial_cost

        # Reset cumulative energy with initial power and cost
        # Assuming the initial state has been active for one time unit
        # Adjust as necessary based on your time stepping
        self.cumulative_power += initial_power
        self.cumulative_elec_cost += initial_cost
        self.cumulative_energy += initial_power  # Energy = Power * Time (Time=1)

        obs = self._get_obs()
        # print("[DEBUG] Reset complete. Initial observation shape:", obs.shape,
            #   "Tasks in queue:", len(self.task))
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        if len(self.task) == 0:
            # Episode done before processing any tasks
            done = True
            obs = self._get_obs()

            # Add resource usage metrics to info
            if self.accepted_tasks_count > 0:
                avg_cpu_usage = self.total_accepted_cpu / self.accepted_tasks_count
                avg_ram_usage = self.total_accepted_ram / self.accepted_tasks_count
            else:
                avg_cpu_usage = 0
                avg_ram_usage = 0

            info = {
                'EpElecCost': self.cumulative_elec_cost,
                'EpPowerUsed': self.cumulative_power,
                'EpEnergyConsumed': self.cumulative_energy,
                'EpTasksRejected': self.rej,
                'AvgCPUUsagePerTask': avg_cpu_usage,
                'AvgRAMUsagePerTask': avg_ram_usage
            }

            print("[EPISODE DONE!!!!] . TOTAL TASKS:", self.total_tasks,
                  "Total Power Used:", self.cumulative_power, 
                  "Total Electricity Cost:", self.cumulative_elec_cost,
                  "Total Energy Consumed:", self.cumulative_energy,
                  "Total Tasks Rejected:", self.rej,
                  "Avg CPU usage:", avg_cpu_usage,
                  "Avg RAM usage:", avg_ram_usage)
            return obs, 0.0, done, info

        # Process the first task in the queue
        t = self.task[0]
        # print("[DEBUG] Processing task:", t.index, "with CPU:", t.CPU, "RAM:", t.RAM, "Disk:", t.disk)
        farm_i, server_i, vm_i = self._decode_action(action)
        
        # print(f"[DEBUG] Action chosen: Farm={farm_i}, Server={server_i}, VM={vm_i}")

        # Release any tasks that have completed on this VM
        self._releaseByTime(farm_i, server_i, vm_i)

        # Check if the task can be assigned to the chosen VM
        rej_code = self._checkRej(farm_i, server_i, vm_i, t)
        
        # Initialize reward
        reward = 0.0

        if rej_code == -1:
            # Task rejected due to deadline/misaligned size
            t.status = -1
            self.dag.updateStatus(t)
            self.task.pop(0)
            reward = -2.0
            # self.total_tasks+=1
            print("[DEBUG] Task rejected due to deadline/oversize. Current tasks left:", len(self.task))

        elif rej_code == 1:
            # Task rejected due to insufficient resources
            t.status = -1
            self.dag.updateStatus(t)
            self.task.pop(0)
            reward = -1.0
            print("[DEBUG] Task rejected due to insufficient resources. Current tasks left:", len(self.task))
        else:
            # Accepted task
            # Adjust task runtime based on server utilization
            Ur = self._get_utilization(farm_i, server_i, vm_i)
            scaling_factor = 4.0  # Adjust based on desired responsiveness
            adjusted_runtime = t.original_runtime / (1 + Ur * scaling_factor)
            t.runtime = adjusted_runtime
            t.endtime = time.time() + t.runtime
            self.total_tasks+=1

            # Update server state with task assignment
            self._updateServerState(farm_i, server_i, vm_i, t)
            self.VMtask[farm_i][server_i][vm_i].append(t)
            t.status = 2  # Running
            self.task.pop(0)

            # Compute resource penalty
            # Removed resource_penalty as per request
            # resource_penalty = self.alpha * (t.CPU + t.RAM)

            # Calculate current power and cost
            current_power, current_cost = self._calculate_power_and_cost()
            
            # Calculate incremental power and cost
            delta_power = current_power - self.last_power_measurement
            delta_cost = current_cost - self.last_cost_measurement

            # Update cumulative metrics
            self.cumulative_power += delta_power
            self.cumulative_elec_cost += delta_cost

            # Update cumulative energy (Energy = Power * Time)
            # Assuming the time step is the task's adjusted runtime
            self.cumulative_energy += current_power * t.runtime

            # Update last measurements
            self.last_power_measurement = current_power
            self.last_cost_measurement = current_cost

            # Calculate idle power and penalty
            idle_power = self._calculate_idle_power()
            idle_penalty = self.beta * idle_power

            # Calculate energy consumption penalty
            energy_penalty = self.gamma * (current_power * t.runtime)

            # Adjust reward with penalties
            # Removed resource_penalty from reward
            reward = 1.0 - idle_penalty - energy_penalty

            # Update metrics for CPU/RAM usage
            self.total_accepted_cpu += t.CPU
            self.total_accepted_ram += t.RAM
            self.accepted_tasks_count += 1

            print("[DEBUG] Task accepted. CPU:", t.CPU, "RAM:", t.RAM,
                  "Idle Penalty:", idle_penalty,
                  "Energy Penalty:", energy_penalty,
                  "Final Reward:", reward,
                  "Tasks left:", len(self.task))
            # print(f"[DEBUG] Cumulative Power: {self.cumulative_power}, "
            #       f"Cumulative Elec Cost: {self.cumulative_elec_cost}, "
            #       f"Cumulative Energy: {self.cumulative_energy}")
            # print(f"[DEBUG] Idle Power: {idle_power}, Idle Penalty: {idle_penalty}")

        # Generate the next queue of tasks
        self._generateQueue()
        done = (len(self.task) == 0)
        if done:
            # Calculate final metrics
            if self.accepted_tasks_count > 0:
                avg_cpu_usage = self.total_accepted_cpu / self.accepted_tasks_count
                avg_ram_usage = self.total_accepted_ram / self.accepted_tasks_count
            else:
                avg_cpu_usage = 0
                avg_ram_usage = 0

            info = {
                'EpElecCost': self.cumulative_elec_cost,
                'EpPowerUsed': self.cumulative_power,
                'EpEnergyConsumed': self.cumulative_energy,
                'EpTasksRejected': self.rej,
                'AvgCPUUsagePerTask': avg_cpu_usage,
                'AvgRAMUsagePerTask': avg_ram_usage
            }
            print("[EPISODE DONE!!!!] . TOTAL TASKS:", self.total_tasks,
                  "Total Power Used:", self.cumulative_power, 
                  "Total Electricity Cost:", self.cumulative_elec_cost,
                  "Total Energy Consumed:", self.cumulative_energy,
                  "Total Tasks Rejected:", self.rej,
                  "Avg CPU usage:", avg_cpu_usage,
                  "Avg RAM usage:", avg_ram_usage)
        else:
            info = {}

        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment's state.
        """
        print(f"Tasks left: {len(self.task)} | Cumulative Power: {self.cumulative_power} | "
              f"Cumulative Electricity Cost: {self.cumulative_elec_cost} | "
              f"Cumulative Energy Consumed: {self.cumulative_energy} | Rejections: {self.rej}")

    def close(self):
        pass

    def _generateQueue(self):
        """
        Generate a queue of tasks that are ready to be processed.
        """
        self.task = []
        self.dag.task = []
        self.dag.taskQueue()
        for t in self.dag.task:
            if t.status == 1 and self._hasParent(t) == False:
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
        """
        # print("[DEBUG] Setting farms and servers.")
        self.farmOri = []
        self.remainFarm = []
        self.FarmResources = []
        self.VMtask = []

        f = self.servers_per_farm
        for _ in range(self.farmNum):
            farm_servers = []
            for _ in range(f):
                # Each VM: [CPU_capacity, RAM_capacity], initially all 1/VMNum
                server_vms = [[1.0/self.VMNum, 1.0/self.VMNum] for _ in range(self.VMNum)]
                farm_servers.append(server_vms)
            self.remainFarm.append(farm_servers)
            self.FarmResources.append([f, f])  # CPU, RAM farm-level resources
            self.farmOri.append(f)

        self.pwrPre = [0]*self.severNum
        self.pwrPFarm = [0]*self.farmNum
        self.VMtask = [[[[] for _ in range(self.VMNum)] for _ in range(self.servers_per_farm)] for _ in range(self.farmNum)]
        # print("[DEBUG] Farms set up complete. Farms:", self.farmNum, "Servers per farm:", f)

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
                    arr.extend(vm)
        return np.array(arr, dtype=np.float32)

    def _checkRej(self, farm_i, server_i, vm_j, task):
        """
        Check if a task can be assigned to a specific VM. Returns rejection code.
        
        Returns:
            -1: Rejected due to deadline or oversize.
             1: Rejected due to insufficient resources.
             0: Accepted.
        """
        # Check if task is oversize first
        if task.CPU > 1/self.VMNum or task.RAM > 1/self.VMNum:
            # self.rej += 1
            # print("[DEBUG] _checkRej: Task oversize. CPU:", task.CPU, "RAM:", task.RAM, "Limit:", 1/self.VMNum)
            return -1

        remain_cpu = self.remainFarm[farm_i][server_i][vm_j][0] - float(task.CPU)
        remain_ram = self.remainFarm[farm_i][server_i][vm_j][1] - float(task.RAM)
        curtime = time.time()

        if curtime + task.runtime > task.ddl:
            # self.rej += 1
            # print("[DEBUG] _checkRej: Task missed deadline. Current time:", curtime, "Deadline:", task.ddl)
            return -1

        if remain_cpu < 0 or remain_ram < 0:
            # print("[DEBUG] _checkRej: Insufficient resources.",
                #   "Remain CPU:", remain_cpu, "Remain RAM:", remain_ram)
            # self.rej += 1
            return 1

        # If here, task is feasible
        return 0

    def _updateServerState(self, farm_i, server_i, vm_numb, task):
        """
        Update the server's remaining resources after assigning a task.
        """
        old_cpu = self.remainFarm[farm_i][server_i][vm_numb][0]
        old_ram = self.remainFarm[farm_i][server_i][vm_numb][1]

        self.remainFarm[farm_i][server_i][vm_numb][0] -= float(task.CPU)
        self.remainFarm[farm_i][server_i][vm_numb][1] -= float(task.RAM)
        self.FarmResources[farm_i][0] -= float(task.CPU)
        self.FarmResources[farm_i][1] -= float(task.RAM)

        # print(f"[DEBUG] _updateServerState: VM before CPU:{old_cpu}, RAM:{old_ram}; "
            #   f"after CPU:{self.remainFarm[farm_i][server_i][vm_numb][0]}, "
            #   f"RAM:{self.remainFarm[farm_i][server_i][vm_numb][1]}")

    def _releaseByTime(self, farm_i, server_i, vm_j):
        """
        Release tasks that have completed execution based on the current time.
        """
        curtime = time.time()
        finished_tasks = []
        for t in self.VMtask[farm_i][server_i][vm_j]:
            if t.endtime < curtime:
                t.status = 0
                # Free resources
                old_cpu = self.remainFarm[farm_i][server_i][vm_j][0]
                old_ram = self.remainFarm[farm_i][server_i][vm_j][1]

                self.remainFarm[farm_i][server_i][vm_j][0] += float(t.CPU)
                self.remainFarm[farm_i][server_i][vm_j][1] += float(t.RAM)
                self.FarmResources[farm_i][0] += float(t.CPU)
                self.FarmResources[farm_i][1] += float(t.RAM)
                finished_tasks.append(t)

                # print(f"[DEBUG] _releaseByTime: Task {t.index} finished. VM CPU before:{old_cpu}, "
                    #   f"after:{self.remainFarm[farm_i][server_i][vm_j][0]}")

        for ft in finished_tasks:
            self.VMtask[farm_i][server_i][vm_j].remove(ft)
            self.dag.updateStatus(ft)
        if finished_tasks:
            print("[DEBUG] Released", len(finished_tasks), "finished tasks at farm:", farm_i, 
                  "server:", server_i, "vm:", vm_j)

    def _calculate_power_and_cost(self):
        """
        Calculate total power and electricity cost based on current resource usage.
        
        Returns:
            total_power (float): Total power consumption across all servers.
            ep_elec_cost (float): Total electricity cost.
        """
        # Calculate instantaneous power usage for each server and sum it up
        server_pwrs = []
        for farm_i in range(self.farmNum):
            for s_i in range(self.servers_per_farm):
                # Sum VM CPU resources for this server to determine utilization
                server_cpu_remain = sum([self.remainFarm[farm_i][s_i][v_i][0] for v_i in range(self.VMNum)])
                c = 1.0  # Total CPU capacity
                pwr = self._getPwr(server_cpu_remain, c)
                server_pwrs.append(pwr)

        total_power = sum(server_pwrs)
        # Convert power to electricity cost
        # Simple model: price tiers at threshold
        ep_elec_cost = self._elecPrice(1, total_power)

        return total_power, ep_elec_cost

    def _getPwr(self, r, c):
        """
        Calculate the power consumption of a server based on resource utilization.
        
        Args:
            r (float): Remaining CPU capacity of the server.
            c (float): Total CPU capacity of the server.
            
        Returns:
            float: Calculated power consumption.
        """
        alpha = 0.5
        beta_pwr = 10  # Renamed to avoid confusion with idle penalty coefficient
        Ur = (c - r) / c  # Utilization ratio

        # Idle baseline:
        if Ur < 0.1:
            # Very low utilization = idle
            pwrS = 0.8  # idle power baseline
        else:
            pwrS = 1.0  # baseline when server is active

        # Dynamic power usage above baseline:
        if Ur < 0.7:
            pwrDy = alpha * Ur
        else:
            pwrDy = 0.7 * alpha + (Ur - 0.7)**2 * beta_pwr

        return pwrDy + pwrS

    def _elecPrice(self, t, pwr):
        """
        Calculate electricity cost based on power consumption.
        
        Args:
            t (float): Time period (not used in current model).
            pwr (float): Power consumption.
            
        Returns:
            float: Electricity cost.
        """
        threshold = 1.5
        if pwr < threshold:
            p = 5.91
        else:
            p = 8.27
        return pwr * p

    def _calculate_idle_power(self):
        """
        Calculate the total power consumed by idle servers.
        
        Returns:
            float: Total idle power consumption.
        """
        idle_power = 0.0
        for farm_i in range(self.farmNum):
            for s_i in range(self.servers_per_farm):
                # Sum VM CPU resources for this server to determine utilization
                server_cpu_remain = sum([self.remainFarm[farm_i][s_i][v_i][0] for v_i in range(self.VMNum)])
                c = 1.0  # Total CPU capacity
                Ur = (c - server_cpu_remain) / c  # Utilization ratio

                if Ur < 0.1:  # Threshold for considering a server idle
                    idle_power += 0.8  # Idle power baseline
        return idle_power

    def _get_utilization(self, farm_i, server_i, vm_j):
        """
        Get the current CPU utilization ratio of a specific VM.
        
        Args:
            farm_i (int): Farm index.
            server_i (int): Server index within the farm.
            vm_j (int): VM index within the server.
            
        Returns:
            float: Utilization ratio.
        """
        remaining_cpu = self.remainFarm[farm_i][server_i][vm_j][0]
        total_cpu = 1.0  # As per initialization
        Ur = (total_cpu - remaining_cpu) / total_cpu
        return Ur
