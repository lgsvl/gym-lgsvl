import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import lgsvl
import os
import random
import math

class LgsvlEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, scene = "SanFrancisco", port = 8181):
    # Loading the SanFrancisco scene by default
    self.env = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
    if self.env.current_scene == scene:
      self.env.reset()
    else:
      self.env.load(scene)

    # List of spawn points in the scene
    self.spawns = self.env.get_spawn()

    # List of all vehicle names available
    self.vehicles = dict()

    # A list of all occupied coordinates (at spawn time)
    self._occupied = list()

    # Random seed used for all random generations
    self.seed = seeding.create_seed()
    random.seed(self.seed)

    self.control = lgsvl.VehicleControl()

    # continuous action space only containing the steering angle and throttle
    self.action_space = spaces.Box(low = np.array([-1.0, -1.0]), high = np.array([1.0, 1.0]), dtype=np.float32)

    self.observation_space = NotImplementedError

    
  def step(self, action):
    jsonable = self.action_space.to_jsonable(action)
    self.control.steering = jsonable[0]

    # use positive values for throttle and negative values for braking
    if (jsonable[1] > 0):
      self.control.throttle = jsonable[1]
      self.control.braking = 0.0
    else:
      self.control.throttle = 0.0
      self.control.braking = abs(jsonable[1])
    self.ego.apply_control(self.control, sticky=True)
    self.env.run(time_limit = 0.1) # TODO: replace with single frame whenever API supports it

    return [None, None, False, None]

  def reset(self):
    self.vehicles.clear()
    self._occupied.clear()
    self.spawns.clear()
    self.env.reset()
    self.seed = seeding.create_seed()
    random.seed(self.seed)
    self.spawns = self.env.get_spawn()
    self._setup_ego()
    count = random.randint(1,10)
    while count > 0:
      self._setup_npc()
      count -= 1

  def render(self, mode='human'):
    pass
  
  def close(self):
    self.env.stop()

  def _setup_ego(self, name = "XE_Rigged-lgsvl", spawn_index = 0, random_spawn = False):
    """
    Spawns ego vehicle at the specified (by default index 0) spawn point in the Unity scene.
    """
    state = lgsvl.AgentState()
    if (random_spawn):
      state.transform = self.spawns[random.randint(0, len(self.spawns) - 1)]
    else:
      state.transform = self.spawns[spawn_index]
    
    self.ego = self.env.add_agent(name, lgsvl.AgentType.EGO, state)
    self.vehicles[self.ego] = "EGO"
    self._occupied.append(state.transform.position)


  def _setup_npc(self, npc_type = None, position = None, follow_lane = True,
                 speed = None, speed_upper = 25.0, speed_lower = 7.0,
                 randomize = False, min_dist = 10.0, max_dist = 40.0):
    
    """
    Spawns an NPC vehicle of a specific type at a specific location with an
    option to have it follow lane annotations in the Unity scene at a given
    speed.

    Not specifying any input results in a random selection of NPC type, a
    random spawn location within the [min_dist, max_dist] range of the ego
    vehicle, and a random speed selected within the [speed_lower, speed_upper]
    range.
    """
    
    npc_types = {"Sedan", "HatchBack", "SUV", "Jeep", "DeliveryTruck", "SchoolBus"}
    
    if (not npc_type):
      npc_type = random.sample(npc_types, 1)[0]

    if (randomize or not position):
      sx = self.ego.transform.position.x
      sy = self.ego.transform.position.y
      sz = self.ego.transform.position.z
      
      while (not position):
        angle = random.uniform(0.0, 2*math.pi)
        dist = random.uniform(min_dist, max_dist)
        point = lgsvl.Vector(sx + dist * math.cos(angle), sy, sz + dist * math.sin(angle))
        transform = self.env.map_point_on_lane(point)

        px = transform.position.x
        py = transform.position.y
        pz = transform.position.z


        mindist = 0.0
        maxdist = 10.0
        dist = random.uniform(mindist, maxdist)
        angle = math.radians(transform.rotation.y)
        position = lgsvl.Vector(px - dist * math.cos(angle), py, pz + dist * math.sin(angle))

        for pos in self._occupied:
          if (position and self._proximity(position, pos) < 7):
            position = None
        
        
    state = lgsvl.AgentState()
    state.transform = self.env.map_point_on_lane(position)
    n = self.env.add_agent(npc_type, lgsvl.AgentType.NPC, state)

    if (follow_lane):
      if (not speed):
        speed = random.uniform(speed_lower, speed_upper)
      n.follow_closest_lane(True, speed)
  
    self.vehicles[n] = npc_type
    self._occupied.append(position)
  
  def _setup_pedestrian(self):
    # Spawn pedestrians randomly on sidewalk near ego vehicle
    NotImplementedError

  def _proximity(self, position1, position2):
    """
    Helper function for calculating Euclidean distance between two Vector objects.
    """
    return math.sqrt((position1.x - position2.x)**2 + (position1.y - position2.y)**2 + (position1.z - position2.z)**2)


