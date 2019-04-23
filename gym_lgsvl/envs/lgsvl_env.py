import gym
from gym import error, spaces, utils
from gym.utils import seeding
import lgsvl
import os
import random
import math

class LgsvlEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # Loading the SanFrancisco scene by default
    scene = "SanFrancisco"
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


  def step(self, action):
    self.env.run(time_limit = 0.1) # TODO: replace with single frame whenever API supports it

  def reset(self):
    self.vehicles.clear()
    self._occupied.clear()
    self.spawns.clear()
    self.env.reset()
    self.seed = seeding.create_seed()
    random.seed(self.seed)

  def render(self, mode='human'):
    pass
  
  def close(self):
    self.env.stop()

  def _setup_ego(self, name = "XE_Rigged-lgsvl", spawn_index = 0, random_spawn = False):
    state = lgsvl.AgentState()
    if (random_spawn):
      state.transform = self.spawns[random.randint(0, len(self.spawns) - 1)]
    else:
      state.transform = self.spawns[spawn_index]
    
    self.ego = self.env.add_agent(name, lgsvl.AgentType.EGO, state)
    self.vehicles[self.ego] = "EGO"
    self._occupied.append(state.transform.position)


  def _setup_npc(self, npc_type = None, position = None, follow_lane = True, speed = None,
    speed_upper = 25.0, speed_lower = 7.0, randomize = False, min_dist = 10.0, max_dist = 40.0):
    
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
    
  def _proximity(self, position1, position2):
    return math.sqrt((position1.x - position2.x)**2 + (position1.y - position2.y)**2 + (position1.z - position2.z)**2)
