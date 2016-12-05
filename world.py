import lane
import car
import math
import feature
import dynamics
import utils
import sys
import theano as th
import theano.tensor as tt
import numpy as np
import pickle
from car import Car

def denso():
    T = 2
    dyn = dynamics.CarDynamics(0.1)
    w = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.14)
    w.roads = [lane.StraightLane([-.07, -1.], [-.07, 1.], 0.14)]
    w.lanes = [clane.shifted(2), clane.shifted(1), clane, clane.shifted(-1)]
    w.fences = [clane.shifted(3), clane.shifted(-2)]
    w.cars.append(car.SimpleOptimizerCar(dyn, [-.035, 0., math.pi/2., 1.], color='red', T=T))
    w.cars.append(car.SimpleOptimizerCar(dyn, [.035, 0., math.pi/2., 1.], color='red', T=T))
    return w

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)

class World(object):
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
    def features(self, theta, exclude_cars=[], traj='linear'):
        if isinstance(exclude_cars, Car):
            exclude_cars = [exclude_cars]
        r  = theta[0]*sum(lane.gaussian() for lane in self.lanes)
        r += theta[1]*sum(fence.gaussian() for fence in self.fences)
        r += theta[2]*sum(road.gaussian(10) for road in self.roads)
        r += theta[3]*feature.control([(-1., 1.), (-1., 1.)])
        r += theta[4]*feature.speed(1.)
        r += theta[5]*sum(getattr(car, traj).gaussian() for car in self.cars if car not in exclude_cars)
        return r

theta_normal     = [1., -50., 10., 100., 10., -50.]
theta_no_speed   = [1., -50., 10., 100., 0. , -50.]
theta_aggressive = [1., -50., 10., 100., 30., -50.]
theta_timid      = [1., -50., 10., 100., 5. , -50.]
theta_distracted = [1., -50., 10., 100., 10., -20.]
theta_attentive  = [1., -50., 10., 100., 10., -70.]
theta_distracted1= [1., -50., 10., 100., 1., 0.]
theta_distracted2= [1., -50., 10., 100., 10., -10.]

theta0 = [1., -50., 10., 100., 10., -30.]
theta1 = [1., -50., 10., 100., 10., -70.]

def highway():
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes = [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads = [clane]
    world.fences = [clane.shifted(2), clane.shifted(-2)]
    return world

def world0(active=True, theta_explore=100., theta_exploit=1.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.2, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj), log_p=3.)
    @feature.feature
    def left_lane(t, x, u):
        return -(x[0]+0.13)**2
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear')+theta_exploit*left_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def parking1(active=True, theta_explore=100., theta_exploit=1.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0., math.pi/2., .2], color='red', T=T))
    world.cars.append(car.Car(dyn, [0.13, 0., math.pi/2., 0.], color='red', T=T))
    world.cars.append(car.Car(dyn, [0.13, .1, math.pi/2., 0.], color='red', T=T))
    world.cars.append(car.Car(dyn, [0.13, .7, math.pi/2., 0.], color='red', T=T))
    world.cars.append(car.Car(dyn, [0.13, .9, math.pi/2., 0.], color='red', T=T))
    @feature.feature
    def right_lane(t, x, u):
        return 100.*np.exp(-((x[0] - .20)**2 + (x[1] - .3)**2)/.02) - 50*x[2]-math.pi/2
    world.cars[0].reward = world.features(theta_normal, world.cars[0], 'linear') + right_lane
    return world

def merge1(active=True, theta_explore=100., theta_exploit=1.):
    theta_fence     = [1., -50., 10., 100., 10., -100.]
    T = 10
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    @feature.feature
    def right_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    world.cars[0].reward = world.features(theta_fence, world.cars[0], 'linear') + right_lane
    return world

def merge2(active=True, theta_explore=100., theta_exploit=1.):
    theta_fence     = [1., -50., 10., 100., 10., -100.]
    T = 10
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0., math.pi/2., 1.], color='yellow', T=T))
    @feature.feature
    def right_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_fence, world.cars[0], 'linear') + right_lane
    world.cars[1].reward = world.features(theta_fence, world.cars[1], 'linear') + center_lane
    return world

def merge5(active=True, theta_explore=100., theta_exploit=10.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [.0, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.0, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    @feature.feature
    def left_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_distracted, world.cars[0], 'linear') + right_lane
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj) + right_lane)
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj) + left_lane)
    obj0 = world.cars[1].traj.total(world.features(theta_attentive, world.cars[1], 'linear')+theta_exploit*center_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def merge310_passive():
    return merge310(False)

def merge310(active=True, theta_explore=100., theta_exploit=10.):
    T = 10
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.00, 0.0, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    @feature.feature
    def left_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_normal, world.cars[0], 'linear') + right_lane
    world.cars[1].add_model(lambda traj: world.features(theta_normal, world.cars[0], traj) + right_lane)
    world.cars[1].add_model(lambda traj: world.features(theta_normal, world.cars[0], traj) + left_lane)
    obj0 = world.cars[1].traj.total(world.features(theta_attentive, world.cars[1], 'linear')+theta_exploit*center_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def merge_symmetric_simple(active=True, theta_explore=100., theta_exploit=10.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.00, 0.0, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    @feature.feature
    def left_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return 1.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + right_lane
    world.cars[1].reward = world.features(theta_attentive, world.cars[1], 'linear') + center_lane
    return world

def merge_symmetric(active=True, theta_explore=100., theta_exploit=10.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.00, 0.0, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    @feature.feature
    def left_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return 1.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + right_lane
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj) + right_lane)
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj) + left_lane)
    obj0 = world.cars[1].traj.total(world.features(theta_attentive, world.cars[1], 'linear')+theta_exploit*center_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def merge_symmetric_passive():
    return merge_symmetric(active=False)

def merge_symmetric_active():
    return merge_symmetric(active=True)

def slow_merge_simple(active=True, theta_explore=100., theta_exploit=10.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., .8], color='red', T=T))
    @feature.feature
    def right_lane(t, x, u):
        return (1-np.exp(-t/5.))*5.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + right_lane
    return world

def merge3(active=True, theta_explore=100., theta_exploit=10.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.00, 0.0, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    @feature.feature
    def left_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 50.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_normal, world.cars[0], 'linear') + right_lane
    world.cars[1].add_model(lambda traj: world.features(theta_normal, world.cars[0], traj) + right_lane)
    world.cars[1].add_model(lambda traj: world.features(theta_normal, world.cars[0], traj) + left_lane)
    obj0 = world.cars[1].traj.total(world.features(theta_attentive, world.cars[1], 'linear')+theta_exploit*center_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def merge4(active=True, theta_explore=100., theta_exploit=1.):
    theta_fence     = [1., -50., 10., 100., 10., -100.]
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [.0, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [-0.13, 0.0, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    @feature.feature
    def left_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 100.*np.exp((-0.5*(x[0])**2)/.04)
    world.cars[0].reward = world.features(theta_fence, world.cars[0], 'linear') + right_lane
    world.cars[1].add_model(lambda traj: world.features(theta_fence, world.cars[0], traj) + right_lane)
    world.cars[1].add_model(lambda traj: world.features(theta_fence, world.cars[0], traj) + left_lane)
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear')+theta_exploit*center_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world


def merge3_passive():
    return merge3(active=False)
def merge3_active():
    return merge3()

def single(active=False):
    T = 5
    theta_explore = 100.
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes = [clane]
    world.roads = [clane]
    world.fences = [clane.shifted(1), clane.shifted(-1)]

    theta_normal = [1., -50., 10., 100., 10., -70.]
    theta_speedy = [1., -50., 10., 100., 30., -70.]

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0., math.pi/2., .8], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.2, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    world.cars[0].reward = world.features(theta_speedy, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_normal, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_speedy, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def single_active():
    return single(active=True)



def make_world0(version, explore=100., exploit=10.):
    def active():
        return world0(True, explore, exploit)
    def passive():
        return world0(False, explore, exploit)
    globals()['world0_{}_active'.format(version)]=active
    globals()['world0_{}_passive'.format(version)]=passive

make_world0(1, 100., 100.)

def world_test():
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.8], color='yellow'))
    world.cars[1].reward = world.features(theta0, world.cars[1])
    return world

def perturb():
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.1, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    world.cars[0].reward = world.features(theta_distracted1, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
    world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
    return world

def world1(active=True, model='human'):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    if model=='human':
        world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.1, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    if model=='attentive':
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    elif model=='distracted':
        world.cars[0].reward = world.features(theta_distracted1, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
    if model=='human':
        if active:
            world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
        else:
            world.cars[1].objective = obj0
    else:
        with open('robot/S1.{}'.format('active' if active else 'passive')) as f:
            ur = pickle.load(f)[0][1]
        world.cars[1].objective = ur
        world.cars[1].dumb = True
    return world

def world1_active():
    return world1(True)
def world1_passive():
    return world1(False)
def world1_active_attentive():
    return world1(True, 'attentive')
def world1_active_distracted():
    return world1(True, 'distracted')
def world1_passive_attentive():
    return world1(False, 'attentive')
def world1_passive_distracted():
    return world1(False, 'distracted')

def world2(active=True, model='human'):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    if model=='human':
        world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [-0.13, 0.3, math.pi/2., 1.], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    if model=='attentive':
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    elif model=='distracted':
        world.cars[0].reward = world.features(theta_distracted1, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
    if model=='human':
        if active:
            world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
        else:
            world.cars[1].objective = obj0
    else:
        with open('robot/S2.{}'.format('active' if active else 'passive')) as f:
            ur = pickle.load(f)[0][1]
        world.cars[1].objective = ur
        world.cars[1].dumb = True
    return world

def world2_active():
    return world2(True)
def world2_passive():
    return world2(False)
def world2_active_attentive():
    return world2(True, 'attentive')
def world2_active_distracted():
    return world2(True, 'distracted')
def world2_passive_attentive():
    return world2(False, 'attentive')
def world2_passive_distracted():
    return world2(False, 'distracted')


def world3(active=True, model='human'):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes = [vlane, hlane]
    world_r =  World()
    world_r.lanes = [hlane]
    world_r.roads = [hlane]
    world_r.fences = [hlane.shifted(-2), hlane.shifted(2)]
    world_h = World()
    world_h.lanes = [vlane]
    world_h.roads = [vlane]
    world_h.fences = [vlane.shifted(-2), vlane.shifted(2)]
    if model=='human':
        world.cars.append(car.UserControlledCar(dyn, [0., -0.4, math.pi/2., .5], color='red', T=T))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.4, math.pi/2., .5], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [-0.15, 0., 0., 0.05], color='yellow', T=T))
    world_h.cars = world.cars
    world_r.cars = world.cars
    world.cars[1].human = world.cars[0]
    if model=='attentive':
        world.cars[0].reward = world_h.features(theta_attentive, world.cars[0], 'linear')
    elif model=='distracted':
        world.cars[0].reward = world_h.features(theta_distracted2, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world_h.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world_h.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world_r.features(theta_no_speed, world.cars[1], 'linear'))
    if model=='human':
        if active:
            world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
        else:
            world.cars[1].objective = obj0
    else:
        with open('robot/S3.{}'.format('active' if active else 'passive')) as f:
            ur = pickle.load(f)[0][1]
        world.cars[1].objective = ur
        world.cars[1].dumb = True
    return world

def world3_active():
    return world3(True)
def world3_passive():
    return world3(False)
def world3_active_attentive():
    return world3(True, 'attentive')
def world3_active_distracted():
    return world3(True, 'distracted')
def world3_passive_attentive():
    return world3(False, 'attentive')
def world3_passive_distracted():
    return world3(False, 'distracted')

# December 12th, Merge Experiment {{{

def merge50_12_2(merge=True, multi=False, belief=True, active=True, collab=False, T=5, hold_weight=50., theta_explore=100., theta_exploit=1.):
    dyn = dynamics.CarDynamics(0.1)
    world = highway()

    @feature.feature
    def left_lane(t, x, u):
        return hold_weight*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def right_lane(t, x, u):
        return hold_weight*np.exp((-0.5*(x[0]-0.13)**2)/.04)
    @feature.feature
    def center_lane(t, x, u):
        return 50*np.exp((-0.5*(x[0])**2)/.04)

    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., .8], color='red', T=T))

    if multi:
        if belief:
            world.cars.append(car.BeliefOptimizerCar(dyn, [0.00, 0.0, math.pi/2., .8], color='yellow', T=T))
            world.cars[1].human = world.cars[0]
            if collab:
                world.cars[1].collab = True
        else:
            world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2, .8], color="yellow", T=T))

        if belief:
            world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj) + left_lane)
            world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj) + right_lane)
            obj0 = world.cars[1].traj.total(world.features(theta_attentive, world.cars[1], 'linear')+theta_exploit*center_lane)
            if active:
                world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
            else:
                world.cars[1].objective = obj0
        else:
            world.cars[1].reward = world.features(theta_attentive, world.cars[1], 'linear') + center_lane
    if merge:
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + right_lane
    else:
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + left_lane
    return world

def merge50_12_2_1simple():
    return merge50_12_2(multi=False)
def merge50_12_2_2simple():
    return merge50_12_2(multi=True, belief=False)
def merge50_12_2_2passive():
    return merge50_12_2(multi=True, belief=True, active=False)
def merge50_12_2_2active():
    return merge50_12_2(multi=True, belief=True, active=True)
def merge50_12_2_2passive_collab():
    return merge50_12_2(multi=True, belief=True, collab=True, active=False)
def merge50_12_2_2active_collab():
    return merge50_12_2(multi=True, belief=True, collab=True, active=True)

def merge50_12_2_1simple_subtle():
    return merge50_12_2(multi=False, hold_weight=5)
def merge50_12_2_2simple_subtle():
    return merge50_12_2(multi=True, belief=False, hold_weight=5)
def merge50_12_2_2passive_subtle():
    return merge50_12_2(multi=True, belief=True, active=False, hold_weight=5)
def merge50_12_2_2active_subtle():
    return merge50_12_2(multi=True, belief=True, active=True, hold_weight=5)
def merge50_12_2_2passive_subtle_collab():
    return merge50_12_2(multi=True, belief=True, collab=True, active=False, hold_weight=5)
def merge50_12_2_2active_subtle_collab():
    return merge50_12_2(multi=True, belief=True, collab=True, active=True, hold_weight=5)

def subtle_solo():
    return merge50_12_2(hold_weight=5, merge=True, multi=False, belief=False, active=False, collab=False)
def hold_solo():
    return merge50_12_2(hold_weight=5, merge=False, multi=False, belief=False, active=False, collab=False)

def subtle_merge_passive():
    return merge50_12_2(hold_weight=5, merge=True, multi=True, belief=True, active=False, collab=False)
def subtle_merge_active():
    return merge50_12_2(hold_weight=5, merge=True, multi=True, belief=True, active=True, collab=False)
def subtle_merge_passive_collab():
    return merge50_12_2(hold_weight=5, merge=True, multi=True, belief=True, active=False, collab=True)
def subtle_merge_active_collab():
    return merge50_12_2(hold_weight=5, merge=True, multi=True, belief=True, active=True, collab=True)

def subtle_hold_passive():
    return merge50_12_2(hold_weight=5, merge=False, multi=True, belief=True, active=False, collab=False)
def subtle_hold_active():
    return merge50_12_2(hold_weight=5, merge=False, multi=True, belief=True, active=True, collab=False)
def subtle_hold_passive_collab():
    return merge50_12_2(hold_weight=5, merge=False, multi=True, belief=True, active=False, collab=True)
def subtle_hold_active_collab():
    return merge50_12_2(hold_weight=5, merge=False, multi=True, belief=True, active=True, collab=True)

# }}}

# December 12th, Speed Experiment {{{

def speed_12_2(speedy=True, multi=False, belief=True, collab=False, active=True, T=5, hold_weight=50., theta_explore=100.):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes = [clane]
    world.roads = [clane]
    world.fences = [clane.shifted(1), clane.shifted(-1)]

    theta_normal = [1., -50., 10., 100., 10., -70.]
    theta_speedy = [1., -50., 10., 100., 20., -70.]

    world.cars.append(car.SimpleOptimizerCar(dyn, [.0, 0., math.pi/2., .7], color='red', T=T))

    if multi:
        if belief:
            world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.2, math.pi/2., .7], color='yellow', T=T))
            world.cars[1].human = world.cars[0]
            if collab:
                world.cars[1].collab = True
        else:
            world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.2, math.pi/2, .7], color="yellow", T=T))

        if belief:
            world.cars[1].add_model(lambda traj: world.features(theta_normal, world.cars[0], traj))
            world.cars[1].add_model(lambda traj: world.features(theta_speedy, world.cars[0], traj))
            obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
            if active:
                world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
            else:
                world.cars[1].objective = obj0
        else:
            world.cars[1].reward = world.features(theta_normal, world.cars[1], 'linear')

    if speedy:
        world.cars[0].reward = world.features(theta_speedy, world.cars[0], 'linear')
    else:
        world.cars[0].reward = world.features(theta_normal, world.cars[0], 'linear')
    return world

def speed_12_2_1simple():
    return speed_12_2(multi=False, hold_weight=5)
def speed_12_2_2simple():
    return speed_12_2(speedy=True, multi=True, belief=False, hold_weight=5)
def speed_12_2_2passive():
    return speed_12_2(multi=True, belief=True, active=False, hold_weight=5)
def speed_12_2_2active():
    return speed_12_2(multi=True, belief=True, active=True, hold_weight=5)
def speed_12_2_2passive_collab():
    return speed_12_2(multi=True, belief=True, active=False, collab=True, hold_weight=5)
def speed_12_2_2active_collab():
    return speed_12_2(multi=True, belief=True, active=True, collab=True, hold_weight=5)

# }}}

def mergex(exit=True, stop=False, multi=False, belief=False, active=True, collab=False, T=5, hold_weight=10., theta_explore=100., theta_exploit=1.):
    dyn = dynamics.CarDynamics(0.1)
    world = highway()

    @feature.feature
    def left_lane(t, x, u):
        return hold_weight*np.exp((-0.5*(x[0]+0.13)**2)/.04)
    @feature.feature
    def merge_lane(t, x, u):
        return hold_weight*np.exp(-0.5*(((x[0])**2)/.04 + ((x[1] - 1.5)**2)/.4))

    if not stop: #original
        ## NICK THIS IS THE ORIGINAL EXIT THAT YOUO ARE CURRENTLY RUNNING
        def merge_exit(t, x, u):
            return hold_weight*np.exp(-0.5*(((x[0]-.13)**2)/.04 + ((x[1] - 1.5)**2)/.4))
    else: # should sstopp
        # THIS IS THE THING YOU WANT TO TRY NEXT
        def merge_exit(t, x, u):
            return hold_weight*np.exp(-0.5*(((x[0]-.13)**2)/.04 + ((x[1] - 1.5)**2)/.4)) + tt.ge(x[0], 0.065)*-10*x[3]
    def merge_exit_later(t, x, u):
        return hold_weight*np.exp(-0.5*(((x[0]-.13)**2)/.04 + ((x[1] - 2.5)**2)/.4))
    @feature.feature
    def center_lane(t, x, u):
        return 12*np.exp((-0.5*(x[0])**2)/.04)

    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.0, math.pi/2., .8], color='red', T=T))

    if multi:
        if belief:
            world.cars.append(car.BeliefOptimizerCar(dyn, [0.00, 0.0, math.pi/2., .8], color='yellow', T=T))
            world.cars[1].human = world.cars[0]
            if collab:
                world.cars[1].collab = True
        else:
            world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.0, math.pi/2, .8], color="yellow", T=T))

        if belief:
            world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj) + merge_lane)
            world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj) + merge_exit)
            obj0 = world.cars[1].traj.total(world.features(theta_attentive, world.cars[1], 'linear')+theta_exploit*center_lane)
            if active:
                world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
            else:
                world.cars[1].objective = obj0
        else:
            world.cars[1].reward = world.features(theta_attentive, world.cars[1], 'linear') + center_lane

    if exit:
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + merge_exit
    else:
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear') + merge_lane
    return world

def exit_passive():
    return mergex(exit=True, multi=True, belief=True, active=False, collab=False)
def exit_active():
    return mergex(exit=True, multi=True, belief=True, active=True, collab=False)
def exit_passive_collab():
    return mergex(exit=True, multi=True, belief=True, active=False, collab=True)
def exit_active_collab():
    return mergex(exit=True, multi=True, belief=True, active=True, collab=True)

def merge_passive():
    return mergex(exit=False, multi=True, belief=True, active=False, collab=False)
def merge_active():
    return mergex(exit=False, multi=True, belief=True, active=True, collab=False)
def merge_passive_collab():
    return mergex(exit=False, multi=True, belief=True, active=False, collab=True)
def merge_active_collab():
    return mergex(exit=False, multi=True, belief=True, active=True, collab=True)

def stop_passive():
    return mergex(exit=True, stop=True, multi=True, belief=True, active=False, collab=False)
def stop_active():
    return mergex(exit=True, stop=True, multi=True, belief=True, active=True, collab=False)
def no_stop_passive():
    return mergex(exit=False, stop=True, multi=True, belief=True, active=False, collab=False)
def no_stop_active():
    return mergex(exit=False, stop=True, multi=True, belief=True, active=True, collab=False)

def stop_solo():
    return mergex(exit=True, stop=True, multi=False, belief=False, active=False, collab=False)

def exit_solo():
    return mergex(exit=True, multi=False, belief=False, active=False, collab=False)
def merge_solo():
    return mergex(exit=False, multi=False, belief=False, active=False, collab=False)

def exit_no_inference():
    return mergex(exit=True, multi=True, belief=False, active=False, collab=False)
def merge_no_inference():
    return mergex(exit=False, multi=True, belief=False, active=False, collab=False)
