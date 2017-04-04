import numpy as np
import itertools
import matplotlib
#matplotlib.use('WebAgg')
#matplotlib.rcParams['webagg.open_in_browser']=True
#matplotlib.rcParams['webagg.port']=12345
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from matplotlib.image import BboxImage, AxesImage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Slider, Button, RadioButtons
import math
import os, signal

GRASS = np.tile(plt.imread('images/grass.png'), (3, 3, 1))

CAR = {
    color: zoom(plt.imread('images/car-{}.png'.format(color)), [0.3, 0.3, 1.])
    for color in ['gray', 'orange', 'purple', 'red', 'white', 'yellow']
}
CAR_HUMAN = CAR['orange']
CAR_ROBOT = CAR['white']
CAR_SCALE = 0.15/max(CAR.values()[0].shape[:2])

LANE_SCALE = 10.
LANE_COLOR = (0.4, 0.4, 0.4) # 'gray'
LANE_BCOLOR = 'white'

STEPS = 15


def set_image(obj, data, scale=CAR_SCALE, x=[0., 0., 0., 0.]):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent([ox-scale*w*0.5, ox+scale*w*0.5, oy-scale*h*0.5, oy+scale*h*0.5])

class Scene(object):
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value
        set_image(self.human, CAR_HUMAN, x=self.snapshot.human.ix(self.t))
        if self.no_human==False:
            xs = self.snapshot.human.ix(self.ts[self.ts<=self.t])
            self.h_past.set_data(xs[:, 0], xs[:, 1])
            xs = self.snapshot.human.ix(self.ts[self.ts>=self.t])
            self.h_future.set_data(xs[:, 0], xs[:, 1])
        for r_past, r_future, snapshot_robot in zip(self.r_past, self.r_future, self.snapshot.robots):
            xs = snapshot_robot.ix(self.ts[self.ts<=self.t])
            r_past.set_data(xs[:, 0], xs[:, 1])
            xs = snapshot_robot.ix(self.ts[self.ts>=self.t])
            r_future.set_data(xs[:, 0], xs[:, 1])
        for robot, traj in zip(self.robots, self.snapshot.robots):
            set_image(robot, CAR_ROBOT, x=traj.ix(self.t))
    def center(self, cx=0., cy=0., wx=1., wy=1.):
        self.ax.set_xlim(cx-wx/2., cx+wx/2.)
        self.ax.set_ylim(cy-wy/2., cy+wy/2.)
    def __init__(self, ax, snapshot, no_human=False):
        self.snapshot = snapshot
        self.ax = ax
        self.ax.set_aspect('equal', 'box-forced')
        self.no_human = no_human

        self.grass = BboxImage(ax.bbox, interpolation='bicubic', zorder=-1000)
        self.ax.add_artist(self.grass)
        self.grass.set_data(GRASS)

        for lane in self.snapshot.lanes:
            path = Path([
                lane.p-LANE_SCALE*lane.m-lane.n*lane.w*0.5,
                lane.p-LANE_SCALE*lane.m+lane.n*lane.w*0.5,
                lane.q+LANE_SCALE*lane.m+lane.n*lane.w*0.5,
                lane.q+LANE_SCALE*lane.m-lane.n*lane.w*0.5,
                lane.p-LANE_SCALE*lane.m-lane.n*lane.w*0.5
            ], [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY
            ])
            ax.add_artist(PathPatch(path, facecolor=LANE_COLOR, lw=0.5, edgecolor=LANE_BCOLOR, zorder=-100))

        self.human = AxesImage(self.ax, interpolation='bicubic', zorder=10000)
        self.ax.add_artist(self.human)
        self.ts = np.linspace(0., self.snapshot.human.T, STEPS)
        xs = self.snapshot.human.ix(self.ts)
        STYLE = {
            'marker': 'o',
            'markeredgecolor': 'none'
        }
        if self.no_human is False:
            self.h_future = ax.plot(xs[:, 0], xs[:, 1], zorder=50, linestyle='-', color='orange', linewidth=2., **STYLE)[0]
            self.h_past = ax.plot(xs[:, 0], xs[:, 1], zorder=40, linestyle='-', color='orange', linewidth=2., **STYLE)[0]

        self.robots = [AxesImage(self.ax, interpolation='bicubic', zorder=1000) for robot in self.snapshot.robots]
        self.r_future = []
        self.r_past = []
        for robot, snapshot_robot in zip(self.robots, self.snapshot.robots):
            self.ax.add_artist(robot)
            xs = snapshot_robot.ix(self.ts)
            self.r_future.append(ax.plot(xs[:, 0], xs[:, 1], zorder=50, linestyle=('-' if self.no_human else '-'), color='white', linewidth=2., **STYLE)[0])
            self.r_past.append(ax.plot(xs[:, 0], xs[:, 1], zorder=40, linestyle='-', color='white', linewidth=2., **STYLE)[0])

        #self.ax.set_xlim(-0.5, 0.5)
        #self.ax.set_ylim(-0.2, 0.8)

class CloseException(Exception): pass

class Visualizer(object):
    def __init__(self, snapshot, save_on_close=None):
        self.snapshot = snapshot
        self.save_on_close = save_on_close
        self.T = self.snapshot.human.values()[0].T
        self.choice = None
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value
        for scene in self.scenes:
            scene.t = self.t
    def adjust(self, *args, **vargs):
        for scene in self.scenes:
            scene.center(*args, **vargs)
    def select(self, key):
        rates = tuple([radio.value_selected for radio in self.radios])
        if len(rates)>0:
            self.choice = (key, rates)
        else:
            self.choice = key
        if self.save_on_close is not None:
            self.fig.savefig(self.save_on_close)
        raise KeyboardInterrupt
        plt.close(self.fig)
    def close(self, event):
        if self.save_on_close is not None:
            self.fig.savefig(self.save_on_close)
        raise CloseException('Exit')
    def run(self, rank=False, show=True, same=False, no_human=False, *args, **vargs):
        self.fig, self.ax = plt.subplots(1, len(self.snapshot.keys()), sharex=True, sharey=True, figsize=(13, 7))
        if len(self.snapshot.keys())==1:
            self.ax = [self.ax]
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.scenes = [Scene(ax, self.snapshot.view(key), no_human) for ax, key in zip(self.ax, self.snapshot.keys())]
        self.radios=[]

        if show:
            self.fig.subplots_adjust(bottom=0.15, top=0.85)

            box = self.fig.add_axes([0.15, 0.05, 0.7, 0.05])
            self.slider = Slider(box, 'Time', 0., self.T, valinit=0.)

        self.t = 0.
        if show:
            def update_t(t):
                self.t = t
            self.slider.on_changed(update_t)

        if show:
            def click(key):
                def f(event):
                    self.select(key)
                return f
            self.buttons = []
            if rank=='radio':
                for ax, key in zip(self.ax, self.snapshot.keys()):
                    box = ax.figbox
                    box = self.fig.add_axes([box.x0, box.y1-0.12, box.width, 0.25])
                    self.radios.append(RadioButtons(box, range(1,8)))
                    #self.buttons[-1].on_clicked(click(key))
            for ax, key in zip(self.ax, self.snapshot.keys()):
                if rank==True:
                    i = 0
                    for order in itertools.permutations(self.snapshot.keys()):
                        if order[0]!=key:
                            continue
                        box = ax.figbox
                        box = self.fig.add_axes([box.x0, box.y1+0.05-0.05*i, box.width, 0.05])
                        self.buttons.append(Button(box, '>'.join(order)))
                        self.buttons[-1].on_clicked(click(order))
                        i += 1
                elif rank=='radio':
                    box = ax.figbox
                    box = self.fig.add_axes([box.x0, box.y0, box.width, 0.05])
                    self.buttons.append(Button(box, 'Prefer {}'.format(key)))
                    self.buttons[-1].on_clicked(click(key))
                else:
                    box = ax.figbox
                    box = self.fig.add_axes([box.x0, box.y1+0.05, box.width, 0.05])
                    self.buttons.append(Button(box, 'Prefer {}'.format(key)))
                    self.buttons[-1].on_clicked(click(key))
        self.adjust(*args, **vargs)
        if show:
            plt.show()
    def key_press(self, event):
        if event.key=='escape':
            plt.close(self.fig)
        elif event.key=='r':
            self.slider.set_val(0.)
        elif event.key=='up':
            self.slider.set_val(min(max(self.t+0.2, 0), self.T))
        elif event.key=='down':
            self.slider.set_val(min(max(self.t-0.2, 0), self.T))
        elif event.key.lower() in [s.lower() for s in self.snapshot.keys()]:
            for key in self.snapshot.keys():
                if event.key.lower()==key.lower():
                    self.select(key)

def visualize(snapshot, *args, **vargs):
    vis = Visualizer(snapshot, False)
    vis.run(rank=False, *args, **vargs)
    return vis

def select(snapshot, save_on_close=None, rank=False):
    vis = Visualizer(snapshot, save_on_close)
    try:
        vis.run(rank=rank, cx=0., cy=0.3)
    except CloseException: pass
    return vis.choice

def test_visualizer():
    from world import world
    for v in ['A', 'B']:
        traj = world.human[v]
        traj.x[0].set_value([0., 0., math.pi/2., 0.5])
        for u in traj.u:
            u.set_value([1 if v=='A' else -1, 1])
    world.robots[0].x[0].set_value([-0.13, 0., math.pi/2., 0.5])
    print select(world.dump())

if __name__=='__main__':
    from worldvis import world, Snapshot
    s = Snapshot.load('users.back2/Dorsa/tests/4.pickle')
    s.human['B'], s.human['C'] = s.human['C'], s.human['B']
    v = visualize(s, cx=0., cy=0.3, show=False)
    v.t = 0.
    for ax in v.ax:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('plots/test0.pdf')
    v.t = 5.
    plt.savefig('plots/test1.pdf')
    quit()
    from world import world, Snapshot
    s = Snapshot.load('data/2016-09-08T18:52:43.779626.pickle')
    v = visualize(s, show=False, cx=0., cy=0.3, same=True)
    v.t = 3.2
    v.ax[0].axis('off')
    v.ax[1].axis('off')
    #v.ax[0].annotate('$\\xi_A$', (-0.1, 0.1), fontsize=20, color='orange')
    #v.ax[1].annotate('$\\xi_B$', (-0.1, 0.1), fontsize=20, color='orange')
    plt.tight_layout()
    plt.savefig('plots/example.pdf')
    quit()
    for v in ['A', 'B']:
        traj = world.human[v]
        traj.x[0].set_value([0., 0., math.pi/2., 0.5])
        for u in traj.u:
            u.set_value([1 if v=='A' else -1, 1])
    world.robots[0].x[0].set_value([-0.13, 0., math.pi/2., 0.5])
    test_visualizer()
