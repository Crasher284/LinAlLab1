from tkinter import *
from math import ceil
import numpy as np

class GraphGraphApp:

    root = 0
    size = []
    canvas = 0
    x, y, z = [0]*3
    r, g, b = [0]*3
    k = 15
    s = ""
    start_T, T = [[]], [[]]
    phi, theta, delta = [0]*3
    num, acc = [0]*2
    points, poligons = [[], []]
    N, M = [0]*2
    params, max_params, min_params, d_params = [[]], [[]], [[]], [[]]

    axis = True
    fill = True
    pars = False

    def __init__(self, root, size):
        self.root = root
        self.size = size

        self.canvas = Canvas(self.root, bg="white", width=self.size[0], height=self.size[1])
        self.canvas.pack(anchor=CENTER, expand=1)
        self.canvas.focus_set()
        self.start_T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.phi = np.pi/4
        self.theta = np.pi/3
        self.delta = np.radians(5)
        self.update_T()
        self.num, self.acc = 0, 2
        self.params = [[1, 0.2], [1, 0.5], [2, 0.5], [0.5, 0.5], [0.1, 0.05]]
        self.min_params = [[1, 0.2], [1, 0.5], [2, 0.5], [0.5, 0.5], [0.1, 0.05]]
        self.max_params = [[5, 1.5], [3, 2], [5, 2], [2, 2], [0.5, 0.2]]
        self.d_params = [[0.2, 0.1], [0.2, 0.1], [0.3, 0.1], [0.1, 0.1], [0.05, 0.01]]
                
        self.canvas.bind('<MouseWheel>', self.roll)
        self.canvas.bind('<KeyPress>', self.press)

        self.create()

    def curve(self, num, alpha, beta, acc):
        points = []

        match num:
            case 0:
                u_steps = ceil(4*np.pi/acc)
                v_steps = ceil(2*np.pi/acc)
                u = np.linspace(0, 4*np.pi, u_steps, endpoint=True)
                v = np.linspace(0, 2*np.pi, v_steps, endpoint=True)
                U, V = np.meshgrid(u, v, indexing='ij')
                
                x = (alpha + beta * np.cos(V)) * np.cos(U)
                y = (alpha + beta * np.cos(V)) * np.sin(U)
                z = beta * np.sin(V) + alpha * U

                points = np.dstack((x, y, z))
                
                self.N, self.M, _ = points.shape
                points = points.reshape(-1, 3)
                            
            case 1:
                u_steps = ceil(2*np.pi/acc)
                v_steps = ceil(1/acc)
                u = np.linspace(0, 2*np.pi, u_steps, endpoint=True)
                v = np.linspace(-0.5, 0.5, v_steps, endpoint=True)
                U, V = np.meshgrid(u, v, indexing='ij')
                
                x = (alpha + V * np.cos(U/2)) * np.cos(U)
                y = (alpha + V * np.cos(U/2)) * np.sin(U)
                z = beta * V * np.sin(U/2)

                points = np.dstack((x, y, z))
                
                self.N, self.M, _ = points.shape
                points = points.reshape(-1, 3)

            case 2:
                u_steps = ceil(2*np.pi/acc)
                v_steps = ceil(2*np.pi/acc)
                u = np.linspace(0, 2*np.pi, u_steps, endpoint=True)
                v = np.linspace(0, 2*np.pi, v_steps, endpoint=True)
                U, V = np.meshgrid(u, v, indexing='ij')
                
                x = (alpha + beta * np.cos(V)) * np.cos(U)
                y = (alpha + beta * np.cos(V)) * np.sin(U)
                z = beta * np.sin(V)

                points = np.dstack((x, y, z))
                
                self.N, self.M, _ = points.shape
                points = points.reshape(-1, 3)

            case 3:
                u_steps = ceil(4*np.pi/acc)
                v_steps = ceil(4/acc)
                u = np.linspace(0, 4*np.pi, u_steps, endpoint=True)
                v = np.linspace(-2, 2, v_steps, endpoint=True)
                U, V = np.meshgrid(u, v, indexing='ij')
                
                x = alpha * U * np.cos(U)
                y = beta * U * np.sin(U)
                z = V

                points = np.dstack((x, y, z))
                
                self.N, self.M, _ = points.shape
                points = points.reshape(-1, 3)

            case 4:
                u_steps = ceil(2*np.pi/acc)
                v_steps = ceil(6*np.pi/acc)
                u = np.linspace(0, 2*np.pi, u_steps, endpoint=True)
                v = np.linspace(0, 6*np.pi, v_steps, endpoint=True)
                U, V = np.meshgrid(u, v, indexing='ij')
                
                x = alpha * np.exp(beta * V) * np.cos(V) * (1 + np.cos(U))
                y = alpha * np.exp(beta * V) * np.sin(V) * (1 + np.cos(U))
                z = alpha * np.exp(beta * V) * np.sin(U)

                points = np.dstack((x, y, z))
                
                self.N, self.M, _ = points.shape
                points = points.reshape(-1, 3)

            case _:
                pass
        
        return points

    def update_T(self):
        dx = np.sin(self.theta) * np.cos(self.phi)
        dy = np.sin(self.theta) * np.sin(self.phi)
        dz = np.cos(self.theta)

        right = np.array([-dy, dx, 0])
        up = np.array([dx*dz, dy*dz, -np.sin(self.theta)**2])
        forward = np.array([dx, dy, dz])

        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)
        forward /= np.linalg.norm(forward)

        self.T = self.start_T @ np.vstack([right, up, forward])

    def to_screen(self, coords):
        crds = np.array([[coords[0]], [coords[1]], [coords[2]]])
        screen = (self.T @ crds) * self.k
        out = (self.size[0]/2 + screen[0, 0], self.size[1]/2 + screen[1, 0], screen[2, 0])
        return out

    def points_to_screen(self, points):
        screen = (self.T @ points.T).T * self.k
        screen[:, 0] += self.size[0] / 2
        screen[:, 1] += self.size[1] / 2
        return screen

    def points_to_poligons(self, points):
        points = points.reshape(self.N, self.M, 3)

        i, j = np.indices((self.N-1, self.M-1))
        polygons = np.stack([points[i, j, :2],
                             points[i, j+1, :2],
                             points[i+1, j+1, :2],
                             points[i+1, j, :2]], axis=2)

        zs = np.mean([points[i, j, 2],
                      points[i, j+1, 2],
                      points[i+1, j+1, 2],
                      points[i+1, j, 2]], axis=0)

        color_ratios = np.stack([i/(self.N-2) if self.N > 2 else np.zeros_like(i),
                                 j/(self.M-2) if self.M > 2 else np.zeros_like(j),
                                 np.zeros_like(i)], axis=-1)
        
        colors = np.round(255 * (1 - color_ratios)).astype(np.uint8)
        colors[..., 2] = 127
        colors = np.array([f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors.reshape(-1, 3)])

        dtype = [('vertices', 'f4', (4, 2)), ('depth', 'f4'), ('color', 'U7')]

        full_polygons = np.rec.fromarrays([polygons.reshape(-1, 4, 2),
                                                 zs.ravel(),
                                                 colors], dtype=dtype)

        full_polygons = full_polygons[np.argsort(-full_polygons.depth)]
        
        return full_polygons.tolist()

    def poligons_to_see(self, poligons):
        for i in range(len(poligons)):
            if self.fill:
                if poligons[i][1]>-1000:
                    self.canvas.create_polygon(poligons[i][0][0][0], poligons[i][0][0][1],
                                      poligons[i][0][1][0], poligons[i][0][1][1],
                                      poligons[i][0][2][0], poligons[i][0][2][1],
                                      poligons[i][0][3][0], poligons[i][0][3][1],
                                      fill=poligons[i][2])
            else:
                self.canvas.create_polygon(poligons[i][0][0][0], poligons[i][0][0][1],
                                      poligons[i][0][1][0], poligons[i][0][1][1],
                                      poligons[i][0][2][0], poligons[i][0][2][1],
                                      poligons[i][0][3][0], poligons[i][0][3][1],
                                      fill="", outline=poligons[i][2])                

    def draw(self, points):
        self.canvas.delete("all")

        if self.axis:
            screen = self.to_screen([1000, 0, 0])
            self.canvas.create_line(250, 250, screen[0], screen[1], fill="red")
            screen = self.to_screen([0, 1000, 0])
            self.canvas.create_line(250, 250, screen[0], screen[1], fill="green")
            screen = self.to_screen([0, 0, 1000])
            self.canvas.create_line(250, 250, screen[0], screen[1], fill="blue")
        
        self.poligons_to_see(self.points_to_poligons(self.points_to_screen(points)))

        if self.pars:
            self.canvas.create_text(5, 5, anchor=NW, text="num = "+str(self.num+1), fill="#000000")
            self.canvas.create_text(5, 15, anchor=NW, text="alpha = "+str(self.params[self.num][0]), fill="#000000")
            self.canvas.create_text(5, 25, anchor=NW, text="beta = "+str(self.params[self.num][1]), fill="#000000")
            self.canvas.create_text(5, 35, anchor=NW, text="acc = "+str(self.acc), fill="#000000")

    def create(self):
        self.points = self.curve(self.num, self.params[self.num][0], self.params[self.num][1], 1/self.acc)
        self.draw(self.points)

    def roll(self, event):
        if not self.pars:
            self.k*=pow(1.1, event.delta/120)
            self.draw(self.points)
        else:
            self.acc = max(1, self.acc+event.delta/120)
            self.create()

    def press(self, event):
        if event.keysym == 'F1' and not self.pars:
            self.axis = not self.axis
            self.draw(self.points)
        if event.keysym == 'F2' and not self.pars:
            self.fill = not self.fill
            self.draw(self.points)
        if event.keysym == 'F3':
            self.pars = not self.pars
            self.draw(self.points)
        if (event.keysym == 'w' or event.keysym == 'Up') and not self.pars:
            self.theta = max(self.delta, self.theta-self.delta)
            self.update_T()
            self.draw(self.points)
        if (event.keysym == 's' or event.keysym == 'Down') and not self.pars:
            self.theta = min(np.pi-self.delta, self.theta+self.delta)
            self.update_T()
            self.draw(self.points)
        if (event.keysym == 'a' or event.keysym == 'Left') and not self.pars:
            self.phi -= self.delta
            self.update_T()
            self.draw(self.points)
        if (event.keysym == 'd' or event.keysym == 'Right') and not self.pars:
            self.phi += self.delta
            self.update_T()
            self.draw(self.points)
        if (event.keysym == 'w' or event.keysym == 'Up') and self.pars:
            self.params[self.num][1] = round(min(self.max_params[self.num][1], self.params[self.num][1]+self.d_params[self.num][1]), 2)
            self.create()
        if (event.keysym == 's' or event.keysym == 'Down') and self.pars:
            self.params[self.num][1] = round(max(self.min_params[self.num][1], self.params[self.num][1]-self.d_params[self.num][1]), 2)
            self.create()
        if (event.keysym == 'a' or event.keysym == 'Left') and self.pars:
            self.params[self.num][0] = round(max(self.min_params[self.num][0], self.params[self.num][0]-self.d_params[self.num][0]), 2)
            self.create()
        if (event.keysym == 'd' or event.keysym == 'Right') and self.pars:
            self.params[self.num][0] = round(min(self.max_params[self.num][0], self.params[self.num][0]+self.d_params[self.num][0]), 2)
            self.create()
        if event.keysym == '1' and self.pars:
            self.num = 0
            self.create()
        if event.keysym == '2' and self.pars:
            self.num = 1
            self.create()
        if event.keysym == '3' and self.pars:
            self.num = 2
            self.create()
        if event.keysym == '4' and self.pars:
            self.num = 3
            self.create()
        if event.keysym == '5' and self.pars:
            self.num = 4
            self.create()

width, height = 500, 500
size = [width, height]

root = Tk()
root.title("Lab1")
root.geometry(str(width)+"x"+str(height))

GGA = GraphGraphApp(root, size)

root.mainloop()
