import pygame
import math
import csv
import time
import numpy as np
from collections import deque


WIDTH, HEIGHT = 1920, 1080
FPS = 60
FOV = 90
NEAR_PLANE = 0.1
FAR_PLANE = 1000


ROBOT_LENGTH = 1.0
ROBOT_WIDTH = 0.7
ROBOT_HEIGHT = 0.5
MAX_SPEED = 3.0
MAX_ANG_VEL = math.radians(90)


FIELD_SIZE = 60
GRID_RES = 1.0


LIDAR_RANGE = 12.0
LIDAR_RES_DEG = 5


BG_COLOR = (30, 30, 40)
GRID_COLOR = (80, 80, 90)
ROBOT_COLOR = (200, 150, 50)
OBST_COLOR = (180, 60, 60)
HUD_COLOR = (220, 220, 220)


def clamp(x, a, b):
    return max(a, min(b, x))


def rot_z(vec, yaw):
    x, y, z = vec
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y, z


class Camera:
    def __init__(self, width, height, fov_deg=FOV):
        self.width = width
        self.height = height
        self.fov = math.radians(fov_deg)
        self.aspect = width / height
        self.focal = (width / 2) / math.tan(self.fov / 2)
        self.pos = np.array([FIELD_SIZE + 260, FIELD_SIZE - 150, 55.0])
        self.yaw = math.radians(45)
        self.pitch = math.radians(-20)

    def project_point(self, point3):
        p = np.array(point3) - self.pos / 4
        c_yaw = math.cos(-self.yaw)
        s_yaw = math.sin(-self.yaw)
        x = c_yaw * p[0] - s_yaw * p[1]
        y = s_yaw * p[0] + c_yaw * p[1]
        z = p[2]

        c_pitch = math.cos(-self.pitch)
        s_pitch = math.sin(-self.pitch)
        y2 = c_pitch * y - s_pitch * z
        z2 = s_pitch * y + c_pitch * z
        y, z = y2, z2

        if y <= 0.01:
            return None
        px = (x * self.focal / y) + self.width / 2
        py = (-z * self.focal / y) + self.height / 2
        return (int(px), int(py), y)


class Environment:
    def __init__(self, field_size=FIELD_SIZE, grid_res=GRID_RES):
        self.size = field_size
        self.grid_res = grid_res
        self.cells = int(field_size / grid_res)
        self.heightmap = np.zeros((self.cells, self.cells))
        self.moisture = np.zeros((self.cells, self.cells))
        self.obstacles = []  # list of dict(x,y,w,l,h)
        self._generate_terrain()
        self._place_obstacles()

    def _generate_terrain(self):
        xs = np.linspace(0, math.pi * 4, self.cells)
        ys = np.linspace(0, math.pi * 4, self.cells)
        xv, yv = np.meshgrid(xs, ys)
        self.heightmap = 0.6 * np.sin(xv) * np.cos(yv) + 0.2 * np.sin(xv * 0.3 + yv * 0.5)
        h_norm = (self.heightmap - self.heightmap.min()) / (self.heightmap.max() - self.heightmap.min())
        self.moisture = 1.0 - h_norm + 0.1 * np.random.rand(self.cells, self.cells)

    def _place_obstacles(self):
        rng = np.random.RandomState(42)
        for _ in range(18):
            w = rng.uniform(0.6, 2.5)
            l = rng.uniform(0.6, 2.5)
            x = rng.uniform(2, self.size - 2 - w)
            y = rng.uniform(2, self.size - 2 - l)
            h = rng.uniform(0.6, 3.0)
            self.obstacles.append({'x': x, 'y': y, 'w': w, 'l': l, 'h': h})

    def sample_height(self, x, y):
        xi = int(clamp(x / self.grid_res, 0, self.cells - 1))
        yi = int(clamp(y / self.grid_res, 0, self.cells - 1))
        return float(self.heightmap[yi, xi])  # scale

    def sample_moisture(self, x, y):
        xi = int(clamp(x / self.grid_res, 0, self.cells - 1))
        yi = int(clamp(y / self.grid_res, 0, self.cells - 1))
        return float(self.moisture[yi, xi])


class Robot:
    def __init__(self, env):
        self.env = env
        self.x = 4.0
        self.y = 4.0
        self.z = env.sample_height(self.x, self.y) + ROBOT_HEIGHT/2
        self.yaw = 0.0
        self.speed = 0.0
        self.ang_vel = 0.0
        self.wheel_encoder = 0.0
        self.last_update = time.time()

    def set_controls(self, forward, strafe, ang):
        self.speed = forward * MAX_SPEED
        self.ang_vel = ang * MAX_ANG_VEL

    def update(self, dt):
        self.yaw += self.ang_vel * dt
        dx = math.cos(self.yaw) * self.speed * dt
        dy = math.sin(self.yaw) * self.speed * dt
        self.x += dx
        self.y += dy
        self.x = clamp(self.x, 0.5, self.env.size - 0.5)
        self.y = clamp(self.y, 0.5, self.env.size - 0.5)
        target_z = self.env.sample_height(self.x, self.y) + ROBOT_HEIGHT/2
        self.z = 0.9 * self.z + 0.1 * target_z
        self.wheel_encoder += math.hypot(dx, dy)

    def get_pose(self):
        return (self.x, self.y, self.z, self.yaw)

    def read_gps(self):
        return self.x + np.random.normal(0, 0.02), self.y + np.random.normal(0, 0.02), self.z + np.random.normal(0, 0.01)

    def read_imu(self):
        return self.yaw + np.random.normal(0, math.radians(0.5))

    def read_wheel_odometry(self):
        return self.wheel_encoder + np.random.normal(0, 0.01)

    def read_lidar(self, env, num_samples=360 // LIDAR_RES_DEG, max_range=LIDAR_RANGE):
        samples = []
        angles = np.linspace(0, 2 * math.pi, num_samples, endpoint=False)
        for a in angles:
            ray_yaw = self.yaw + a
            r = 0.0
            hit = False
            while r < max_range:
                r += 0.2
                tx = self.x + math.cos(ray_yaw) * r
                ty = self.y + math.sin(ray_yaw) * r
                if not (0 <= tx <= env.size and 0 <= ty <= env.size):
                    hit = True
                    break
                for o in env.obstacles:
                    if (o['x'] <= tx <= o['x'] + o['w']) and (o['y'] <= ty <= o['y'] + o['l']):
                        hit = True
                        break
                if hit:
                    break
            samples.append(r if hit else max_range)
        return np.array(samples)


class Renderer:
    def __init__(self, screen, cam, env, robot):
        self.screen = screen
        self.cam = cam
        self.env = env
        self.robot = robot

    def draw(self):
        self.screen.fill(BG_COLOR)
        step = 2.0
        for gx in np.arange(0, self.env.size + 0.001, step):
            pts = []
            for gy in np.arange(0, self.env.size + 0.001, GRID_RES):
                h = self.env.sample_height(gx, gy)
                p = (gx, gy, h)
                pr = self.cam.project_point(p)
                if pr:
                    pts.append((pr[0], pr[1]))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, GRID_COLOR, False, pts, 1)
        for gy in np.arange(0, self.env.size + 0.001, step):
            pts = []
            for gx in np.arange(0, self.env.size + 0.001, GRID_RES):
                h = self.env.sample_height(gx, gy)
                p = (gx, gy, h)
                pr = self.cam.project_point(p)
                if pr:
                    pts.append((pr[0], pr[1]))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, GRID_COLOR, False, pts, 1)

        for o in self.env.obstacles:
            corners = [
                (o['x'], o['y'], self.env.sample_height(o['x'], o['y'])),
                (o['x'] + o['w'], o['y'], self.env.sample_height(o['x']+o['w'], o['y'])),
                (o['x'] + o['w'], o['y'] + o['l'], self.env.sample_height(o['x']+o['w'], o['y']+o['l'])),
                (o['x'], o['y'] + o['l'], self.env.sample_height(o['x'], o['y']+o['l']))
            ]
            proj = [self.cam.project_point((cx, cy, cz)) for (cx, cy, cz) in corners]
            proj = [p for p in proj if p]
            if len(proj) >= 3:
                poly = [(p[0], p[1]) for p in proj]
                pygame.draw.polygon(self.screen, OBST_COLOR, poly)

        rx, ry, rz, yaw = self.robot.get_pose()
        l = ROBOT_LENGTH; w = ROBOT_WIDTH; h = ROBOT_HEIGHT
        corners = [
            ( l/2,  w/2, h/2),
            ( l/2, -w/2, h/2),
            (-l/2, -w/2, h/2),
            (-l/2,  w/2, h/2),
            ( l/2,  w/2, -h/2),
            ( l/2, -w/2, -h/2),
            (-l/2, -w/2, -h/2),
            (-l/2,  w/2, -h/2),
        ]
        world_corners = []
        for cx, cy, cz in corners:
            wx, wy, wz = rot_z((cx, cy, cz), yaw)
            world_corners.append((rx + wx, ry + wy, rz + wz))
        proj = [self.cam.project_point(p) for p in world_corners]
        if all(p for p in proj):
            top = [ (p[0], p[1]) for p in proj[:4] ]
            bot = [ (p[0], p[1]) for p in proj[4:] ]
            pygame.draw.polygon(self.screen, ROBOT_COLOR, top)
            pygame.draw.polygon(self.screen, ROBOT_COLOR, bot)
            for i in range(4):
                pygame.draw.line(self.screen, ROBOT_COLOR, top[i], bot[i], 2)

        samples = self.robot.read_lidar(self.env, num_samples=360 // 15)  # coarse for render
        angs = np.linspace(0, 2*math.pi, len(samples), endpoint=False)
        for r, a in zip(samples, angs):
            ray_yaw = yaw + a
            tx = rx + math.cos(ray_yaw) * r
            ty = ry + math.sin(ray_yaw) * r
            tz = self.env.sample_height(tx if 0<=tx<=self.env.size else rx, ty if 0<=ty<=self.env.size else ry)
            pr1 = self.cam.project_point((rx, ry, rz))
            pr2 = self.cam.project_point((tx, ty, tz))
            if pr1 and pr2:
                pygame.draw.line(self.screen, (120,220,120), (pr1[0], pr1[1]), (pr2[0], pr2[1]), 1)

        self._draw_hud()

    def _draw_hud(self):
        font = pygame.font.SysFont(None, 20)
        x,y,z,yaw = self.robot.get_pose()
        gps = self.robot.read_gps()
        imu = self.robot.read_imu()
        odo = self.robot.read_wheel_odometry()
        moist = self.env.sample_moisture(x, y)
        lines = [
            f"Положение: x={x:.2f} m, y={y:.2f} m, z={z:.2f} m, yaw={math.degrees(yaw):.1f}°",
            f"GPS: x={gps[0]:.2f}, y={gps[1]:.2f}, z={gps[2]:.2f}  IMU yaw={math.degrees(imu):.1f}°",
            f"Скорость={self.robot.speed:.2f} m/s  Одометрия={odo:.2f} m",
            f"Влажность почвы={moist:.3f}  Расстояние от лидара={LIDAR_RANGE} m",
            "Управление: WS - двигаться, Q/E поворот, T автотест, L логгировать, +/- speed, Z/X/стрелки - управление камерой"
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, HUD_COLOR)
            self.screen.blit(surf, (10, 10 + i*20))


class TestPath:
    def __init__(self, robot):
        self.robot = robot
        self.waypoints = deque([
            (6,6), (20,6), (20,20), (6,20), (40,10), (50,40), (10,50), (4,40)
        ])
        self.active = False
        self.current_target = None

    def toggle(self):
        self.active = not self.active
        if self.active and not self.current_target:
            self._next_target()

    def _next_target(self):
        if self.waypoints:
            self.current_target = self.waypoints[0]
        else:
            self.current_target = None

    def step(self, dt):
        if not self.active or not self.current_target:
            return (0.0, 0.0)
        tx, ty = self.current_target
        dx = tx - self.robot.x
        dy = ty - self.robot.y
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        yaw_err = (desired_yaw - self.robot.yaw + math.pi) % (2*math.pi) - math.pi
        ang = clamp(yaw_err / math.radians(30), -1.0, 1.0)
        forward = clamp(dist / 2.5, -1.0, 1.0)
        # arrival
        if dist < 0.6:
            self.waypoints.popleft()
            self._next_target()
        return forward, ang


class Logger:
    def __init__(self):
        self.enabled = False
        self.rows = []
        self.start_time = time.time()

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            print("Началось логгирование")
            self.start_time = time.time()
        else:
            print("Логгирование завершено")

    def log(self, robot, env):
        if not self.enabled:
            return
        t = time.time() - self.start_time
        x,y,z,yaw = robot.get_pose()
        gps = robot.read_gps()
        imu = robot.read_imu()
        odo = robot.read_wheel_odometry()
        lidar = robot.read_lidar(env, num_samples=360//LIDAR_RES_DEG)
        min_lidar = float(lidar.min())
        moist = env.sample_moisture(x, y)
        row = [t, x, y, z, yaw, gps[0], gps[1], gps[2], imu, odo, min_lidar, moist]
        self.rows.append(row)

    def save(self, filename="agrobot_log.csv"):
        if not self.rows:
            print("Нечего сохранять")
            return
        header = ["t","x","y","z","yaw","gps_x","gps_y","gps_z","imu_yaw","odo","min_lidar","moisture"]
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)
        print(f"Сохранено {len(self.rows)} строк в {filename}")


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D окружение робота")
    clock = pygame.time.Clock()

    cam = Camera(WIDTH, HEIGHT)
    env = Environment()
    robot = Robot(env)
    renderer = Renderer(screen, cam, env, robot)
    testpath = TestPath(robot)
    logger = Logger()

    running = True
    auto_forward = 0.0
    auto_ang = 0.0
    manual_control = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t:
                    testpath.toggle()
                elif event.key == pygame.K_l:
                    logger.toggle()
                elif event.key == pygame.K_SPACE:
                    robot.set_controls(0.0,0.0,0.0)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    global MAX_SPEED
                    MAX_SPEED = min(6.0, MAX_SPEED + 0.2)
                    print("Максимальная скорость=", MAX_SPEED)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                    MAX_SPEED = max(0.2, MAX_SPEED - 0.2)
                    print("Максимальная скорость=", MAX_SPEED)
                elif event.key == pygame.K_s:
                    logger.save()

        keys = pygame.key.get_pressed()
        if not testpath.active:
            forward = 0.0
            ang = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                forward += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                forward -= 1.0
            if keys[pygame.K_q]:
                ang -= 1.0
            if keys[pygame.K_e]:
                ang += 1.0
            if keys[pygame.K_LEFT]:
                ang -= 0.8
            if keys[pygame.K_RIGHT]:
                ang += 0.8
            robot.set_controls(forward, 0.0, ang)
        else:
            f,a = testpath.step(dt)
            robot.set_controls(f, 0.0, a)
        cam_speed = 5.0 * dt
        if keys[pygame.K_UP]:
            cam.pos[2] += cam_speed * 50
        if keys[pygame.K_DOWN]:
            cam.pos[2] -= cam_speed * 50
        if keys[pygame.K_RIGHT]:
            cam.pos[1] += cam_speed * 50
        if keys[pygame.K_LEFT]:
            cam.pos[0] -= cam_speed * 50
        if keys[pygame.K_a]:
            cam.yaw -= 0.8 * dt
        if keys[pygame.K_d]:
            cam.yaw += 0.8 * dt
        if keys[pygame.K_z]:
            cam.pitch -= 0.8 * dt
        if keys[pygame.K_x]:
            cam.pitch += 0.8 * dt

        cam.pitch = clamp(cam.pitch, math.radians(-80), math.radians(80))



        robot.update(dt)
        logger.log(robot, env)
        renderer.draw()
        pygame.display.flip()

    if logger.enabled:
        logger.save()
    pygame.quit()


if __name__ == '__main__':
    main()
