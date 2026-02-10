    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import random
    from collections import deque
    import os
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем: {device}")

    class FastPatrolNet(nn.Module):
        def __init__(self):
            super(FastPatrolNet, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 10 * 10, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 4)
            )
        
        def forward(self, x):
            return self.net(x)

    class FixedPatrolEnv:
        def __init__(self):
            self.size = 10
            self.max_steps = 100
            self.walls = {(2,2),(2,3),(3,2),(3,3),
                        (6,6),(6,7),(7,6),(7,7),
                        (2,7),(3,7),(7,2),(7,3)}
            self.risk_zones = [(1,1),(1,8),(8,1),(8,8),(5,5)]
            self.reset()
        
        def reset(self):
            self.pos = [5, 5]
            self.targets = []
            self.steps = 0
            self.caught = 0
            self.missed = 0
            self.visit_count = np.zeros((10, 10), dtype=np.float32)
            self.spawn(force=True)
            return self.get_state()
        
        def get_state(self):
            state = np.zeros((4, 10, 10), dtype=np.float32)
            state[0, self.pos[0], self.pos[1]] = 1.0
            for (y, x) in self.walls:
                state[1, y, x] = 1.0
            for t in self.targets:
                state[1, t[0], t[1]] = -1.0
            max_visits = np.max(self.visit_count) + 1
            state[2] = self.visit_count / max_visits
            state[3] = 1.0 - state[2]
            return state
        
        def spawn(self, force=False):
            if len(self.targets) >= 1:
                return
            if not force and random.random() > 0.1:
                return
            point = random.choice(self.risk_zones)
            y = max(0, min(9, point[0] + random.randint(-1, 1)))
            x = max(0, min(9, point[1] + random.randint(-1, 1)))
            if (y, x) not in self.walls and [y, x] != self.pos:
                self.targets.append([y, x, 20])
        
        def step(self, action):
            reward = 0
            old_pos = self.pos.copy()
            y, x = self.pos
            
            if action == 0: y -= 1
            elif action == 1: y += 1
            elif action == 2: x -= 1
            elif action == 3: x += 1
            
            hit_wall = False
            if (y, x) in self.walls or not (0 <= y < 10 and 0 <= x < 10):
                hit_wall = True
                y, x = old_pos
                reward -= 0.5
            
            self.pos = [y, x]
            self.steps += 1
            self.visit_count[y, x] += 1
            
            # Награды
            reward -= 0.02
            
            visits = self.visit_count[y, x]
            if visits == 1:
                reward += 2.0
            elif visits <= 2:
                reward += 0.5
            else:
                reward -= 0.5 * visits
            
            # Соседние неисследованные
            unexplored = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 10 and 0 <= nx < 10 and self.visit_count[ny, nx] == 0:
                        unexplored += 1
            reward += 0.3 * unexplored
            
            # Цели
            caught = False
            new_targets = []
            for t in self.targets:
                if t[0] == y and t[1] == x:
                    reward += 1.0
                    self.caught += 1
                    caught = True
                else:
                    t[2] -= 1
                    if t[2] > 0:
                        new_targets.append(t)
                    else:
                        reward -= 1.0
                        self.missed += 1
            self.targets = new_targets
            
            if old_pos == self.pos and not hit_wall:
                reward -= 0.3
            
            self.spawn()
            done = (self.missed >= 2) or (self.steps >= self.max_steps)
            
            if done:
                coverage = np.sum(self.visit_count > 0) / 100.0
                reward += coverage * 5.0
            
            return self.get_state(), reward, done, caught

    class FastPatrolAgent:
        def __init__(self, load_model=True):
            self.net = FastPatrolNet().to(device)
            self.target_net = FastPatrolNet().to(device)
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()
            
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
            self.memory = deque(maxlen=10000)
            
            self.batch_size = 32
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995
            self.update_every = 100
            
            self.steps_done = 0
            self.model_file = "fast_patrol_v2.pth"
            
            if load_model and os.path.exists(self.model_file):
                self.net.load_state_dict(torch.load(self.model_file))
                self.target_net.load_state_dict(self.net.state_dict())
                print("🧠 Загружен опыт")
        
        def select_action(self, state, training=True):
            if training and random.random() < self.epsilon:
                return random.randint(0, 3)
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q = self.net(state_t)
                return q.argmax().item()
        
        def store(self, s, a, r, s2, d):
            self.memory.append((s, a, r, s2, d))
        
        def learn(self):
            if len(self.memory) < self.batch_size:
                return 0
            batch = random.sample(self.memory, self.batch_size)
            s, a, r, s2, d = zip(*batch)
            s = torch.FloatTensor(np.array(s)).to(device)
            a = torch.LongTensor(a).to(device)
            r = torch.FloatTensor(r).to(device)
            s2 = torch.FloatTensor(np.array(s2)).to(device)
            d = torch.BoolTensor(d).to(device)
            
            q = self.net(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q2 = self.target_net(s2).max(1)[0]
                target = r + self.gamma * q2 * ~d
            
            loss = nn.functional.smooth_l1_loss(q, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            return loss.item()
        
        def update_target(self):
            self.target_net.load_state_dict(self.net.state_dict())
        
        def decay(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def save(self):
            torch.save(self.net.state_dict(), self.model_file)
            print(f"💾 Сохранено, epsilon={self.epsilon:.3f}")

    def train_fast(episodes=2000):  # УВЕЛИЧИЛ до 2000 (можно 3000, 5000)
        env = FixedPatrolEnv()
        agent = FastPatrolAgent(load_model=True)
        
        rewards = []
        coverages = []
        
        print(f"Обучение: {episodes} эпизодов...")
        start = time.time()
        
        for ep in range(episodes):
            state = env.reset()
            total = 0
            
            for step in range(env.max_steps):
                action = agent.select_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                
                agent.store(state, action, reward, next_state, done)
                loss = agent.learn()
                
                total += reward
                state = next_state
                agent.steps_done += 1
                
                if agent.steps_done % agent.update_every == 0:
                    agent.update_target()
                
                if done:
                    break
            
            agent.decay()
            rewards.append(total)
            
            coverage = np.sum(env.visit_count > 0) / 100.0
            coverages.append(coverage)
            
            if ep % 100 == 0:
                elapsed = time.time() - start
                avg_r = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                avg_c = np.mean(coverages[-100:]) if len(coverages) >= 100 else np.mean(coverages)
                print(f"Эп {ep}: Награда={avg_r:.2f}, "
                    f"Покрытие={avg_c*100:.1f}%, "
                    f"Eps={agent.epsilon:.3f}, "
                    f"Время={elapsed:.1f}с")
        
        agent.save()
        
        # Графики
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(rewards, alpha=0.3, color='blue')
        if len(rewards) >= 30:
            ma = np.convolve(rewards, np.ones(30)/30, mode='valid')
            axes[0].plot(ma, color='red', linewidth=2)
        axes[0].set_xlabel('Эпизод')
        axes[0].set_ylabel('Награда')
        axes[0].set_title('Обучение')
        axes[0].grid(True)
        
        axes[1].plot(coverages, alpha=0.3, color='green')
        if len(coverages) >= 30:
            ma = np.convolve(coverages, np.ones(30)/30, mode='valid')
            axes[1].plot(ma, color='darkgreen', linewidth=2)
        axes[1].set_xlabel('Эпизод')
        axes[1].set_ylabel('Покрытие')
        axes[1].set_title(f'Покрытие (финал: {coverages[-1]*100:.1f}%)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return agent, env

    def demo_fixed(agent, episodes=3):
        """
        ИСПРАВЛЕННАЯ демонстрация с нормальной картой посещений.
        Это 3 отдельных тестовых запуска обученного агента (не обучение!).
        """
        env = FixedPatrolEnv()
        plt.ion()
        
        # Создаем фигуру с двумя подграфиками: карта и тепловая карта
        fig = plt.figure(figsize=(14, 6))
        ax_map = plt.subplot(1, 2, 1)
        ax_heatmap = plt.subplot(1, 2, 2)
        
        # Создаем colorbar ОДИН РАЗ
        im = ax_heatmap.imshow(np.zeros((10, 10)), cmap='YlOrRd', vmin=0, vmax=5)
        cbar = plt.colorbar(im, ax=ax_heatmap)
        cbar.set_label('Количество посещений')
        
        for demo_ep in range(episodes):
            state = env.reset()
            total = 0
            path = [tuple(env.pos)]
            
            for step in range(100):
                action = agent.select_action(state, training=False)
                state, reward, done, _ = env.step(action)
                total += reward
                path.append(tuple(env.pos))
                
                # Рисуем каждые 5 шагов
                if step % 5 == 0 or done:
                    # Левая панель - карта патруля
                    ax_map.clear()
                    
                    # Стены черные
                    for (y, x) in env.walls:
                        ax_map.add_patch(Rectangle((x, 9-y), 1, 1, facecolor='black'))
                    
                    # Зоны риска красные полупрозрачные
                    for (y, x) in env.risk_zones:
                        ax_map.add_patch(Rectangle((x-0.5, 9-(y-0.5)), 2, 2, 
                                                facecolor='red', alpha=0.1))
                    
                    # Путь синий
                    if len(path) > 1:
                        xs = [p[1]+0.5 for p in path]
                        ys = [9-p[0]+0.5 for p in path]
                        ax_map.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
                    
                    # Цель красный круг
                    if env.targets:
                        t = env.targets[0]
                        circle = Circle((t[1]+0.5, 9-t[0]+0.5), 0.4, color='red')
                        ax_map.add_patch(circle)
                        ax_map.text(t[1]+0.5, 9-t[0]+0.5, str(t[2]), 
                                ha='center', va='center', color='white', fontsize=9)
                    
                    # Агент синий квадрат
                    ax_map.plot(env.pos[1]+0.5, 9-env.pos[0]+0.5, 
                            's', color='blue', markersize=20)
                    
                    ax_map.set_xlim(0, 10)
                    ax_map.set_ylim(0, 10)
                    ax_map.set_aspect('equal')
                    ax_map.set_title(f'ПАТРУЛЬ #{demo_ep+1} (тест) | Шаг {step} | Очки: {total:.1f}')
                    
                    # Правая панель - тепловая карта посещений (ОБНОВЛЯЕМ ДАННЫЕ)
                    im.set_array(env.visit_count)
                    ax_heatmap.set_title(f'Карта посещений\nУникальных клеток: {np.sum(env.visit_count > 0)}/100')
                    
                    plt.pause(0.3)
                
                if done:
                    break
            
            coverage = np.sum(env.visit_count > 0) / 100.0
            print(f"Патруль {demo_ep+1}: {total:.1f} очков, покрытие {coverage*100:.0f}%")
            print("  (это тест обученного агента, не обучение)")
            plt.pause(1)
        
        plt.ioff()
        plt.show()

    if __name__ == "__main__":
        # Чтобы увеличить эпизоды - просто измени число здесь:
        # episodes=2000 (сейчас), можно episodes=3000 или 5000
        agent, env = train_fast(episodes=2000)
        
        # Это 3 тестовых запуска для проверки:
        print("\n=== Демонстрация обученного агента (3 тестовых патруля) ===")
        demo_fixed(agent, episodes=3)