import pygame
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os # Model kaydetme/yükleme için
import argparse # Komut satırı argümanları için

# Stable Baselines3 importları
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback # İlerleme takibi için (opsiyonel)

# Pygame'i başlat (render için) - Opsiyonel: Sadece render modunda başlatılabilir
# pygame.init() # init'i __init__ içine taşıyalım

class SnakeEnv(gym.Env):
    # Varsayılan görsel hızı 1x (örn. 15 FPS) yapalım
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 15}

    def __init__(self, grid_size=40, render_mode=None):
        # grid_size'ı parametre olarak alalım, __main__ içinde ayarlansın
        super(SnakeEnv, self).__init__()

        self.grid_size = grid_size # Oyun alanını kare grid olarak düşünelim
        self.block_size = 20 # Her grid hücresinin piksel boyutu
        self.width = self.grid_size * self.block_size
        self.height = self.grid_size * self.block_size

        # Aksiyon alanı: 4 yön (0: Yukarı, 1: Aşağı, 2: Sol, 3: Sağ)
        self.action_space = spaces.Discrete(4)

        # Gözlem alanı: Yılan başı (x, y), Yem (x, y), Yılan yönü (dx, dy), Belki tehlike bilgileri
        # Şimdilik basit: [yılan_x, yılan_y, yem_x, yem_y] - Normalized [-1, 1]
        # low = np.array([-1] * 4, dtype=np.float32)
        # high = np.array([1] * 4, dtype=np.float32)
        # self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # Daha iyi bir gözlem: Yılanın etrafındaki 3x3'lük alan + yön + yem yönü
        # Veya tüm grid'i temsil eden bir matris (CNN için uygun olabilir)
        # Şimdilik daha bilgilendirici bir Box kullanalım:
        # [yılan_baş_x, yılan_baş_y, yem_delta_x, yem_delta_y, sol_tehlike, ön_tehlike, sağ_tehlike]
        # Tehlike: 0 (yok), 1 (var - duvar veya kendi)
        low = np.array([0] * (self.grid_size * self.grid_size) + [0] * (self.grid_size * self.grid_size), dtype=np.float32) # yılan + yem
        high = np.array([1] * (self.grid_size * self.grid_size) + [1] * (self.grid_size * self.grid_size), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 2), dtype=np.float32) # yılan + yem katmanı
        # Daha basit bir state ile başlayalım:
        # [head_x, head_y, food_x, food_y, body_part1_x, body_part1_y, ...]
        # Boyut değiştiği için bu Box için uygun değil.
        # Sabit boyutlu state kullanalım:
        # [yem_x_relative, yem_y_relative, sol_tehlike, ön_tehlike, sağ_tehlike, yön_x, yön_y] (7 değer)
        low = np.array([-self.grid_size, -self.grid_size, 0, 0, 0, -1, -1], dtype=np.float32)
        high = np.array([self.grid_size, self.grid_size, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)


        # Renkler
        self.WHITE = (255, 255, 255)
        self.RED = (213, 50, 80)
        self.GREEN = (0, 255, 0)
        self.BLUE = (50, 153, 213)
        self.BLACK = (0, 0, 0)
        self.HEAD_COLOR = (0, 150, 0) # Baş farklı renk olsun

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.paused = False # Duraklatma durumu için flag
        self.episode_step = 0 # Mevcut bölümdeki adım sayacı
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption('Yapay Zeka Yılan Oyunu')
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()


    def _get_obs(self):
        # Gözlem (state) hesaplama
        head_x, head_y = self.snake_body[0]
        food_x, food_y = self.food_pos

        # Yeme göreceli konum
        food_rel_x = food_x - head_x
        food_rel_y = food_y - head_y

        # Yön vektörü (dx, dy)
        dx, dy = 0, 0
        if self.direction == 0: # UP
            dy = -1
        elif self.direction == 1: # DOWN
            dy = 1
        elif self.direction == 2: # LEFT
            dx = -1
        elif self.direction == 3: # RIGHT
            dx = 1

        # Tehlike kontrolü (yılanın mevcut yönüne göre sol, ön, sağ)
        # Döndürme matrisi veya if/else ile yönleri kontrol et
        # Mevcut yön (dx, dy)
        # Sol yön: (-dy, dx)
        # Sağ yön: (dy, -dx)

        current_dir_vector = np.array([dx, dy])
        left_dir_vector = np.array([-dy, dx])
        right_dir_vector = np.array([dy, -dx])

        # Tehlike noktaları (yılanın bir sonraki potansiyel konumları)
        point_l = (head_x + left_dir_vector[0] * self.block_size, head_y + left_dir_vector[1] * self.block_size)
        point_r = (head_x + right_dir_vector[0] * self.block_size, head_y + right_dir_vector[1] * self.block_size)
        point_f = (head_x + current_dir_vector[0] * self.block_size, head_y + current_dir_vector[1] * self.block_size)

        danger_left = self._is_collision(point_l)
        danger_front = self._is_collision(point_f)
        danger_right = self._is_collision(point_r)

        obs = np.array([
            food_rel_x / self.block_size, # Normalize edilebilir
            food_rel_y / self.block_size, # Normalize edilebilir
            danger_left,
            danger_front,
            danger_right,
            dx,
            dy
        ], dtype=np.float32)

        # print(f"Obs: {obs}") # Debug
        return obs


    def _get_info(self):
        # Ek bilgi (opsiyonel)
        return {"score": self.score, "snake_length": len(self.snake_body)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Rastgelelik için seed ayarı

        # Başlangıç durumu
        center_x = (self.grid_size // 2) * self.block_size
        center_y = (self.grid_size // 2) * self.block_size
        self.snake_body = [[center_x, center_y],
                           [center_x - self.block_size, center_y],
                           [center_x - 2 * self.block_size, center_y]]
        self.direction = 3 # Başlangıç yönü: RIGHT (0:U, 1:D, 2:L, 3:R)

        self.score = 0
        self.episode_step = 0 # Adım sayacını sıfırla
        self._place_food()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_food(self):
        # Yemi yılanın üzerine denk gelmeyecek şekilde yerleştir
        while True:
            self.food_pos = [random.randrange(0, self.grid_size) * self.block_size,
                             random.randrange(0, self.grid_size) * self.block_size]
            if self.food_pos not in self.snake_body:
                break

    def step(self, action):
        # Aksiyonu işle (0: Yukarı, 1: Aşağı, 2: Sol, 3: Sağ)
        # Yılanın ters yöne gitmesini engelle
        if action == 0 and self.direction == 1: # Yukarı gitmeye çalışırken aşağı gidiyorsa
            action = self.direction
        elif action == 1 and self.direction == 0: # Aşağı gitmeye çalışırken yukarı gidiyorsa
            action = self.direction
        elif action == 2 and self.direction == 3: # Sol'a gitmeye çalışırken sağa gidiyorsa
            action = self.direction
        elif action == 3 and self.direction == 2: # Sağ'a gitmeye çalışırken sola gidiyorsa
            action = self.direction

        self.direction = action # Yeni yönü ata

        # Önceki kafa konumu ve yeme mesafe
        prev_head = self.snake_body[0]
        prev_dist_to_food = np.linalg.norm(np.array(prev_head) - np.array(self.food_pos))

        # Yılan başının yeni konumunu hesapla
        head_x, head_y = self.snake_body[0]
        if self.direction == 0: # UP
            head_y -= self.block_size
        elif self.direction == 1: # DOWN
            head_y += self.block_size
        elif self.direction == 2: # LEFT
            head_x -= self.block_size
        elif self.direction == 3: # RIGHT
            head_x += self.block_size

        new_head = [head_x, head_y]

        # Çarpışma kontrolü
        terminated = self._is_collision(new_head)
        truncated = False # Zaman sınırı vb. nedenlerle bitme durumu (şimdilik False)

        reward = 0
        ate_food = False # Yem yiyip yemediğini takip et

        if terminated:
            reward = -100 # Büyük ceza
            # print("Game Over!") # Debug
        else:
            # Yeni başı ekle
            self.snake_body.insert(0, new_head)

            # Yem yeme kontrolü
            if new_head == self.food_pos:
                self.score += 10
                reward = 50 # Büyük ödül
                ate_food = True
                self._place_food()
            else:
                # Yem yemezse kuyruğu sil
                self.snake_body.pop()
                # Yem yemedi, mesafe ve zaman ödülünü hesapla
                new_dist_to_food = np.linalg.norm(np.array(new_head) - np.array(self.food_pos))
                # Yeme yaklaşma ödülü (mesafe azaldıysa pozitif)
                reward += (prev_dist_to_food - new_dist_to_food) * 0.1 # Katsayı ayarlanabilir
                # Zaman cezası (her adımda küçük negatif ödül)
                reward -= 0.1 # Miktar ayarlanabilir

        # Adım sayacını artır
        self.episode_step += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame() # Render içinde pause kontrolü olacak
            # self.clock.tick(self.metadata['render_fps']) # Tick'i render sonuna taşıyalım

        # print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}") # Debug
        return observation, reward, terminated, truncated, info

    def _is_collision(self, point=None):
        if point is None:
            point = self.snake_body[0]
        px, py = point

        # Sınırlara çarpma
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True

        # Kendi kendine çarpma
        if point in self.snake_body[1:]:
            return True

        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(to_rgb_array=True)
        elif self.render_mode == "human":
             self._render_frame()
        # Diğer modlarda render yapma (None)

    def _render_frame(self, to_rgb_array=False):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption('Yapay Zeka Yılan Oyunu')
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Duraklatma kontrolü (sadece human modda)
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                     self.close()
                     # QUIT olayında da programdan çıkalım
                     print("Pencere kapatıldı, çıkılıyor...")
                     sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused # Pause durumunu değiştir
                        if self.paused:
                            print("Eğitim Duraklatıldı. P: Devam, Q: Çıkış")
                        else:
                            print("Eğitim Devam Ediyor...")
                    # Pause aktif değilken Q'ya basılırsa da çıkılsın mı? Şimdilik sadece pause içinde
                    # elif event.key == pygame.K_q:
                    #      print("Q'ya basıldı, çıkılıyor...")
                    #      self.close()
                    #      sys.exit()

            # Eğer duraklatıldıysa, döngüde bekle
            while self.paused:
                # Pause mesajını ekrana yaz (Q seçeneği ile)
                pause_font = pygame.font.SysFont('arial', 50)
                pause_text = 'DURAKLATILDI (P: Devam, Q: Çıkış)'
                pause_surface = pause_font.render(pause_text, True, self.RED)
                pause_rect = pause_surface.get_rect(center=(self.width / 2, self.height / 2))
                temp_surface = self.screen.copy() # Anlık görüntüyü al
                temp_surface.blit(pause_surface, pause_rect) # Üzerine yaz
                self.screen.blit(temp_surface, (0,0)) # Ekrana bas
                pygame.display.flip()

                # Pause döngüsünde olayları kontrol et (P, Q veya QUIT)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Pencere kapatıldı, çıkılıyor...")
                        self.close()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False # Döngüden çık
                            print("Eğitim Devam Ediyor...")
                            break # İç olay döngüsünden çık
                        elif event.key == pygame.K_q:
                            print("Q'ya basıldı, çıkılıyor... Modelin son hali kaydedilmiş olmalı.")
                            self.close()
                            sys.exit() # Programı sonlandır
                self.clock.tick(15) # Düşük FPS'de bekle

        # --- Normal Render Kodu ---
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.BLACK)
        # Yemi çiz
        pygame.draw.rect(canvas, self.WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))

        # Yılanı çiz (başı farklı renk)
        head = True
        for pos in self.snake_body:
            color = self.HEAD_COLOR if head else self.GREEN
            pygame.draw.rect(canvas, color, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
            pygame.draw.rect(canvas, self.BLACK, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size), 1) # Kenarlık
            head = False

        # Skoru göster (Sol Üst)
        font = pygame.font.SysFont('arial', 20)
        score_surface = font.render('Skor: ' + str(self.score), True, self.WHITE)
        score_rect = score_surface.get_rect()
        score_rect.topleft = (10, 10)
        canvas.blit(score_surface, score_rect)

        # Eğitim bilgilerini göster (Sağ Üst)
        step_surface = font.render(f'Adım: {self.episode_step}', True, self.WHITE)
        step_rect = step_surface.get_rect()
        step_rect.topright = (self.width - 10, 10)
        canvas.blit(step_surface, step_rect)

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            #pygame.event.pump() # Olaylar yukarıda işlendi
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps']) # FPS ayarlaması burada
        elif to_rgb_array:
             return np.transpose(
                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
             ) # (width, height, rgb) -> (height, width, rgb) numpy array

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            # sys.exit() # Ortamı kapatınca programdan çıkmak yerine sadece pygame'i kapat


# Ortamı test etmek için basit bir örnek (opsiyonel)
# if __name__ == '__main__':
#     # Human render modu ile ortamı oluştur
#     env = SnakeEnv(grid_size=15, render_mode='human')
#     obs, info = env.reset()
#
#     # # Stable Baselines3 ile ortam kontrolü (opsiyonel)
#     # from stable_baselines3.common.env_checker import check_env
#     # try:
#     #     check_env(env, warn=True)
#     #     print("Environment check passed!")
#     # except Exception as e:
#     #     print(f"Environment check failed: {e}")
#
#
#     # Manuel kontrol veya rastgele ajan ile test
#     terminated = False
#     truncated = False
#     total_reward = 0
#     step_count = 0
#
#     # Rastgele aksiyonlarla test döngüsü
#     while not terminated and not truncated:
#         # Rastgele bir aksiyon seç (veya klavyeden al)
#         # action = env.action_space.sample()
#
#         # Klavye kontrolü için (eğer human modundaysa)
#         action = env.direction # Mevcut yönü koru
#         if env.render_mode == 'human':
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     terminated = True
#                 if event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_UP:
#                         action = 0
#                     elif event.key == pygame.K_DOWN:
#                         action = 1
#                     elif event.key == pygame.K_LEFT:
#                         action = 2
#                     elif event.key == pygame.K_RIGHT:
#                         action = 3
#                     elif event.key == pygame.K_q: # Çıkış
#                          terminated = True
#
#         if terminated: break
#
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
#         step_count += 1
#         # env.render() # step içinde zaten render çağrılıyor
#
#         if terminated or truncated:
#             print(f"Episode finished after {step_count} steps.")
#             print(f"Score: {info['score']}, Total Reward: {total_reward}")
#             # obs, info = env.reset() # İstersen yeniden başlat
#
#     env.close()

# --- Eğitim Fonksiyonu ---
def train(env, total_timesteps=50000, model_path="dqn_snake_model.zip"):
    """Yılan ajanını eğitir ve modeli kaydeder."""
    # Hiperparametreler (DQN için)
    learning_rate = 0.001
    buffer_size = 10000 # Daha büyük buffer daha stabil ama daha fazla RAM
    learning_starts = 1000
    batch_size = 64
    tau = 1.0
    gamma = 0.99
    train_freq = 4 # Her 4 adımda bir güncelleme
    gradient_steps = 1
    target_update_interval = 1000 # Hedef ağ güncelleme sıklığı
    exploration_fraction = 0.2 # Toplam adımların % kaçında keşif yapılacak
    exploration_initial_eps = 1.0 # Başlangıç epsilon
    exploration_final_eps = 0.05 # Bitiş epsilon

    # Modeli oluştur (veya varsa yükle ve devam et)
    model_file_path = os.path.abspath(model_path) # Tam yolu alalım
    print(f"Model dosyası kontrol ediliyor: {model_file_path}")
    if os.path.exists(model_file_path):
        print(f"Var olan model yükleniyor: {model_file_path}")
        # Ortamı sarmalamadan önce yüklemek genellikle daha iyidir (örneğin VecNormalize için)
        try:
            model = DQN.load(model_file_path, env=env)
            print("Model başarıyla yüklendi. Eğitime devam ediliyor...")
            # Replay buffer'ın yüklenmediğini belirtelim
            print("(Not: Replay buffer genellikle yüklenmez, ajan başlangıçta tekrar veri toplayacaktır.)")
        except Exception as e:
            print(f"Model yüklenirken HATA oluştu: {e}")
            print("Yeni model oluşturuluyor...")
            model = DQN('MlpPolicy', env,
                        learning_rate=learning_rate,
                        buffer_size=buffer_size,
                        learning_starts=learning_starts,
                        batch_size=batch_size,
                        tau=tau,
                        gamma=gamma,
                        train_freq=train_freq,
                        gradient_steps=gradient_steps,
                        target_update_interval=target_update_interval,
                        exploration_fraction=exploration_fraction,
                        exploration_initial_eps=exploration_initial_eps,
                        exploration_final_eps=exploration_final_eps,
                        verbose=1)
            reset_num_timesteps = True # Hata durumunda sıfırdan başla
        else:
            # Başarıyla yüklendiyse
             reset_num_timesteps = False
    else:
        print(f"Model dosyası bulunamadı. Yeni model oluşturuluyor...")
        model = DQN('MlpPolicy', env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    verbose=1)
        reset_num_timesteps = True

    print(f"{total_timesteps} adım boyunca eğitiliyor...")
    try:
        model.learn(total_timesteps=total_timesteps,
                    log_interval=10,
                    reset_num_timesteps=reset_num_timesteps)
    except KeyboardInterrupt:
        print("Eğitim kullanıcı tarafından kesildi (Ctrl+C).")
    finally:
        # Eğitim bittiğinde veya kesildiğinde modeli kaydet
        print(f"Model şuraya kaydediliyor: {model_file_path}")
        model.save(model_file_path)
        print("Model kaydedildi.")

# --- Yapay Zeka ile Oynama Fonksiyonu ---
def play_ai(env, model_path="dqn_snake_model.zip"):
    """Eğitilmiş modeli yükler ve oyunu oynatır."""
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        print("Lütfen önce modeli eğitin (örneğin: python snake_game.py --mode train)")
        return

    print(f"Model yükleniyor: {model_path}")
    model = DQN.load(model_path, env=env)
    print("Yapay zeka oynuyor... (Çıkmak için pencereyi kapatın)")

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    episodes = 0

    while True: # Pencere kapatılana kadar oyna
        action, _states = model.predict(obs, deterministic=True) # En iyi aksiyonu seç
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # env.render() # step içinde render çağrılıyor

        if terminated or truncated:
            episodes += 1
            print(f"Bölüm {episodes} Bitti. Skor: {info['score']}, Toplam Ödül: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            # Pygame penceresi kapatıldıysa döngüden çık
            if not pygame.display.get_init():
                 break

# --- Manuel Oynama Fonksiyonu ---
def play_manual(env):
    """Kullanıcının klavye ile oynamasını sağlar."""
    print("Manuel oyun modu. Yön tuşlarını kullanın. Çıkmak için Q'ya basın veya pencereyi kapatın.")
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        action = env.direction # Mevcut yönü koru varsayılan olarak
        user_action_taken = False

        if env.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    break
                if event.type == pygame.KEYDOWN:
                    user_action_taken = True
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3
                    elif event.key == pygame.K_q: # Çıkış
                         terminated = True
                         break
            if terminated: break # Olay döngüsünden çıkıldıysa ana döngüden de çık
        else:
            # Render modu human değilse manuel oynamak mantıklı değil
            print("Manuel oyun için render_mode='human' olmalı.")
            break

        # Sadece kullanıcı bir tuşa bastıysa veya başlangıç değilse step at
        # if user_action_taken or step_count > 0:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        # env.render() # step içinde render çağrılıyor

        if terminated or truncated:
            print(f"Oyun bitti. Adım: {step_count}, Skor: {info['score']}, Toplam Ödül: {total_reward:.2f}")

    env.close()


# --- Yardımcı Fonksiyonlar ---

def draw_menu():
    """Konsola menü seçeneklerini yazdırır ve kullanıcıdan girdi alır."""
    print("\n--- Yılan Oyunu Menüsü ---")
    print("1. Eğit (Train - Render Kapalı)")
    print("2. Eğitilmiş AI ile Oyna (Render Açık)")
    print("3. Manuel Oyna (Render Açık)")
    print("4. Çıkış")
    print("-------------------------")

    while True:
        choice = input("Seçiminizi yapın (1-4): ")
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("Geçersiz giriş. Lütfen 1 ile 4 arasında bir sayı girin.")

# --- Ana Çalıştırma Bloğu (Menü ile) ---
if __name__ == '__main__':
    GRID_SIZE = 40 # Grid boyutunu 2 katına çıkararak alanı 4 katına çıkaralım
    # Daha uzun eğitim için adım sayısını artıralım
    TIMESTEPS = 500000 # Örneğin 500,000 adım
    MODEL_PATH = "dqn_snake_model.zip"

    while True:
        choice = draw_menu()

        if choice == '1':
            print("Eğitim modu başlatılıyor (Render kapalı)...")
            # Eğitim için render'ı kapatarak hızlandır
            env = SnakeEnv(grid_size=GRID_SIZE, render_mode=None)
            train(env, total_timesteps=TIMESTEPS, model_path=MODEL_PATH)
            env.close() # Ortamı kapat
            print("Eğitim tamamlandı veya kesildi.")
        elif choice == '2':
            print("Eğitilmiş AI ile oynama modu...")
            if not os.path.exists(MODEL_PATH):
                print(f"Model dosyası bulunamadı: {MODEL_PATH}")
                print("Lütfen önce modeli eğitin (Seçenek 1).")
                continue # Menüye dön
            # AI ile oynarken render açık
            env = SnakeEnv(grid_size=GRID_SIZE, render_mode='human')
            play_ai(env, model_path=MODEL_PATH)
            env.close()
        elif choice == '3':
            print("Manuel oynama modu...")
            # Manuel oynarken render açık
            env = SnakeEnv(grid_size=GRID_SIZE, render_mode='human')
            play_manual(env)
            env.close()
        elif choice == '4':
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, tekrar deneyin.")

    print("Program sonlandı.") 