import pygame
import random
import numpy as np
import gym
from gym import spaces
import time
import sys

# Inicializar pygame
pygame.init()

# Definir colores
BLANCO = (255, 255, 255)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)

# Definir dimensiones
ANCHO_VENTANA = 640
ALTO_VENTANA = 480
TAMANO_CELDA = 20

# Crear ventana y reloj
ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
pygame.display.set_caption("Snake")
reloj = pygame.time.Clock()

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()

        # Definir el espacio de acción (4 acciones: arriba, abajo, izquierda, derecha)
        self.action_space = spaces.Discrete(4)
        
        # Definir el espacio de observación (por ahora, lo mantendremos simple)
        # Por ejemplo, podríamos tener la dirección actual de la serpiente y su posición relativa al alimento
        # Pero para simplificar, lo dejaremos como un espacio discreto basado en el tamaño del tablero
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Inicializar el juego
        self.serpiente = [(5, 5), (4, 5), (3, 5)]
        self.direccion = (1, 0)  # (x, y) -> derecha
        self.alimento = self.generar_alimento()

    def generar_alimento(self):
        x = random.randint(2, (ANCHO_VENTANA // TAMANO_CELDA) - 2)
        y = random.randint(2, (ALTO_VENTANA // TAMANO_CELDA) - 2)
        return (x, y)
    
    def get_state(self):
        # Dirección
        dir_vector = [0, 0, 0, 0]
        if self.direccion == (0, -1):  # Arriba
            dir_vector[0] = 1
        elif self.direccion == (0, 1):  # Abajo
            dir_vector[1] = 1
        elif self.direccion == (-1, 0):  # Izquierda
            dir_vector[2] = 1
        elif self.direccion == (1, 0):  # Derecha
            dir_vector[3] = 1

        # Posición relativa del alimento
        food_dx = self.alimento[0] - self.serpiente[0][0]
        food_dy = self.alimento[1] - self.serpiente[0][1]

        return tuple(dir_vector + [food_dx, food_dy])

    def reset(self):
        self.serpiente = [(5, 5), (4, 5), (3, 5)]
        self.direccion = (1, 0)
        self.alimento = self.generar_alimento()
        return self.get_state()  # Estado inicial: posición de la cabeza

    def step(self, action):
        # Traducir acción a una dirección
        if action == 0 and self.direccion != (0, 1):   # Arriba
            self.direccion = (0, -1)
        elif action == 1 and self.direccion != (0, -1): # Abajo
            self.direccion = (0, 1)
        elif action == 2 and self.direccion != (1, 0): # Izquierda
            self.direccion = (-1, 0)
        elif action == 3 and self.direccion != (-1, 0): # Derecha
            self.direccion = (1, 0)

        cabeza = self.serpiente[0]
        nuevo_x = cabeza[0] + self.direccion[0]
        nuevo_y = cabeza[1] + self.direccion[1]

        # Verificar colisión con alimento
        if cabeza == self.alimento:
            self.serpiente.insert(0, self.alimento)
            self.alimento = self.generar_alimento()
            reward = 5 # Recompensa positiva por obtener alimento
        else:
            reward = -0.1

        done = False
        if (nuevo_x < 0 or nuevo_x >= ANCHO_VENTANA // TAMANO_CELDA or
            nuevo_y < 0 or nuevo_y >= ALTO_VENTANA // TAMANO_CELDA or
            (nuevo_x, nuevo_y) in self.serpiente):
            done = True
            reward = -10

        self.serpiente.insert(0, (nuevo_x, nuevo_y))
        self.serpiente.pop()

        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_UP:
                    self.direccion = (0, -1)
                elif evento.key == pygame.K_DOWN:
                    self.direccion = (0, 1)
                elif evento.key == pygame.K_LEFT:
                    self.direccion = (-1, 0)
                elif evento.key == pygame.K_RIGHT:
                    self.direccion = (1, 0)

        ventana.fill(BLANCO)
        for segmento in self.serpiente:
            pygame.draw.rect(ventana, VERDE, (segmento[0]*TAMANO_CELDA, segmento[1]*TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))
        pygame.draw.rect(ventana, ROJO, (self.alimento[0]*TAMANO_CELDA, self.alimento[1]*TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))
    
        fuente = pygame.font.SysFont(None, 35)
        puntaje = fuente.render(f"Puntuación: {len(self.serpiente) - 3}", True, (0, 0, 0))
        ventana.blit(puntaje, (10, 10))

        pygame.display.flip()
        reloj.tick(10)

    def close(self):
        pass

def state_to_index(state):
    dir_vector, food_dx, food_dy = state[:4], state[4], state[5]
    
    # Convertir dir_vector a un índice único
    dir_idx = np.argmax(dir_vector)
    
    # Discretizar food_dx y food_dy
    # (Esto es solo un ejemplo, puedes ajustar los límites según lo que observes en el juego)
    if food_dx < -10:
        dx_idx = 0
    elif food_dx < 0:
        dx_idx = 1
    elif food_dx == 0:
        dx_idx = 2
    elif food_dx <= 10:
        dx_idx = 3
    else:
        dx_idx = 4

    if food_dy < -10:
        dy_idx = 0
    elif food_dy < 0:
        dy_idx = 1
    elif food_dy == 0:
        dy_idx = 2
    elif food_dy <= 10:
        dy_idx = 3
    else:
        dy_idx = 4

    # Convertir todo a un índice único
    index = dir_idx * 25 + dx_idx * 5 + dy_idx
    return index

# Parámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 0.01  # Probabilidad de elegir una acción al azar (exploración)

# Inicializar tabla Q
q_table = np.zeros([125, 4])

env = SnakeEnv()

for i in range(10000):
    state = env.reset()
    state_index = state_to_index(state)
    done = False
    
    while not done:
        # Elegir acción basada en la política epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Acción al azar
        else:
            action = np.argmax(q_table[state_index, :])  # Mejor acción conocida

        # Realizar acción y observar resultado
        next_state, reward, done, _ = env.step(action)
        next_state_index = state_to_index(next_state)
        
        # Actualizar tabla Q
        best_next_action = np.argmax(q_table[next_state_index, :])
        q_table[state_index, action] = q_table[state_index, action] + alpha * (reward + gamma * q_table[next_state_index, best_next_action] - q_table[state_index, action])
        
        state = next_state
        state_index = next_state_index

    # Opcional: mostrar progreso
    if i % 100 == 0:
        print(f"Episodio {i}")

epsilon = 0  # Desactivar la elección de acciones aleatorias

num_episodios_play = 10

for i in range(num_episodios_play):
    state = env.reset()
    state_index = state_to_index(state)
    done = False

    while not done:
        # Elegir la mejor acción conocida usando la tabla Q
        action = np.argmax(q_table[state_index, :])

        # Realizar acción y observar resultado
        next_state, _, done, _ = env.step(action)
        next_state_index = state_to_index(next_state)

        # Visualizar el juego
        env.render()
        
        state = next_state
        state_index = next_state_index

    # Espera un poco antes del siguiente episodio para que puedas ver claramente cada juego
    time.sleep(2)

env.close()