import argparse
import sys

# ==========================================
# 1. CABEÇALHO DO DOCKER COMPOSE (Fixos)
# ==========================================
YAML_HEADER = """
version: '3.8'

# Template base para todos os clientes (evita repetição)
x-client-template: &client-template
  image: fl-framework
  volumes:
    - ./data:/app/data
    - ./fl_logs:/app/fl_logs
    - ./client:/app/client
    - ./shared:/app/shared
    - ./config.py:/app/config.py
  # AQUI ESTA A CORRECAO: Lista simples, sem condicao de saude
  depends_on: 
    - server

services:
  # Controlador de Rede (Latência/Banda)
  docker-tc:
    image: lukaszlach/docker-tc
    container_name: docker-tc
    network_mode: host
    cap_add:
      - NET_ADMIN
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /var/docker-tc:/var/docker-tc
    restart: always

  # Servidor Central
  server:
    build: .
    image: fl-framework
    container_name: fl-rest_server_1
    command: >
      sh -c "tensorboard --logdir=/app/fl_logs/tensorboard --port=6006 --host=0.0.0.0 &
             python -m server.app"
    ports:
      - "5000:5000"
      - "6006:6006"
    volumes:
      - ./data:/app/data
      - ./fl_logs:/app/fl_logs
      - ./server:/app/server
      - ./shared:/app/shared
      - ./config.py:/app/config.py
    # Removemos o Healthcheck para evitar que o Docker mate o servidor por lentidao
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

# ==========================================
# 2. TEMPLATES DOS CLIENTES (Variáveis)
# ==========================================

# --- CLIENTE RÁPIDO (High Performance) ---
CLIENT_HIGH_PERF_TEMPLATE = """
  client_{client_id_str}:
    <<: *client-template
    container_name: fl-rest_client_{client_id_str}_1
    command: python -m client.app
    environment:
      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
      - TRAIN_FRACTION=1.0 
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.limit=1gbit"
      - "com.docker-tc.delay=20ms"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

# --- CLIENTE LENTO (Low Performance) ---
CLIENT_LOW_PERF_TEMPLATE = """
  client_{client_id_str}:
    <<: *client-template
    container_name: fl-rest_client_{client_id_str}_1
    command: python -m client.app
    environment:
      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
      - TRAIN_FRACTION=0.5 
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.limit=5mbps"
      - "com.docker-tc.delay=200ms"
      - "com.docker-tc.loss=1%"
    # Removemos limite de CPU para evitar travamentos no boot
"""

def generate_compose_file(num_high, num_low):
    # Começa com o cabeçalho
    compose_content = YAML_HEADER
    client_counter = 1

    # Adiciona Clientes Rápidos
    for _ in range(num_high):
        client_id_str = f"{client_counter:03d}"
        compose_content += CLIENT_HIGH_PERF_TEMPLATE.format(client_id_str=client_id_str)
        client_counter += 1

    # Adiciona Clientes Lentos
    for _ in range(num_low):
        client_id_str = f"{client_counter:03d}"
        compose_content += CLIENT_LOW_PERF_TEMPLATE.format(client_id_str=client_id_str)
        client_counter += 1

    total_clients = num_high + num_low
    
    try:
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        print(f"✅ Successfully generated 'docker-compose.yml' with {total_clients} clients.")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.")
    parser.add_argument('--high', type=int, required=True)
    parser.add_argument('--low', type=int, required=True)
    args = parser.parse_args()
    
    if args.high < 0 or args.low < 0 or (args.high + args.low) < 1:
        print("❌ Error: Must have at least one client.")
        sys.exit(1)
        
    generate_compose_file(args.high, args.low)

if __name__ == "__main__":
    main()