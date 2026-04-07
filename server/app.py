try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required for the web interface.") from e

from models import TrafficAction, TrafficObservation
from server.traffic_environment import TrafficEnvironment

app = create_app(
    TrafficEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="traffic_rl",
    max_concurrent_envs=5,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
