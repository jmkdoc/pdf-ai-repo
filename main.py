import uvicorn
from src.api.endpoints import app
from src.utils import ConfigManager, Logger

logger = Logger.setup_logger(__name__)
config = ConfigManager()

if __name__ == "__main__":
    logger.info(f"Starting PDF AI Repository API")
    logger.info(f"Host: {config.get('api.host')}, Port: {config.get('api.port')}")
    
    uvicorn.run(
        app,
        host=config.get("api.host", "127.0.0.1"),
        port=config.get("api.port", 8080),
        reload=config.get("app.debug", True)
    )
