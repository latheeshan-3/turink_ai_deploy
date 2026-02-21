import os
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class RedisConfig:
    url = os.getenv("REDIS_URL")  # Preferred (Render provides this)
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB", 0))
    password = os.getenv("REDIS_PASSWORD")


def create_redis_client():
    try:
        # If REDIS_URL exists (Render), use it
        if RedisConfig.url:
            logger.info("Connecting to Redis using REDIS_URL")
            return redis.from_url(
                RedisConfig.url,
                decode_responses=True
            )

        # Otherwise fallback to host/port (local dev)
        logger.info("Connecting to Redis using host/port")
        return redis.Redis(
            host=RedisConfig.host,
            port=RedisConfig.port,
            db=RedisConfig.db,
            password=RedisConfig.password,
            decode_responses=True
        )

    except Exception as e:
        logger.exception("Failed to create Redis client")
        raise


redis_client = create_redis_client()


"""import os
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

class RedisConfig:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB", 0))
    password = os.getenv("REDIS_PASSWORD", None)

redis_client = redis.Redis(
    host=RedisConfig.host,
    port=RedisConfig.port,
    db=RedisConfig.db,
    password=RedisConfig.password,
    decode_responses=True
)
"""