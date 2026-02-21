import os
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
