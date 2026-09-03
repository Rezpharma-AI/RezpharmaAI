import json
import queue
import os

class EventBus:
    """
    Smart PubSub Bus.
    Connects to real Redis if available, otherwise uses an in-memory queue.
    """
    def __init__(self):
        self.use_redis = False
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        try:
            import redis
            self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.r.ping()
            self.use_redis = True
            print(f"Connected to Redis Server at {redis_host}:{redis_port}")
        except Exception:
            print("Redis server not found. Using In-Memory Local Bus.")
            self.local_queues = {}
            
    def publish(self, channel, data):
        msg = json.dumps(data)
        if self.use_redis:
            self.r.publish(channel, msg)
        else:
            if channel not in self.local_queues:
                self.local_queues[channel] = []
            for q in self.local_queues[channel]:
                q.put(msg)
                
    def subscribe(self, channel):
        if self.use_redis:
            pubsub = self.r.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        else:
            if channel not in self.local_queues:
                self.local_queues[channel] = []
            q = queue.Queue()
            self.local_queues[channel].append(q)
            return q