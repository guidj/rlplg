import time
import uuid


def run_id():
    uuid4_head = next(iter(str(uuid.uuid4()).split("-")))
    timestamp = int(time.time())
    return f"{uuid4_head}-{timestamp}"
