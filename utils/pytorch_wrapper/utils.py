from datetime import datetime

class TimestampNamer:
    # Creates a name from a timestamp
    def __init__(self, prefix="model"):
        self.prefix = prefix

    def get_name(self, epoch: int)  -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_epoch{epoch+1}"