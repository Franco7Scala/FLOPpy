class CompositeLogger:
    def __init__(self, loggers):
        self.loggers = [l for l in loggers if l is not None]

    def log_batch(self, summary: dict):
        result = summary
        for logger in self.loggers:
            try:
                result = logger.log_batch(summary)
            except Exception:
                pass
        return result

    def log_epoch(self, summary: dict):
        result = summary
        for logger in self.loggers:
            try:
                result = logger.log_epoch(summary)
            except Exception:
                pass
        return result

    def log_summary(self, summary: dict):
        result = summary
        for logger in self.loggers:
            try:
                result = logger.log_summary(summary)
            except Exception:
                pass
        return result

    def close(self):
        for logger in self.loggers:
            try:
                logger.close()
            except Exception:
                pass
