class IncompleteOutput(ValueError):
    """A model response exists but must not execute; safe metadata only."""
    def __init__(self, details):
        super().__init__('Model output incomplete; no action executed')
        self.details = details
