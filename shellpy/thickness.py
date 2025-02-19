class ConstantThickness:
    def __init__(self, h):
        self._h = h

    def __call__(self, xi1=None, xi2=None):
        return self._h