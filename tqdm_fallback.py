"""
Utility module for robust progress bar implementation
providing fallback mechanisms when tqdm.notebook is not available
"""

def get_tqdm():
    """
    Returns the appropriate tqdm implementation based on the environment.
    Tries notebook version first, then falls back to standard tqdm,
    and finally returns a simple progress tracking class if all else fails.
    """
    try:
        # First try notebook tqdm
        from tqdm.notebook import tqdm
        return tqdm
    except ImportError:
        try:
            # Fall back to standard tqdm
            from tqdm import tqdm
            return tqdm
        except ImportError:
            # Create a simple progress class as last resort
            class SimpleTqdm:
                def __init__(self, iterable=None, desc=None, total=None, **kwargs):
                    self.iterable = iterable
                    self.desc = desc
                    self.total = total if total is not None else (
                        len(iterable) if hasattr(iterable, "__len__") else None
                    )
                    self.n = 0
                    print(f"Starting {desc}: 0/{self.total if self.total else '?'}")
                
                def __iter__(self):
                    for item in self.iterable:
                        yield item
                        self.n += 1
                        if self.n % 10 == 0:  # Print progress every 10 items
                            print(f"{self.desc}: {self.n}/{self.total if self.total else '?'}")
                
                def set_postfix(self, **kwargs):
                    # Convert kwargs to str
                    postfix_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                           for k, v in kwargs.items())
                    print(f"{self.desc}: {self.n}/{self.total if self.total else '?'} [{postfix_str}]")
                
                def update(self, n=1):
                    self.n += n
                
                def close(self):
                    print(f"Finished {self.desc}: {self.n}/{self.total if self.total else '?'}")
            
            return SimpleTqdm

def create_progress_bar(iterable=None, desc=None, total=None, **kwargs):
    """
    Creates a progress bar with the appropriate implementation
    """
    tqdm_class = get_tqdm()
    return tqdm_class(iterable=iterable, desc=desc, total=total, **kwargs)
