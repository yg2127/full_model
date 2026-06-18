__all__ = ["FullDMSSystem", "FullDMSConfig"]

def __getattr__(name):
    if name in __all__:
        from .full_system import FullDMSSystem, FullDMSConfig
        return {"FullDMSSystem": FullDMSSystem, "FullDMSConfig": FullDMSConfig}[name]
    raise AttributeError(name)
