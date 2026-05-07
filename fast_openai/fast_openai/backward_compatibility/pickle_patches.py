import pickle

class PatchedUnpickler(pickle.Unpickler):
    """
    Account for refactor from fast_openai.workers -> fast_openai.results_collector
    """
    def find_class(self, module, name):
        if module == "fast_openai.workers" and name.endswith("Collector"):
            module = "fast_openai.results_collector"
        
        elif module == 'fast_openai.workers' and name == '_WorkerOutput':
            module = 'fast_openai.worker'
            name = 'WorkerOutput'

        return super().find_class(module, name)
    