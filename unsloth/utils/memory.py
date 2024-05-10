import torch
def get_memory_stats():
    print(torch.cuda.memory_summary())
    
def get_max_memory_reserved():
    mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak reserved memory = {mem} GB.")
    return mem
def get_mem_stat_keys():
    mem_stats = torch.cuda.memory_stats_as_nested_dict()
    return mem_stats.keys()

def get_mem_stat(key):
    mem_stats = torch.cuda.memory_stats_as_nested_dict()
    try:
        return mem_stats[key]
    except:
        print(f"Key {key} not found in memory stats, run `get_mem_stat_keys()` to see all keys.")
        return None
    