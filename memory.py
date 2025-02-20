import subprocess

def get_gpu_total_memory():
    try:
        # Run nvidia-smi and query total memory for each GPU
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpu_memories = {}
        total_memory = 0
        # Each line in the output corresponds to a GPU.
        for i, line in enumerate(output.strip().splitlines()):
            mem_mb = int(line.strip())
            gpu_memories[f'GPU_{i}'] = mem_mb
            total_memory += mem_mb
        return gpu_memories, total_memory
    except subprocess.CalledProcessError as e:
        print("Error running nvidia-smi:", e)
        return None, None
    except Exception as e:
        print("An error occurred:", e)
        return None, None

if __name__ == '__main__':
    gpu_memories, total_memory = get_gpu_total_memory()
    if gpu_memories is not None:
        for gpu, mem in gpu_memories.items():
            print(f"{gpu}: {mem} MB")
        print(f"Total GPU memory: {total_memory} MB")
