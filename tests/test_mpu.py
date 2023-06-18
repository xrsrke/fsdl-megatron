import torch
from torch.multiprocessing import Process, Queue

from fsdl.mpu import MPU


def init_mpu(
    rank, world_size,
    tensor_model_parallel_size, pipeline_model_parallel_size,
    master_addr, master_port, backend, queue
):
    mpu = MPU(
        rank=rank,
        world_size=world_size,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend
    )

    queue.put(mpu)

    # assert mpu.rank == 22
    # assert isinstance(mpu._data_paralell_group, torch.distributed.ProcessGroup)


def test_mpu():
    # launch new process and establish distributed communication using MPU
    WORLD_SIZE = 4
    MASTER_ADDR = "localhost"
    MATER_PORT = 5678
    BACKEND = "gloo"

    TENSOR_MODEL_PARALLEL_SIZE = 2
    PIPELINE_MODEL_PARALLEL_SIZE = 2

    processes = []
    queue = Queue()
    for rank in range(WORLD_SIZE):
        process = Process(target=init_mpu, args=(
            rank, WORLD_SIZE,
            TENSOR_MODEL_PARALLEL_SIZE, PIPELINE_MODEL_PARALLEL_SIZE,
            MASTER_ADDR, MATER_PORT, BACKEND,
            queue
        ))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    while not queue.empty():
        mpu = queue.get()
        assert isinstance(mpu, MPU)
