import os

import torch


class MPU:
    def __init__(
        self,
        rank,
        world_size,
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        master_addr,
        master_port,
        backend
    ):
        if not torch.distributed.is_initialized():
            os.environ["MASTER_ADDR"] = str(master_addr)
            os.environ["MASTER_PORT"] = str(master_port)

            # self.set_device(rank)
            torch.distributed.init_process_group(
                rank=rank,
                world_size=world_size,
                backend=backend,
            )

        current_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.debug = True

        self.num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
        self._data_paralell_group = None

        # init data parallel group
        self.init_data_parallel_group(
            rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size
        )
        # init tensor parallel and pipeline paralell groups

    def set_device(self, rank):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            device = rank % num_gpus
            torch.cuda.set_device(device)

    def init_data_parallel_group(
        self,
        rank,
        tensor_model_parallel_size,
        pipeline_model_parallel_size
    ):
        for i in range(pipeline_model_parallel_size):
            start_rank = i*self.num_pipeline_model_parallel_groups
            end_rank = (i+1)*self.num_pipeline_model_parallel_groups

            for j in range(tensor_model_parallel_size):
                ranks = list(range(
                    start_rank+j,
                    end_rank,
                    tensor_model_parallel_size
                ))

                if rank in ranks:
                    group = torch.distributed.new_group(ranks=ranks)
                    self._data_paralell_group = group
