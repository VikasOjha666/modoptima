import math

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import SparsificationGroupLogger

from modoptima.Train.yolov7tiny.utils.torch_utils import is_parallel


class SparseMLWrapper(object):
    def __init__(self, model, recipe):
        self.enabled = bool(recipe)
        self.model = model.module if is_parallel(model) else model
        self.recipe = recipe
        self.manager = ScheduledModifierManager.from_yaml(recipe) if self.enabled else None
        self.logger = None

    def state_dict(self):
        return {
            'recipe': str(self.manager) if self.enabled else None,
        }

    def apply(self):
        if not self.enabled:
            return

        self.manager.apply(self.model)

    def initialize(self, start_epoch):
        if not self.enabled:
            return

        self.manager.initialize(self.model, start_epoch)

    def initialize_loggers(self, logger, tb_writer, wandb_logger, rank):
        self.logger = logger

        if not self.enabled or rank not in [-1, 0]:
            return

        def _logging_lambda(tag, value, values, step, wall_time, level):
            if not wandb_logger or not wandb_logger.wandb:
                return

            if value is not None:
                wandb_logger.log({tag: value})

            if values:
                wandb_logger.log(values)

        self.manager.initialize_loggers([
            SparsificationGroupLogger(
                lambda_func=_logging_lambda,
                tensorboard=tb_writer,
            )
        ])

        if wandb_logger.wandb:
            artifact = wandb_logger.wandb.Artifact('recipe', type='recipe')
            with artifact.new_file('recipe.yaml') as file:
                file.write(str(self.manager))
            wandb_logger.wandb.log_artifact(artifact)

    def modify(self, scaler, optimizer, model, dataloader):
        if not self.enabled:
            return scaler

        return self.manager.modify(model, optimizer, steps_per_epoch=len(dataloader), wrap_optim=scaler)

    def check_lr_override(self, scheduler):
        # Override lr scheduler if recipe makes any LR updates
        if self.enabled and self.manager.learning_rate_modifiers:
            self.logger.info('Disabling LR scheduler, managing LR using SparseML recipe')
            scheduler = None

        return scheduler

    def check_epoch_override(self, epochs):
        # Override num epochs if recipe explicitly modifies epoch range
        if self.enabled and self.manager.epoch_modifiers and self.manager.max_epochs:
            epochs = self.manager.max_epochs or epochs  # override num_epochs
            #self.logger.info(f'Overriding number of epochs from SparseML manager to {epochs}')

        return epochs

    def qat_active(self, epoch):
        if not self.enabled or not self.manager.quantization_modifiers:
            return False

        qat_start = max([mod.start_epoch for mod in self.manager.quantization_modifiers])

        return qat_start < epoch + 1

    def reset_best(self, epoch):
        if not self.enabled:
            return False

        # if pruning is active or quantization just started, need to reset best checkpoint
        # this is in case the pruned and/or quantized model do not fully recover
        pruning_start = math.floor(max([mod.start_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        pruning_end = math.ceil(max([mod.end_epoch for mod in self.manager.pruning_modifiers])) \
            if self.manager.pruning_modifiers else -1
        qat_start = math.floor(max([mod.start_epoch for mod in self.manager.quantization_modifiers])) \
            if self.manager.quantization_modifiers else -1

        return (pruning_start <= epoch <= pruning_end) or epoch == qat_start