import copy
import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import datasets
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_datasets_available, PreTrainedTokenizerBase, PreTrainedModel, TrainingArguments, \
    DataCollator, TrainerCallback
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import nested_truncate, IterableDatasetShard, nested_concat, nested_numpify, \
    find_batch_size
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, EvalPrediction, has_length, speed_metrics
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CustomTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            eval_data_collator=None,
            precision_at=1,
            precision_dict=None,
            target_sov=False,
            dot_test=False,
            alias_lookup=None
    ):
        super(CustomTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                                            model_init, compute_metrics, callbacks, optimizers)
        self.eval_data_collator = eval_data_collator
        self.switch = True
        self.train_off = False
        self.dot_test = dot_test
        self.alias_lookup = alias_lookup

        # If we target a SOV language, we can rewrite (e1, r, e2) triple to (e1, e2, r)
        self.target_sov = target_sov

        # Maximum k for precision@k
        self.precision_at = precision_at

        # Dict, Key: Tokenized Subj+Rel, Value: K
        self.precision_dict = precision_dict

        # Precision@K value for every sample
        self.k_list = []
        self.compute_k(eval_dataset)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        # Only used for evaluation purposes
        if 'eval_correct_predictions' in logs:
            del logs['eval_correct_predictions']

        if 'train_runtime' not in logs.keys() and not self.train_off:
            if self.switch:
                logger.info('Epoch: ' + str(int(logs['epoch'])))
            self.switch = not self.switch
        elif not self.train_off:
            logger.info('Summary')
            self.train_off = True

        logger.info(logs)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # Computes the k's for evaluation set (for precision@k)
    def compute_k(self, eval_dataset):
        if self.precision_at > 1 and self.compute_metrics.__name__ not in ['precision_at_k_fixed',
                                                                           'mean_reciprocal_rank']:
            # Get the eval dataloader to iterate over the evaluation data
            eval_dataloader = self.get_eval_dataloader(eval_dataset)  # Validation list essentially

            self.k_list = []
            # Inputs are enumerated in batch_size
            for _, inputs in enumerate(eval_dataloader):
                for label in inputs['labels']:
                    # To get the k's in the right order, we need to iterate over the evaluation data
                    tensor_list = label.tolist()
                    # Since there might be padding, we need to also consider -100's that were used for padding
                    label_list = label.tolist()[:-(2 + tensor_list.count(-100))]
                    k = self.precision_dict[repr(label_list)]
                    self.k_list.append(k)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            custom_eval=False
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        # If we are running evaluation on a new dataset, compute labels (in case of precision@k)
        if eval_dataset is not None and eval_dataset != self.eval_dataset:
            self.compute_k(eval_dataset)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # To avoid deepcopy during training...
        if not custom_eval:
            self.log(output.metrics)
        else:
            self.log(copy.deepcopy(output.metrics))

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        # logger.info(f"***** Running {description} *****")
        # if has_length(dataloader.dataset):
        #     logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        # else:
        #     logger.info("  Num examples: Unknown")
        # logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length
        all_preds_alias = []
        all_labels_alias = []

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # ***** CUSTOM BEHAVIOUR *****
            # Unpad inputs to use them as key for alias dict later
            if self.alias_lookup is not None:
                inputs_unpadded = []
                inputs_padded = copy.deepcopy(inputs['input_ids'])
                for input_padded in inputs_padded:
                    # Is it padded?
                    if 0 in input_padded:
                        # Remove 0s
                        for i, j in enumerate(reversed(input_padded)):
                            if j:
                                input_padded = input_padded[:-1 * i]
                                break

                    inputs_unpadded.append(input_padded.tolist())

            if self.target_sov:
                # ~~~ FOR SOV, i.e. (entity, entity, relation):
                cls_indices = (inputs['labels'] == self.tokenizer.cls_token_id).nonzero(as_tuple=True)
                list(cls_indices)[1] += 2

                masked_indices = torch.full(inputs['labels'].shape, False)
                masked_indices[cls_indices] = True

                # Replace all other tokens with -100
                inputs['labels'][~masked_indices] = -100
                inputs['input_ids'][masked_indices] = self.tokenizer.mask_token_id
                # ~~~
            else:
                # MASK object entity token of (Entity, Relation, Entity) Triple
                sep_indices = (inputs['labels'] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)

                # Since we know that the last token before the sep token is supposed to be the object..
                if self.dot_test:
                    list(sep_indices)[1] -= 2
                else:
                    list(sep_indices)[1] -= 1

                masked_indices = torch.full(inputs['labels'].shape, False)
                masked_indices[sep_indices] = True

                # Replace all other tokens with -100
                inputs['labels'][~masked_indices] = -100
                inputs['input_ids'][masked_indices] = self.tokenizer.mask_token_id

            # ***************************

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # For every fact in this batch
            if self.alias_lookup is not None:

                if len(inputs_unpadded) < 64:
                    alias_step = len(inputs_unpadded)
                    step_rate = 1
                else:
                    step_rate = 4
                    alias_step = int(len(inputs_unpadded)/step_rate)

                logits_alias_list = []
                labels_alias_list = []

                for i in range(step_rate):
                    # Collect all aliases of this batch
                    test_alias = None
                    alias_offset_list = []  # Saves how many aliases and translons relation has

                    # Iterate over all facts in this step
                    for fact in inputs_unpadded[i*alias_step:(i+1)*alias_step]:
                        # Get the aliases and translations - {'fact_tokenized': [tokenized_alias_facts]
                        alias_out = self.alias_lookup[str(fact)]

                        if len(alias_out) != 0:
                            alias_offset_list.append(alias_out['input_ids'].shape[0])
                        else:
                            alias_offset_list.append(0)
                            continue

                        # Concat all aliases together (that makes training much faster)
                        if test_alias is None:
                            test_alias = copy.deepcopy(alias_out)
                        else:
                            for key in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']:
                                test_alias[key] = torch.cat((test_alias[key], alias_out[key]))

                    # What if relation has NO alias! We only do this if there were alias in the list
                    if test_alias is not None:
                        # Mask them
                        sep_indices = (test_alias['labels'] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)

                        # Since we know that the last token before the sep token is supposed to be the object..
                        if self.dot_test:
                            list(sep_indices)[1] -= 2
                        else:
                            list(sep_indices)[1] -= 1

                        masked_indices = torch.full(test_alias['labels'].shape, False)
                        masked_indices[sep_indices] = True

                        # Replace all other tokens with -100
                        test_alias['labels'][~masked_indices] = -100
                        test_alias['input_ids'][masked_indices] = self.tokenizer.mask_token_id

                        # Predict
                        _, logits_alias, labels_alias = self.prediction_step(model, test_alias, prediction_loss_only, ignore_keys=ignore_keys)

                        logits_alias, labels_alias = logits_alias.cpu(), labels_alias.cpu()

                        # Flatten and select precision@k
                        logits_reduced_alias = torch.topk(logits_alias, k=self.precision_at, dim=-1)[1]  # For Precision@K
                        if self.precision_at == 1:
                            logits_reduced_alias = torch.flatten(logits_reduced_alias, start_dim=-2, end_dim=-1)

                    # Reshape alias results such that we have a list of all alias results for each relation
                    total_offset = 0
                    for alias_offset in alias_offset_list:
                        if alias_offset == 0:
                            logits_alias_list.append([])
                            labels_alias_list.append([])
                            continue

                        logits_alias_list.append(logits_reduced_alias[total_offset:total_offset + alias_offset].cpu())
                        labels_alias_list.append(labels_alias[total_offset:total_offset + alias_offset].cpu())

                        total_offset += alias_offset

                # Save it for metrics
                all_preds_alias += logits_alias_list
                all_labels_alias += labels_alias_list

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)

                # Necessary to not get OOM Issues when passing around huge tensors
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                # logits_reduced = torch.argmax(logits, dim=-1)
                if self.compute_metrics.__name__ == 'mean_reciprocal_rank':
                    logits_reduced = torch.topk(logits, k=1000, dim=-1)[1]
                else:
                    logits_reduced = torch.topk(logits, k=self.precision_at, dim=-1)[1]  # For Precision@K
                    if self.precision_at == 1:
                        logits_reduced = torch.flatten(logits_reduced, start_dim=-2, end_dim=-1)
                preds_host = logits_reduced if preds_host is None else nested_concat(preds_host, logits_reduced,
                                                                                     padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if self.compute_metrics.__name__ in ['precision_at_k_fixed', 'mean_reciprocal_rank']:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
            elif self.precision_at > 1:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), self.k_list)
            elif self.compute_metrics.__name__ in ['precision_at_one_alias']:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), (all_preds_alias, all_labels_alias))
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    # Override to replace data-collator for custom evaluation loop
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.eval_data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
