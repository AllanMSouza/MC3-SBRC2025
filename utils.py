import math
import os
import torch
from flwr.common import Context
from flwr.server import ServerConfig
from transformers import Trainer
from functools import lru_cache
from typing import Any, Dict
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling,
)
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# Treinamento
class CustomTrainer(Trainer):
    def __init__(self, cid, **kwargs):
        super().__init__(**kwargs)
        self.train_losses = {}
        self.validation_losses = {}
        self.cid = cid

    def log(self, logs):
        # Save client losses
        super().log(logs)
        if "loss" in logs:
            self.train_losses[self.cid] = float(logs["loss"])
        if "eval_loss" in logs:
            self.validation_losses[self.cid] = float(logs["eval_loss"])

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=1e-5):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

# Injeção de dependências
# ===== Builders =====
# ===== Model Builder =====
class ModelBuilder:
    def __init__(self):
        self._model_name: str = ""
        self._use_lora: bool = False
        self._lora_rank: int = 8

    def with_model_name(self, model_name: str) -> "ModelBuilder":
        """Define qual modelo carregar."""
        self._model_name = model_name
        return self

    def enable_lora(self, flag: bool = True) -> "ModelBuilder":
        """Ativa ou desativa LoRA."""
        self._use_lora = flag
        return self

    def with_lora_rank(self, rank: int) -> "ModelBuilder":
        """Define o rank de LoRA (r)."""
        self._lora_rank = rank
        return self

    def build(self):
        """Efetivamente carrega o modelo e (se pedido) aplica LoRA."""
        if not self._model_name:
            raise ValueError("Você deve chamar .with_model_name() antes de build()")

        # 1) Escolhe a família de modelo
        lname = self._model_name.lower()
        if "bert" in lname:
            model = AutoModelForMaskedLM.from_pretrained(self._model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(self._model_name)

        # 2) Se LoRA estiver habilitado, aplica-o
        if self._use_lora:
            lora_cfg = LoraConfig(
                r=self._lora_rank,
                lora_alpha=self._lora_rank * 2,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_cfg)

        return model


# ===== ServerConfig Builder =====
class ServerConfigBuilder:
    def __init__(self):
        self._num_rounds = 1

    def with_rounds(self, n: int) -> "ServerConfigBuilder":
        """Define o número de rounds do servidor."""
        self._num_rounds = n
        return self

    def build(self) -> ServerConfig:
        """Constroi e retorna a configuração do servidor."""
        return ServerConfig(num_rounds=self._num_rounds)


# ===== Tokenizer Builder =====
class TokenizerBuilder:
    def __init__(self):
        self._model_name: str = ""

    def with_model_name(self, model_name: str) -> "TokenizerBuilder":
        """Define qual modelo carregar."""
        self._model_name = model_name
        return self

    def build(self):
        """
        Carrega e retorna um AutoTokenizer configurado:
        - modelos BERT (contêm 'bert' no nome) sem padding.
        - demais modelos com padding=True e pad_token = eos_token.
        """
        name_lower = self._model_name.lower()
        if "bert" in name_lower:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                use_fast=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                padding=True,
                use_fast=True,
            )
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


# ===== Traning Config Builder =====
class TraningConfigBuilder:
    def __init__(self):
        self.output_dir: str = ""
        self.logging_dir = ""
        self.logging_steps = 11
        self.learning_rate = 1e-3
        self.weight_decay = 0.01
        self.max_steps = 10
        self.num_train_epochs = 1
        self.save_steps = 1000
        self.eval_strategy = "steps"
        self.eval_steps = 11
        self.fp16 = True
        self.optim = "paged_adamw_8bit"
        self.lr_scheduler_type = "constant"

    def with_output_dir(self, output_dir: str) -> "TraningConfigBuilder":
        """Define qual modelo carregar."""
        self.output_dir = output_dir
        return self

    def with_logging_dir(self, logging_dir: str) -> "TraningConfigBuilder":
        self.logging_dir = logging_dir
        return self

    def with_logging_steps(self, logging_steps: int) -> "TraningConfigBuilder":
        self.logging_steps = logging_steps
        return self

    def with_learning_rate(self, learning_rate: float) -> "TraningConfigBuilder":
        self.learning_rate = learning_rate
        return self

    def with_weight_decay(self, weight_decay: float) -> "TraningConfigBuilder":
        self.weight_decay = weight_decay
        return self

    def with_max_steps(self, max_steps: int) -> "TraningConfigBuilder":
        self.max_steps = max_steps
        return self

    def with_num_train_epochs(self, num_train_epochs: int) -> "TraningConfigBuilder":
        self.num_train_epochs = num_train_epochs
        return self

    def with_save_steps(self, save_steps: int) -> "TraningConfigBuilder":
        self.save_steps = save_steps
        return self

    def with_eval_strategy(self, eval_strategy: str) -> "TraningConfigBuilder":
        self.eval_strategy = eval_strategy
        return self

    def with_eval_steps(self, eval_steps: int) -> "TraningConfigBuilder":
        self.eval_steps = eval_steps
        return self

    def with_fp16(self, fp16: bool) -> "TraningConfigBuilder":
        self.fp16 = fp16
        return self

    def with_optim(self, optim: str) -> "TraningConfigBuilder":
        self.optim = optim
        return self

    def with_lr_scheduler_type(self, lr_scheduler_type: str) -> "TraningConfigBuilder":
        self.lr_scheduler_type = lr_scheduler_type
        return self

    def build(self):
        return TrainingArguments(output_dir=f"{self.output_dir}/fl-results", logging_dir=f"{self.logging_dir}/logs",
                                 logging_steps=self.logging_steps, learning_rate=self.learning_rate,
                                 weight_decay=self.weight_decay, max_steps=self.max_steps,
                                 num_train_epochs=self.num_train_epochs, save_steps=self.save_steps,
                                 eval_strategy=self.eval_strategy, eval_steps=self.eval_steps, fp16=self.fp16,
                                 optim=self.optim, lr_scheduler_type=self.lr_scheduler_type, report_to=[])


# ===== Trainer Builder =====
class TrainerBuilder:
    def __init__(self):
        self.cid = 0
        self.model = None
        self.args = None
        self.train_dataset = None
        self.tokenizer = None
        self.eval_dataset = None
        self.model_name = ""

    def with_cid(self, cid) -> "TrainerBuilder":
        self.cid = cid
        return self

    def with_model(self, model) -> "TrainerBuilder":
        self.model = model
        return self

    def with_args(self, args) -> "TrainerBuilder":
        self.args = args
        return self

    def with_train_dataset(self, train_dataset) -> "TrainerBuilder":
        self.train_dataset = train_dataset
        return self

    def with_tokenizer(self, tokenizer) -> "TrainerBuilder":
        self.tokenizer = tokenizer
        return self

    def with_eval_dataset(self, eval_dataset) -> "TrainerBuilder":
        self.eval_dataset = eval_dataset
        return self

    def with_model_name(self, model_name) -> "TrainerBuilder":
        self.model_name = model_name
        return self

    def build(self):
        if "bert" in self.model_name:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15,  # 15% tokens will be masked
            )

            trainer = CustomTrainer(cid=self.cid,
                                    model=self.model,
                                    args=self.args,
                                    train_dataset=self.train_dataset,
                                    tokenizer=self.tokenizer,
                                    eval_dataset=self.eval_dataset,
                                    data_collator=data_collator)
        else:
            trainer = CustomTrainer(cid=self.cid,
                                    model=self.model,
                                    args=self.args,
                                    train_dataset=self.train_dataset,
                                    tokenizer=self.tokenizer,
                                    eval_dataset=self.eval_dataset)
        return trainer


# ===== Singletons =====
# ===== Tokenizer Singleton =====
@lru_cache(maxsize=1)
def get_tokenizer(model_name: str) -> Any:
    """Retorna sempre a mesma instância de AutoTokenizer para um dado model_name."""
    builder = TokenizerBuilder().with_model_name(model_name)
    return builder.build()


# ===== Factories =====
# ===== on_fit_config_fn Factory =====
class FitConfigFactory:
    def __init__(self, context: Dict):
        # extrai e atribui apenas uma vez
        self.num_rounds = context["num-rounds"]
        self.initial_lr = context["initial-lr"]
        self.min_lr = context["min-lr"]
        self.dataset_path = context["dataset-path"]
        self.results_path = context["results-path"]
        self.model_name = context["model-name"]
        self.lora = context["lora"]

    def __call__(self, server_round: int) -> Dict[str, Any]:
        # devolve o dict completo ao FL
        return {
            "current_round": server_round,
            "num_rounds": self.num_rounds,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "dataset_path": self.dataset_path,
            "results_path": self.results_path,
            "model_name": self.model_name,
            "lora": self.lora,
        }

# Modelo
def get_parameters(model, lora: bool = False):
    params = []
    for name, param in model.named_parameters():
        if lora:
            if 'lora' in name:
                params.append(param.detach().cpu().numpy())
        else:
            params.append(param.detach().to(torch.float16).cpu().numpy())
    return params


def set_parameters(model, parameters, lora: bool = False):
    i = 0
    for name, param in model.named_parameters():
        if lora:
            if 'lora' in name:
                param.data = torch.tensor(parameters[i]).to(param.dtype)
                i += 1
        else:
            param.data = torch.tensor(parameters[i]).to(param.dtype)
            i += 1

# Avaliação
def is_in_top_k(top_k, target):
    return target in top_k

def next_token_top_k(samples, model, tokenizer, device):
    """
    Predict the next token for a given text using a llm model.
    Calculate with the correct token is in the top k predictions.
    """
    accuracies = {'top1': [], 'top3': [], 'top5': [], 'top10': []}

    model.to(device)
    with torch.no_grad():
        for text in samples:

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                               padding_side="right")
            inputs = {key: inputs[key].to(device) for key in inputs}

            outputs = model(**inputs)
            logits = outputs.logits

            for k in [1, 3, 5, 10]:
                top_k = torch.topk(logits, k, dim=-1).indices
                top_k = top_k.cpu().numpy()

                tokens = inputs['input_ids'][0].cpu().numpy()
                correct = sum(is_in_top_k(top_k[0], token) for token in tokens)

                accuracies[f'top{k}'].append(correct / len(tokens))

    return accuracies

def get_evaluate_fn(context: Context, device: str):
    model_name = context["model-name"]
    lora_rank = context["lora-rank"]
    lora = context["lora"]
    testset_path = context["testset-path"]
    results_path = context["results-path"]
    nrows = context["nrows"]
    experiment_name = context["experiment-name"]
    tokenizer = get_tokenizer(model_name)

    def evaluate(server_round, parameters_ndarrays, config):

        # Preparando o modelo para ser avaliado
        # Construindo
        model = ModelBuilder().with_model_name(model_name).enable_lora(lora).with_lora_rank(lora_rank).build()
        # Definindo os pesos com os do novo modelo global
        set_parameters(model, parameters_ndarrays, lora)
        # Enviando para o dispositivo: gpu ou cpu
        model.to(device)

        # Preparando os dados para avaliação
        # Lendo dataset de teste de um arquivo csv
        data = pd.read_csv(testset_path, nrows=nrows)
        # Embaralhando os dados
        data = data.sample(frac=1, random_state=0).reset_index(drop=True)
        # Obtendo 50/50 de cada classe
        data = pd.concat([data[data['Label'] == 1].head(1000), data[data['Label'] == 0].head(1000)])
        # Extraindo amostras e rótulos
        labels = data["Label"].tolist()
        samples = [str(i) for i in data["Content"].tolist()]

        # Criando diretório de resultados se não existir e DataFrames que armazenam resultados e serão exportados.
        os.makedirs(results_path, exist_ok=True)
        df_acc = pd.DataFrame()
        df_acc.to_csv(f"{results_path}/results_accs_{experiment_name}.csv", index=False)

        df_f1 = pd.DataFrame()
        df_f1.to_csv(f"{results_path}/results_f1_{experiment_name}.csv", index=False)

        # Obtendo as acurácias das amostras
        accuracies = next_token_top_k(samples, model, tokenizer, device)
        # K - índice do top
        for k in [1, 3, 5, 10]:

            df_results = pd.DataFrame(accuracies)
            df_results['label'] = labels
            df_results['round'] = server_round
            df_results['k'] = k

            df_acc = pd.concat([df_acc, df_results])

            df_acc.to_csv(f"{results_path}/results_accs_{experiment_name}.csv", index=False)

            ths = np.linspace(0, 1, 1000)

            best_f1 = 0
            best_th = 0

            for th in ths:
                df_results['pred'] = df_results[f'top{k}'] < th
                df_results['pred'] = df_results['pred'].astype(int)

                f1 = f1_score(df_results['label'], df_results['pred'])

                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th

            df_results['pred'] = df_results[f'top{k}'] < best_th
            df_results['pred'] = df_results['pred'].astype(int)

            f1 = f1_score(df_results['label'], df_results['pred'], zero_division=0.0)
            precision = precision_score(df_results['label'], df_results['pred'], zero_division=0.0)
            recall = recall_score(df_results['label'], df_results['pred'], zero_division=0.0)

            print(f'Round: {server_round}')
            print(f'K: {k}')
            print(f'Threshold: {best_th}')
            print(f'F1: {f1}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')

            df_f1 = pd.concat([df_f1, pd.DataFrame(
                {"round": [server_round], "k": [k], "threshold": [best_th], "f1": [f1], "precision": [precision],
                 "recall": [recall]})])
            df_f1.to_csv(f"{results_path}/results_f1_{experiment_name}.csv", index=False)

    return evaluate