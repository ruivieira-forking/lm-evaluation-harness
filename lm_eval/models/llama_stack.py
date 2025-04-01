import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type
import logging

from llama_stack_client.types import SamplingParams
from llama_stack_client.types.inference_chat_completion_params import Logprobs
from llama_stack_client.types.shared.sampling_params import StrategyGreedySamplingStrategy, StrategyTopKSamplingStrategy
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import simple_parse_args_string

eval_logger = logging.getLogger(__name__)

class LogLikelihoodResult(NamedTuple):
    log_likelihood: float
    is_greedy: bool


@register_model("llama_stack")
class LlamaStackLLM(LM):
    """
    Implementation of LM model interface for evaluating Llama Stack models with the lm_eval framework.
    See https://llama-stack.readthedocs.io/en/latest/references/api_reference/index.html#/paths/v1-inference-completion/post for reference.
    """

    @classmethod
    def create_from_arg_string(
        cls: Type["LlamaStackLLM"],
        arg_string: str,
        tokenizer=None,
    ) -> "LlamaStackLLM":
        """
        Allow the user to specify model parameters in CLI arguments.
        """
        args = simple_parse_args_string(arg_string)
        # Use model_arg's model as model_id
        model_id = args.pop("model_id", args.pop("model", None))
        if model_id is None:
            raise ValueError(
                "Either 'model_id' or 'model' is required, please pass it in 'model_args'"
            )

        base_url = args.pop("base_url", "http://localhost:8321")
        timeout = args.pop("timeout", None)
        # Sampling parameters for Llama Stack API
        sampling_params = {}
        if args.get("temperature", None) is not None:
            sampling_params["temperature"] = float(args.pop("temperature"))
        if args.get("top_p", None) is not None:
            sampling_params["top_p"] = float(args.pop("top_p"))
        if args.get("top_k", None) is not None:
            sampling_params["top_k"] = int(args.pop("top_k"))
        if args.get("max_tokens", None) is not None:
            sampling_params["max_tokens"] = int(args.pop("max_tokens", 256))
        if args.get("seed", None) is not None:
            sampling_params["seed"] = int(args.pop("seed"))
        if args.get("presence_penalty", None) is not None:
            sampling_params["presence_penalty"] = float(args.pop("presence_penalty"))
        if args.get("frequency_penalty", None) is not None:
            sampling_params["frequency_penalty"] = float(args.pop("frequency_penalty"))

        return cls(
            base_url=base_url,
            model_id=model_id,
            sampling_params=sampling_params,
            timeout=timeout,
            **args,
        )

    """
    Args:
        base_url (str): Base URL of the Llama Stack API.
        model_id (str): Model ID to use for inference.
        sampling_params (Optional[Dict[str, Any]]): Sampling parameters for inference.
        timeout (Optional): Timeout in seconds.
        api_key (Optional[str]): API key for the Llama Stack API.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        sampling_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tokenizer: Optional[str] = None,
    ) -> None:
        try:
            from llama_stack_client import LlamaStackClient

        except ImportError:
            raise ImportError(
                "Could not import llama_stack_client: Please install lm_eval[llama_stack] package."
            )
        super().__init__()
        self.base_url = base_url
        self.model_id = model_id
        self.max_new_tokens = max_tokens
        self.lls_sampling_params = LlamaStackLLM._convert_sampling_parameter(
            sampling_params
        )
        self.timeout = timeout
        self.tokenizer = tokenizer

        self.api_key = api_key or os.getenv("LLAMA_STACK_API_KEY")
        self.client = LlamaStackClient(
            base_url=self.base_url,
            timeout=timeout,
        )

    @staticmethod
    def _convert_sampling_parameter(
        sampling_params: Optional[Dict[str, Any]], max_tokens: Optional[int] = 1500
    ) -> "SamplingParams":
        """
        Convert the sampling parameters dictionary to a SamplingParams instance.

        Returns:
            SamplingParams: A properly configured SamplingParams instance for use with the Llama Stack client.
        """
        from llama_stack_client.types import SamplingParams
        from llama_stack_client.types.shared.sampling_params import (
            StrategyGreedySamplingStrategy,
            StrategyTopKSamplingStrategy,
            StrategyTopPSamplingStrategy,
        )

        if "temperature" in sampling_params or "top_p" in sampling_params:
            strategy = StrategyTopPSamplingStrategy(
                type="top_p",
                temperature=sampling_params.get("temperature"),
                top_p=sampling_params.get("top_p"),
            )
        elif "top_k" in sampling_params:
            strategy = StrategyTopKSamplingStrategy(
                type="top_k", top_k=sampling_params.get("top_k")
            )
        else:
            strategy = StrategyGreedySamplingStrategy(type="greedy")

        params = SamplingParams(
            strategy=strategy,
            max_tokens=max_tokens,
            repetition_penalty=sampling_params.get("repetition_penalty"),
        )

        return params

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _check_model_logprobs_support(self):
        """
        Verifies if the model supports returning log probabilities for input tokens.
        This function sends a prompt to the model and checks whether the model's response
        includes log probabilities for the input tokens. If log probabilities are not present,
        it raises a `RuntimeError`.
        Raises:
            RuntimeError: If the model does not return log probabilities for input tokens.
        """
        tokens = self.client.inference.completion(
            content="The best ice cream flavor is:",
            model_id=self.model_id,
            sampling_params=StrategyTopKSamplingStrategy(type="top_k", top_k=1),
            logprobs=Logprobs(top_k=1),
            extra_headers=self._get_headers(),
        )

        # FIXME: Debug remove
        eval_logger.debug(f"Logprobs check response type: {type(tokens)}")
        eval_logger.debug(f"Logprobs check response dir: {dir(tokens)}")
        eval_logger.debug(f"Logprobs check response: {tokens}")

        if hasattr(tokens, "logprobs"):
            eval_logger.debug(f"Logprobs type: {type(tokens.logprobs)}")
            eval_logger.debug(
                f"First logprob item type: {type(tokens.logprobs[0]) if tokens.logprobs else 'None'}"
            )
            if tokens.logprobs:
                eval_logger.debug(f"First logprob item dir: {dir(tokens.logprobs[0])}")

        if not hasattr(tokens, "logprobs") or tokens.logprobs is None:
            raise RuntimeError(
                f"Model {self.model_id} is not supported: does not return logprobs for input tokens"
            )

        if not hasattr(tokens.logprobs, "__len__"):
            raise RuntimeError(
                f"Model {self.model_id} is not supported: unexpected logprobs format"
            )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text responses for a list of requests.
        """
        results = []
        batch_size = 1

        for i in tqdm(
            range(0, len(requests), batch_size),
            desc=f"Generating completions in batches of {batch_size}",
        ):
            batch = requests[i : i + batch_size]
            batch_results = []

            for request in batch:
                prompt = request.args[0]
                try:
                    response = self.client.inference.completion(
                        content=prompt,
                        model_id=self.model_id,
                        sampling_params=self.lls_sampling_params,
                        extra_headers=self._get_headers(),
                    )

                    eval_logger.info(f"Response from generate_until: {response}")
                    batch_results.append(response.content)
                    self.cache_hook.add_partial(
                        "generated_text", prompt, response.content
                    )

                except Exception as e:
                    eval_logger.error(f"Error while generating text: {str(e)}")
                    batch_results.append("")
            results.extend(batch_results)

        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Calculate the log-likelihood of continuation tokens given context tokens.

        Args:
            requests: List of requests containing context tokens and continuation tokens.

        Returns:
            List of tuples containing log-likelihood and is_greedy flag.
        """
        self._check_model_logprobs_support()
        results = []
        for request in tqdm(requests):
            context, continuation = request.args

            try:
                if self.tokenizer:
                    context_enc = self.tokenizer.encode(context)["tokens"]
                    context_continuation_enc = self.tokenizer.encode(
                        context + continuation
                    )["tokens"]

                    if len(context_enc) >= len(context_continuation_enc):
                        # If tokenization is the same, return 0 logprob and True
                        result = (0.0, True)
                        results.append(result)
                        self.cache_hook.add_partial(
                            "loglikelihood", (context, continuation), result
                        )
                        continue

                    continuation_enc = context_continuation_enc[len(context_enc) :]
                    full_text = context + continuation
                else:
                    # Use Llama Stack tokenization if no tokenizer was provided
                    context_response = self.client.inference.completion(
                        content=context,
                        model_id=self.model_id,
                        sampling_params=StrategyTopKSamplingStrategy(
                            type="top_k", top_k=1
                        ),
                        logprobs=Logprobs(top_k=1),
                        extra_headers=self._get_headers(),
                    )

                    eval_logger.debug(
                        f"Context response type: {type(context_response)}"
                    )
                    eval_logger.debug(f"Context response dir: {dir(context_response)}")
                    eval_logger.debug(
                        f"Context response logprobs type: {type(context_response.logprobs)}"
                    )

                    context_tokens = []
                    for token_logprob in context_response.logprobs:
                        if (
                            hasattr(token_logprob, "logprobs_by_token")
                            and token_logprob.logprobs_by_token
                        ):
                            token = list(token_logprob.logprobs_by_token.keys())[0]
                            context_tokens.append(token)

                    eval_logger.debug(f"Extracted context tokens: {context_tokens}")

                    full_text = context + continuation
                    full_response = self.client.inference.completion(
                        content=full_text,
                        model_id=self.model_id,
                        sampling_params=StrategyTopKSamplingStrategy(
                            type="top_k", top_k=1
                        ),
                        logprobs=Logprobs(top_k=5),
                        extra_headers=self._get_headers(),
                    )

                    full_tokens = []
                    for token_logprob in full_response.logprobs:
                        if (
                            hasattr(token_logprob, "logprobs_by_token")
                            and token_logprob.logprobs_by_token
                        ):
                            token = list(token_logprob.logprobs_by_token.keys())[0]
                            full_tokens.append(token)

                    eval_logger.debug(f"Extracted full tokens: {full_tokens}")

                    if len(context_tokens) >= len(full_tokens):
                        result = (0.0, True)
                        results.append(result)
                        self.cache_hook.add_partial(
                            "loglikelihood", (context, continuation), result
                        )
                        continue

                    continuation_enc = full_tokens[len(context_tokens) :]

                response = self.client.inference.completion(
                    content=full_text,
                    model_id=self.model_id,
                    sampling_params=StrategyTopKSamplingStrategy(type="top_k", top_k=1),
                    logprobs=Logprobs(top_k=5),
                    extra_headers=self._get_headers(),
                )

                loglikelihood = 0.0
                is_greedy = True

                skip_tokens = (
                    len(context_enc) if self.tokenizer else len(context_tokens)
                )

                for i, token_logprobs in enumerate(response.logprobs):
                    if i < skip_tokens:
                        continue

                    if i >= len(response.logprobs):
                        break

                    logprobs_dict = token_logprobs.logprobs_by_token

                    actual_token = list(logprobs_dict.keys())[0]
                    actual_logprob = logprobs_dict[actual_token]

                    loglikelihood += actual_logprob

                result = (float(loglikelihood), is_greedy)
                results.append(result)
                self.cache_hook.add_partial(
                    "loglikelihood", (context, continuation), result
                )

            except Exception as e:
                eval_logger.error(f"Error in loglikelihood: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                # Return negative infinity and True on error
                result = (float("-inf"), True)
                results.append(result)
                # We don't cache errors

        return results

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Calculate the rolling log-likelihood of token sequences.

        Args:
            requests: List of requests containing token sequences.

        Returns:
            List of tuples containing log-likelihood and is_greedy flag.
        """
        self._check_model_logprobs_support()
        results = []
        for request in tqdm(requests):
            (text,) = request.args

            try:
                response = self.client.inference.completion(
                    content=text,
                    model_id=self.model_id,
                    sampling_params=StrategyTopKSamplingStrategy(type="top_k", top_k=1),
                    logprobs=Logprobs(top_k=1),
                    extra_headers=self._get_headers(),
                )

                eval_logger.debug(f"Rolling response type: {type(response)}")
                eval_logger.debug(
                    f"Rolling response logprobs type: {type(response.logprobs)}"
                )

                loglikelihood = 0.0

                for token_logprobs in response.logprobs:
                    logprobs_dict = token_logprobs.logprobs_by_token

                    actual_token = list(logprobs_dict.keys())[0]
                    actual_logprob = logprobs_dict[actual_token]

                    loglikelihood += actual_logprob

                result = (float(loglikelihood), True)
                results.append(result)
                self.cache_hook.add_partial("loglikelihood_rolling", text, result)

            except Exception as e:
                eval_logger.error(f"Error in loglikelihood_rolling: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                result = (float("-inf"), True)
                results.append(result)

        return results
