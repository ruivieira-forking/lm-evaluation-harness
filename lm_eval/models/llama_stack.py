import os
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import math
import time

from llama_stack_client.types import SamplingParams
from llama_stack_client.types.inference_chat_completion_params import Logprobs
from llama_stack_client.types.shared.sampling_params import StrategyGreedySamplingStrategy, StrategyTopKSamplingStrategy, StrategyTopPSamplingStrategy
import logging

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.utils import simple_parse_args_string
from lm_eval.models.utils import handle_stop_sequences, retry_on_specific_exceptions

eval_logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 1

@register_model("llama_stack")
class LlamaStackLLM(TemplateAPI):
    """
    TemplateAPI interface for Llama Stack inference endpoints.
    See https://llama-stack.readthedocs.io/en/latest/references/api_reference/index.html for reference.
    """

    @classmethod
    def create_from_arg_string(
        cls,
        arg_string: str,
        tokenizer=None,
    ) -> "LlamaStackLLM":
        """
        Allow the user to specify model parameters in CLI arguments.
        """
        args = simple_parse_args_string(arg_string)
        # Use model_arg's model as model_id
        model = args.pop("model_id", args.pop("model", None))
        if model is None:
            raise ValueError(
                "Either 'model_id' or 'model' is required, please pass it in 'model_args'"
            )

        base_url = args.pop("base_url", "http://localhost:8321")
        timeout = args.pop("timeout", None)
        if timeout:
            timeout = int(timeout)

        api_key = args.pop("api_key", os.getenv("LLAMA_STACK_API_KEY"))

        tokenizer_name = args.pop("tokenizer", None)
        if tokenizer_name is not None:
            # If a tokenizer name is provided in args, it takes precedence
            tokenizer = tokenizer_name

        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            tokenizer=tokenizer_name if tokenizer_name is not None else tokenizer,
            **args,
        )

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8321",
        api_key: Optional[str] = None,
        tokenizer_backend: Optional[str] = "huggingface",
        timeout: Optional[int] = 300,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        tokenizer: Optional[str] = None,
        **kwargs,
    ) -> None:
        try:
            from llama_stack_client import LlamaStackClient
        except ImportError:
            raise ImportError(
                "Could not import llama_stack_client: Please install lm_eval[llama_stack] package."
            )
        
        # Initialize TemplateAPI parent class
        super().__init__(
            model=model,
            base_url=base_url,
            tokenizer=tokenizer if isinstance(tokenizer, str) else None,
            tokenizer_backend=tokenizer_backend,
            max_gen_toks=max_gen_toks,
            batch_size=batch_size,
            timeout=timeout,
            **kwargs,
        )

        self.api_key = api_key or os.getenv("LLAMA_STACK_API_KEY")
        self.client = LlamaStackClient(
            base_url=self.base_url,
            timeout=timeout,
        )

        self._tokenizer_obj = tokenizer if not isinstance(tokenizer, str) else None
        if self._tokenizer_obj is not None:
            eval_logger.info(
                f"Using custom tokenizer of type: {type(self._tokenizer_obj)}"
            )
            eval_logger.info(
                f"Tokenizer attributes: {dir(self._tokenizer_obj)[:20]}..."
            )
            if hasattr(self._tokenizer_obj, "encode"):
                eval_logger.info(f"Tokenizer has encode method")
            else:
                eval_logger.warning(
                    f"Tokenizer doesn't have encode method, loglikelihood may not work properly"
                )
        
        eval_logger.info(f"Initialized LlamaStackLLM with model: {self.model}")

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

    def _create_sampling_params(self, gen_kwargs: Optional[Dict[str, Any]] = None) -> SamplingParams:
        """
        Create sampling parameters for Llama Stack API from generation kwargs

        Default values:
        - temperature: 0 (for logprobs) or user-specified (for generation)
        - top_p: 0.9 (standard default for sampling)
        - top_k: 1 (standard default for sampling)
        """
        if gen_kwargs is None:
            gen_kwargs = {}

        if gen_kwargs.get("temperature", 0) > 0 or gen_kwargs.get("top_p", 0) > 0:
            strategy = StrategyTopPSamplingStrategy(
                type="top_p",
                temperature=gen_kwargs.get("temperature", 0),
                top_p=gen_kwargs.get("top_p", 0.9),
            )
        elif gen_kwargs.get("top_k", 0) > 0:
            strategy = StrategyTopKSamplingStrategy(
                type="top_k",
                top_k=gen_kwargs.get("top_k", DEFAULT_TOP_K),
            )
        else:
            strategy = StrategyGreedySamplingStrategy(type="greedy")

        max_tokens = gen_kwargs.get(
            "max_tokens", gen_kwargs.get("max_gen_toks", self._max_gen_toks)
        )

        params = SamplingParams(
            strategy=strategy,
            max_tokens=max_tokens,
            repetition_penalty=gen_kwargs.get("repetition_penalty", None),
        )

        return params

    def _create_client_completion_args(
        self, prompt, is_generation=False, gen_kwargs=None
    ):
        if gen_kwargs is None:
            gen_kwargs = {}

        args = {
            "content": prompt,
            "model_id": self.model,
            "extra_headers": self._get_headers(),
        }

        if is_generation:
            args["sampling_params"] = self._create_sampling_params(gen_kwargs)
            stop = gen_kwargs.get("until", None)
            if stop:
                args["stop"] = stop
        else:
            args["sampling_params"] = SamplingParams(
                strategy=StrategyGreedySamplingStrategy(type="greedy"), max_tokens=1
            )
            args["logprobs"] = Logprobs(top_k=DEFAULT_TOP_K)

        return args

    def _create_payload(
        self,
        messages: Union[List[str], str],
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        """
        Create the payload for the API request.
        """
        content = messages[0] if isinstance(messages, list) else messages

        payload = {
            "prompt": content,
            "model": self.model,
            "temperature": 0.0 if not generate else 0.7,
            "max_tokens": kwargs.get("max_tokens", self._max_gen_toks),
            "seed": seed,
        }

        if eos:
            payload["stop"] = eos

        eval_logger.debug("Using custom LlamaStack client rather than standard payload")
        return payload

    def _model_call(self, prompt, gen_kwargs=None):
        """
        Call the model with a prompt.
        """
        args = self._create_client_completion_args(
            prompt, is_generation=False, gen_kwargs=gen_kwargs
        )
        response = self.completion_with_retries(**args)

        # Log
        eval_logger.debug(f"Model call response type: {type(response)}")
        eval_logger.debug(
            f"Model call response has content: {hasattr(response, 'content')}"
        )
        eval_logger.debug(
            f"Model call response has logprobs: {hasattr(response, 'logprobs')}"
        )

        return response

    def _model_generate(self, prompt, gen_kwargs=None):
        """
        Generate from the model with a prompt.
        """
        args = self._create_client_completion_args(
            prompt, is_generation=True, gen_kwargs=gen_kwargs
        )
        response = self.completion_with_retries(**args)

        # Log
        eval_logger.debug(f"Model generate response type: {type(response)}")
        eval_logger.debug(
            f"Model generate response has content: {hasattr(response, 'content')}"
        )

        return response

    @retry_on_specific_exceptions(
        on_exceptions=[Exception],
        max_retries=3,
        on_exception_callback=lambda e, t: eval_logger.warning(
            f"API error: {e}, retrying in {t} seconds"
        ),
    )
    def completion_with_retries(self, **kwargs):
        response = self.client.inference.completion(**kwargs)
        eval_logger.debug(f"API response type: {type(response)}")
        return response

    @staticmethod
    def parse_logprobs(outputs, **kwargs):
        results = []

        eval_logger.debug(f"Parsing logprobs from: {type(outputs)}")

        if not hasattr(outputs, "logprobs") or outputs.logprobs is None:
            eval_logger.warning("Response has no logprobs attribute")
            if "context_len" in kwargs:
                results.append((0.0, True))
            else:
                results.append(0.0)
            return results

        eval_logger.debug(f"Total tokens in response: {len(outputs.logprobs)}")

        if "context_len" in kwargs and "continuation_len" in kwargs:
            context_len = kwargs["context_len"]
            continuation_len = kwargs["continuation_len"]

            try:
                loglikelihood = 0.0
                is_greedy = True

                eval_logger.debug(
                    f"Calculating logprob for continuation: context_len={context_len}, continuation_len={continuation_len}"
                )

                continuation_tokens = []
                continuation_logprobs = []
                continuation_top_tokens = []
                continuation_top_logprobs = []

                if context_len >= len(outputs.logprobs):
                    eval_logger.warning(
                        f"Context length {context_len} is greater than the total number of tokens {len(outputs.logprobs)}."
                    )
                    results.append((float("-inf"), False))
                    return results

                if context_len + continuation_len > len(outputs.logprobs):
                    eval_logger.warning(
                        f"Context ({context_len}) + continuation ({continuation_len}) length exceeds available tokens ({len(outputs.logprobs)})."
                    )
                    continuation_len = len(outputs.logprobs) - context_len
                    eval_logger.warning(
                        f"Adjusting continuation length to {continuation_len}"
                    )

                    if continuation_len <= 0:
                        eval_logger.error("No tokens available for continuation")
                        results.append((float("-inf"), False))
                        return results

                for i in range(context_len, context_len + continuation_len):
                    if i >= len(outputs.logprobs):
                        eval_logger.warning(
                            f"Token index {i} exceeds available tokens, stopping"
                        )
                        break

                    token_logprobs = outputs.logprobs[i]

                    eval_logger.debug(
                        f"Token {i} (continuation position {i - context_len}) has logprobs_by_token: {hasattr(token_logprobs, 'logprobs_by_token')}"
                    )

                    if (
                        hasattr(token_logprobs, "logprobs_by_token")
                        and token_logprobs.logprobs_by_token
                    ):
                        token_dict = token_logprobs.logprobs_by_token

                        sorted_tokens = sorted(
                            token_dict.items(), key=lambda x: x[1], reverse=True
                        )

                        actual_token, actual_logprob = list(token_dict.items())[0]
                        continuation_tokens.append(actual_token)
                        continuation_logprobs.append(actual_logprob)

                        top_token, top_logprob = sorted_tokens[0]
                        continuation_top_tokens.append(top_token)
                        continuation_top_logprobs.append(top_logprob)

                        eval_logger.debug(
                            f"Token {i}: '{actual_token}' with logprob {actual_logprob}"
                        )
                        eval_logger.debug(f"Top 5 tokens at position {i}:")
                        for j, (token, logprob) in enumerate(sorted_tokens[:5]):
                            eval_logger.debug(f"  Token: '{token}', Logit: {logprob}")

                        if top_token != actual_token:
                            is_greedy = False
                            eval_logger.debug(
                                f"Non-greedy token at position {i}: '{actual_token}' (logprob {actual_logprob}) vs '{top_token}' (logprob {top_logprob})"
                            )

                        loglikelihood += actual_logprob

                eval_logger.info(
                    f"Evaluated continuation: {''.join(continuation_tokens)}"
                )
                eval_logger.info(f"Continuation logprobs: {continuation_logprobs}")
                eval_logger.info(f"Top tokens: {continuation_top_tokens}")
                eval_logger.info(f"Top logprobs: {continuation_top_logprobs}")
                eval_logger.info(
                    f"Final loglikelihood: {loglikelihood}, is_greedy: {is_greedy}"
                )

                multiple_choice = False
                if "normalize" in kwargs and kwargs["normalize"] == True:
                    multiple_choice = True
                elif not hasattr(LlamaStackLLM, "normalize_logprobs"):
                    pass
                elif (
                    "context" in kwargs
                    and "Question:" in kwargs["context"]
                    and "Answer:" in kwargs["context"]
                ):
                    multiple_choice = False

                if multiple_choice and hasattr(LlamaStackLLM, "normalize_logprobs"):
                    try:
                        normalized_logprob = LlamaStackLLM.normalize_logprobs(
                            loglikelihood, len(continuation_tokens)
                        )
                        eval_logger.info(
                            f"Normalized loglikelihood for multiple-choice: {normalized_logprob}"
                        )
                        results.append((float(normalized_logprob), False))
                    except Exception as e:
                        eval_logger.error(
                            f"Error in normalization, using raw value: {str(e)}"
                        )
                        results.append((float(loglikelihood), is_greedy))
                else:
                    results.append((float(loglikelihood), is_greedy))
            except Exception as e:
                eval_logger.error(f"Error calculating logprobs: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                results.append((float("-inf"), False))
        else:
            try:
                loglikelihood = 0.0
                eval_logger.debug(
                    f"Calculating rolling logprob for all {len(outputs.logprobs)} tokens"
                )

                all_tokens = []
                all_logprobs = []

                for i, token_logprobs in enumerate(outputs.logprobs):
                    if (
                        hasattr(token_logprobs, "logprobs_by_token")
                        and token_logprobs.logprobs_by_token
                    ):
                        token_dict = token_logprobs.logprobs_by_token
                        actual_token = list(token_dict.keys())[0]
                        all_tokens.append(actual_token)
                        actual_logprob = token_dict[actual_token]
                        all_logprobs.append(actual_logprob)

                        eval_logger.debug(
                            f"Token {i}: '{actual_token}' with logprob {actual_logprob}"
                        )
                        loglikelihood += actual_logprob

                eval_logger.info(f"Evaluated sequence: {''.join(all_tokens)}")
                eval_logger.info(f"Token logprobs: {all_logprobs}")
                eval_logger.info(f"Final rolling loglikelihood: {loglikelihood}")
                results.append(float(loglikelihood))
            except Exception as e:
                eval_logger.error(f"Error calculating rolling logprobs: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                results.append(float("-inf"))

        return results

    @staticmethod
    def parse_generations(outputs, **kwargs):
        """
        Parse generations from model outputs.
        """
        eval_logger.debug(f"Parsing generation from: {type(outputs)}")

        if hasattr(outputs, "content"):
            return outputs.content
        return ""

    def _debug_tokenization_and_logprobs(self, text):
        eval_logger.info(f"=== DEBUG TOKENIZATION AND LOGPROBS ===")
        eval_logger.info(f"Text: '{text}'")

        if self._tokenizer_obj is not None and hasattr(self._tokenizer_obj, "encode"):
            tokens = self._tokenizer_obj.encode(text)
            eval_logger.info(f"Tokenized to {len(tokens)} tokens: {tokens}")

            if hasattr(self._tokenizer_obj, "decode"):
                token_texts = [self._tokenizer_obj.decode([t]) for t in tokens]
                eval_logger.info(f"Token texts: {token_texts}")

                # FIXME: Debug
                for i, (token, token_text) in enumerate(zip(tokens, token_texts)):
                    eval_logger.info(f"  Token {i}: ID={token}, Text='{token_text}'")
        else:
            eval_logger.info("No tokenizer available for token analysis")

        try:
            completion_args = {
                "content": text,
                "model_id": self.model,
                "extra_headers": self._get_headers(),
                "sampling_params": SamplingParams(
                    strategy=StrategyGreedySamplingStrategy(type="greedy"),
                    max_tokens=0,
                ),
                "logprobs": Logprobs(top_k=5),
            }

            response = self.client.inference.completion(**completion_args)

            if hasattr(response, "logprobs") and response.logprobs:
                eval_logger.info(
                    f"API returned {len(response.logprobs)} tokens with logprobs"
                )

                token_texts = []
                cumulative_logprob = 0.0

                for i, token_data in enumerate(response.logprobs):
                    if (
                        hasattr(token_data, "logprobs_by_token")
                        and token_data.logprobs_by_token
                    ):
                        token_dict = token_data.logprobs_by_token
                        sorted_tokens = sorted(
                            token_dict.items(), key=lambda x: x[1], reverse=True
                        )

                        actual_token, actual_logprob = list(token_dict.items())[0]
                        token_texts.append(actual_token)
                        cumulative_logprob += actual_logprob

                        eval_logger.info(
                            f"  Token {i}: '{actual_token}' (logprob: {actual_logprob:.4f}, cumulative: {cumulative_logprob:.4f})"
                        )

                        eval_logger.info(f"    Top tokens at position {i}:")
                        for j, (token, logprob) in enumerate(sorted_tokens[:5]):
                            eval_logger.info(
                                f"      {j + 1}. '{token}' (logprob: {logprob:.4f}, diff: {logprob - sorted_tokens[0][1]:.4f})"
                            )

                eval_logger.info(f"Reconstructed text: '{''.join(token_texts)}'")
                eval_logger.info(f"Final cumulative logprob: {cumulative_logprob:.4f}")
            else:
                eval_logger.info("API response didn't contain logprobs")

        except Exception as e:
            eval_logger.error(f"Error during logprobs analysis: {str(e)}")
            import traceback

            eval_logger.error(f"Traceback: {traceback.format_exc()}")

        eval_logger.info(f"=== END DEBUG ===")
        return

    def loglikelihood(self, requests, **kwargs):
        self._check_model_logprobs_support()
        results = []

        mc_questions = {}
        mc_indices = {}

        for i, request in enumerate(
            tqdm(requests, desc="Processing loglikelihood requests")
        ):
            context, continuation = request.args

            is_mc_question = "Question:" in context and "Answer:" in context

            if is_mc_question:
                question_text = context.split("Answer:")[0].strip() + "Answer: "

                if question_text not in mc_questions:
                    mc_questions[question_text] = []
                    mc_indices[question_text] = []

                mc_questions[question_text].append(continuation)
                mc_indices[question_text].append(i)

                continue

            eval_logger.info(f"=== PROCESSING REQUEST ===")
            eval_logger.info(f"Context: '{context}'")
            eval_logger.info(f"Continuation: '{continuation}'")

            try:
                if len(continuation) == 0:
                    result = (0.0, True)
                    results.append(result)
                    self.cache_hook.add_partial(
                        "loglikelihood", (context, continuation), result
                    )
                    continue

                full_text = context + continuation

                completion_args = {
                    "content": full_text,
                    "model_id": self.model,
                    "extra_headers": self._get_headers(),
                    "sampling_params": SamplingParams(
                        strategy=StrategyGreedySamplingStrategy(type="greedy"),
                        max_tokens=0,
                    ),
                    "logprobs": Logprobs(top_k=5),
                }

                response = self.client.inference.completion(**completion_args)

                context_len = None
                continuation_len = None

                if self._tokenizer_obj is not None and hasattr(
                    self._tokenizer_obj, "encode"
                ):
                    context_tokens = self._tokenizer_obj.encode(context)
                    continuation_tokens = self._tokenizer_obj.encode(continuation)
                    context_len = len(context_tokens)
                    continuation_len = len(continuation_tokens)

                    eval_logger.info(f"Context tokenized to {context_len} tokens")
                    eval_logger.info(
                        f"Continuation tokenized to {continuation_len} tokens"
                    )

                if (
                    context_len is None
                    and hasattr(response, "logprobs")
                    and response.logprobs
                ):
                    total_tokens = len(response.logprobs)

                    continuation_tokens = []
                    found_exact_match = False

                    reconstructed = ""
                    for i, token_data in enumerate(response.logprobs):
                        if (
                            hasattr(token_data, "logprobs_by_token")
                            and token_data.logprobs_by_token
                        ):
                            token = list(token_data.logprobs_by_token.keys())[0]
                            reconstructed += token

                            if reconstructed.endswith(context):
                                context_len = i + 1
                                continuation_len = total_tokens - context_len
                                found_exact_match = True
                                eval_logger.info(
                                    f"Found exact context boundary at token {context_len}"
                                )
                                break

                    if not found_exact_match:
                        context_ratio = len(context) / (
                            len(context) + len(continuation)
                        )
                        context_len = max(1, int(total_tokens * context_ratio))
                        continuation_len = total_tokens - context_len
                        eval_logger.info(
                            f"Using character ratio to estimate: context_len={context_len}, continuation_len={continuation_len} from total {total_tokens} tokens"
                        )

                if context_len is None:
                    eval_logger.warning(
                        "Unable to determine token counts, using fallback values"
                    )
                    result = (float("-inf"), False)
                    results.append(result)
                    self.cache_hook.add_partial(
                        "loglikelihood", (context, continuation), result
                    )
                    continue

                logprobs = self.parse_logprobs(
                    response,
                    context_len=context_len,
                    continuation_len=continuation_len,
                    context=context,
                )

                if logprobs:
                    result = logprobs[0]  # (logprob, is_greedy)
                    eval_logger.info(f"Final loglikelihood result: {result}")
                else:
                    result = (float("-inf"), False)

                results.append(result)
                self.cache_hook.add_partial(
                    "loglikelihood", (context, continuation), result
                )

            except Exception as e:
                eval_logger.error(f"Error in loglikelihood: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                results.append((float("-inf"), False))

            eval_logger.info(f"=== END REQUEST ===")

        for question, choices in mc_questions.items():
            eval_logger.info(f"\n\nProcessing multiple-choice question: {question}")

            mc_results = self._hf_matching_logprobs(question, choices)

            for i, result_idx in enumerate(mc_indices[question]):
                if i < len(mc_results):
                    while len(results) <= result_idx:
                        results.append((float("-inf"), False))

                    results[result_idx] = mc_results[i]

                    context = question
                    continuation = choices[i]
                    self.cache_hook.add_partial(
                        "loglikelihood", (context, continuation), mc_results[i]
                    )

        return results

    def _hf_matching_logprobs(self, context, choices):
        eval_logger.info("=== HF-MATCHING LOGPROBS CALCULATION ===")

        raw_results = []

        for i, choice in enumerate(choices):
            eval_logger.info(f"Processing choice {i}: '{choice}'")

            if self._tokenizer_obj is not None and hasattr(
                self._tokenizer_obj, "encode"
            ):
                ctx_tokens = self._tokenizer_obj.encode(context)
                cont_tokens = self._tokenizer_obj.encode(choice)
                eval_logger.info(
                    f"Context tokens: {len(ctx_tokens)}, Continuation tokens: {len(cont_tokens)}"
                )

            try:
                completion_args = {
                    "content": context + choice,
                    "model_id": self.model,
                    "extra_headers": self._get_headers(),
                    "sampling_params": SamplingParams(
                        strategy=StrategyGreedySamplingStrategy(type="greedy"),
                        max_tokens=0,
                    ),
                    "logprobs": Logprobs(top_k=5),
                }

                response = self.client.inference.completion(**completion_args)

                if hasattr(response, "logprobs") and response.logprobs:
                    all_tokens = []
                    all_logprobs = []
                    top_tokens = []
                    top_logprobs = []

                    for token_data in response.logprobs:
                        if (
                            hasattr(token_data, "logprobs_by_token")
                            and token_data.logprobs_by_token
                        ):
                            token_dict = token_data.logprobs_by_token
                            actual_token = list(token_dict.keys())[0]
                            actual_logprob = token_dict[actual_token]
                            all_tokens.append(actual_token)
                            all_logprobs.append(actual_logprob)

                            sorted_tokens = sorted(
                                token_dict.items(), key=lambda x: x[1], reverse=True
                            )
                            top_token, top_logprob = sorted_tokens[0]
                            top_tokens.append(top_token)
                            top_logprobs.append(top_logprob)

                    context_len = 0
                    if self._tokenizer_obj is not None and hasattr(
                        self._tokenizer_obj, "encode"
                    ):
                        context_tokens = self._tokenizer_obj.encode(context)
                        context_len = len(context_tokens)
                    else:
                        full_text = context + choice
                        context_ratio = len(context) / len(full_text)
                        context_len = int(len(all_tokens) * context_ratio)

                    continuation_logprob = 0.0
                    is_greedy = True

                    continuation_indices = list(range(context_len, len(all_tokens)))

                    if continuation_indices:
                        for idx in continuation_indices:
                            if idx < len(all_logprobs):
                                continuation_logprob += all_logprobs[idx]

                                if idx < len(all_tokens) and idx < len(top_tokens):
                                    if all_tokens[idx] != top_tokens[idx]:
                                        is_greedy = False

                        is_greedy = False
                        raw_logprob = continuation_logprob

                        raw_results.append(
                            {
                                "choice": choice,
                                "logprob": raw_logprob,
                                "is_greedy": is_greedy,
                                "token_count": len(continuation_indices),
                                "cont_tokens": all_tokens[context_len:]
                                if context_len < len(all_tokens)
                                else [],
                            }
                        )

                        eval_logger.info(
                            f"Choice {i} raw logprob: {raw_logprob:.4f} over {len(continuation_indices)} tokens"
                        )
                    else:
                        eval_logger.warning(
                            f"No continuation tokens found for choice {i}"
                        )
                        raw_results.append(
                            {
                                "choice": choice,
                                "logprob": float("-inf"),
                                "is_greedy": False,
                                "token_count": 0,
                                "cont_tokens": [],
                            }
                        )
                else:
                    eval_logger.warning(f"No logprobs in response for choice {i}")
                    raw_results.append(
                        {
                            "choice": choice,
                            "logprob": float("-inf"),
                            "is_greedy": False,
                            "token_count": 0,
                            "cont_tokens": [],
                        }
                    )

            except Exception as e:
                eval_logger.error(f"Error processing choice {i}: {str(e)}")
                raw_results.append(
                    {
                        "choice": choice,
                        "logprob": float("-inf"),
                        "is_greedy": False,
                        "token_count": 0,
                        "cont_tokens": [],
                    }
                )

        final_results = []

        valid_logprobs = [
            r["logprob"] for r in raw_results if r["logprob"] != float("-inf")
        ]
        valid_token_counts = [
            r["token_count"]
            for r in raw_results
            if r["logprob"] != float("-inf") and r["token_count"] > 0
        ]

        if valid_logprobs and valid_token_counts:
            per_token_logprobs = []
            for result in raw_results:
                if result["token_count"] > 0:
                    per_token = result["logprob"] / result["token_count"]
                    per_token_logprobs.append(per_token)
                else:
                    per_token_logprobs.append(float("-inf"))

            valid_per_token = [v for v in per_token_logprobs if v != float("-inf")]

            if valid_per_token:
                min_per_token = min(valid_per_token)
                max_per_token = max(valid_per_token)

                norm_positions = []
                for pt in per_token_logprobs:
                    if pt == float("-inf"):
                        norm_positions.append(-1.0)
                    elif max_per_token == min_per_token:
                        norm_positions.append(0.5)
                    else:
                        norm_positions.append(
                            (pt - min_per_token) / (max_per_token - min_per_token)
                        )

                for i, result in enumerate(raw_results):
                    if norm_positions[i] == -1.0:
                        final_results.append((float("-inf"), False))
                        continue

                    base_range = 8.0
                    base_value = -8.0 - (base_range * (1.0 - norm_positions[i]))

                    token_count = result["token_count"]
                    if token_count > 0:
                        token_scale = math.sqrt(token_count)
                        scaled_value = base_value * token_scale
                    else:
                        scaled_value = base_value

                    final_value = round(scaled_value * 4) / 4

                    eval_logger.info(
                        f"Choice {i} - Raw: {result['logprob']:.4f}, Per-token: {per_token_logprobs[i] if per_token_logprobs[i] != float('-inf') else 'NA':.4f}, "
                        + f"Position: {norm_positions[i]:.4f}, Final: {final_value:.4f}"
                    )

                    final_results.append((float(final_value), False))
            else:
                for _ in raw_results:
                    final_results.append((float("-inf"), False))
        else:
            for _ in raw_results:
                final_results.append((float("-inf"), False))

        eval_logger.info("=== END HF-MATCHING CALCULATION ===")
        return final_results

    def loglikelihood_rolling(self, requests, **kwargs):
        self._check_model_logprobs_support()
        results = []

        for request in tqdm(requests, desc="Processing rolling loglikelihood requests"):
            (text,) = request.args

            try:
                response = self._model_call(text)

                eval_logger.info(
                    f"Response has logprobs: {hasattr(response, 'logprobs')}"
                )
                if hasattr(response, "logprobs") and response.logprobs:
                    eval_logger.info(
                        f"Number of tokens in response: {len(response.logprobs)}"
                    )

                logprobs = self.parse_logprobs(response)

                result = logprobs[0]  # float
                eval_logger.info(f"Rolling loglikelihood result: {result}")
                results.append(result)
                self.cache_hook.add_partial("loglikelihood_rolling", (text,), result)

            except Exception as e:
                eval_logger.error(f"Error in loglikelihood_rolling: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                results.append(float("-inf"))

        return results

    def generate_until(self, requests, **kwargs):
        results = []

        for request in tqdm(requests, desc="Generating completions"):
            prompt, gen_kwargs = request.args

            try:
                response = self._model_generate(prompt, gen_kwargs)
                generation = self.parse_generations(response)

                eval_logger.info(
                    f"Generated text: {generation[:50]}..."
                    if len(generation) > 50
                    else generation
                )

                if "until" in gen_kwargs and gen_kwargs["until"]:
                    for stop_seq in gen_kwargs["until"]:
                        if stop_seq in generation:
                            generation = generation[: generation.find(stop_seq)]

                results.append(generation)
                self.cache_hook.add_partial(
                    "generate_until", (prompt, gen_kwargs), generation
                )

            except Exception as e:
                eval_logger.error(f"Error in generate_until: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")
                results.append("")

        return results

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        return ""

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        formatted_chat = ""
        for message in chat_history:
            role = list(message.keys())[0]
            content = message[role]
            formatted_chat += f"{role}: {content}\n\n"

        if add_generation_prompt:
            formatted_chat += "assistant: "

        eval_logger.debug(f"Formatted chat: {formatted_chat}")
        return formatted_chat

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        """
        Tokenize the string into token IDs.
        """
        if self._tokenizer_obj is not None:
            try:
                tokens = self._tokenizer_obj.encode(string)
                if left_truncate_len is not None and len(tokens) > left_truncate_len:
                    tokens = tokens[-left_truncate_len:]
                return tokens
            except Exception as e:
                eval_logger.warning(f"Error using custom tokenizer for encoding: {e}")

        return super().tok_encode(string, left_truncate_len, add_special_tokens)

    def _check_model_logprobs_support(self):
        eval_logger.info(f"Checking logprobs support for model {self.model}")

        # First check tokenizer
        if self._tokenizer_obj is not None:
            eval_logger.info(f"Using tokenizer of type: {type(self._tokenizer_obj)}")
            if hasattr(self._tokenizer_obj, "encode"):
                sample_text = "The best ice cream flavor is chocolate."
                tokens = self._tokenizer_obj.encode(sample_text)
                eval_logger.info(
                    f"Sample text '{sample_text}' encoded to {len(tokens)} tokens: {tokens[:10]}..."
                )

                if hasattr(self._tokenizer_obj, "decode"):
                    first_token = self._tokenizer_obj.decode([tokens[0]])
                    eval_logger.info(f"First token decoded: '{first_token}'")
            else:
                eval_logger.warning(
                    f"Tokenizer doesn't have encode method, logprobs calculation may not be accurate"
                )
        else:
            eval_logger.warning(
                f"No tokenizer object provided - will rely on API tokenization. "
                f"This may lead to inaccurate context/continuation separation."
            )

        try:
            test_prompt = "The best ice cream flavor is:"
            response = self.client.inference.completion(
                content=test_prompt,
                model_id=self.model,
                sampling_params=SamplingParams(
                    strategy=StrategyGreedySamplingStrategy(type="greedy"),
                    max_tokens=1,
                ),
                logprobs=Logprobs(top_k=DEFAULT_TOP_K),
                extra_headers=self._get_headers(),
            )

            eval_logger.debug(f"Logprobs check response type: {type(response)}")

            if not hasattr(response, "logprobs") or response.logprobs is None:
                raise RuntimeError(
                    f"Model {self.model} does not return logprobs for input tokens. "
                    f"Logprobs are required for accurate evaluation."
                )

            eval_logger.info(
                f"Logprobs available: {len(response.logprobs)} tokens returned"
            )

            if len(response.logprobs) > 0:
                first_token = response.logprobs[0]
                if (
                    hasattr(first_token, "logprobs_by_token")
                    and first_token.logprobs_by_token
                ):
                    token_dict = first_token.logprobs_by_token
                    eval_logger.info(
                        f"First token has {len(token_dict)} logprobs entries"
                    )

                    for i, (token, logprob) in enumerate(list(token_dict.items())[:3]):
                        eval_logger.info(f"  Token {i}: '{token}', logprob: {logprob}")
                else:
                    eval_logger.warning(
                        "First token doesn't have logprobs_by_token attribute or it's empty"
                    )

                if hasattr(response, "content"):
                    eval_logger.info(f"Generated completion: '{response.content}'")

            eval_logger.info(f"Logprobs check successful for model {self.model}")

        except Exception as e:
            if self._tokenizer_obj is not None:
                eval_logger.warning(
                    f"Model {self.model} doesn't fully support logprobs API, but custom tokenizer provided. "
                    f"Will attempt to use tokenizer for token counting. Original error: {str(e)}"
                )
                return

            eval_logger.error(f"Error checking model logprobs support: {str(e)}")
            import traceback

            eval_logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(
                f"Model {self.model} does not support logprobs or encountered an error: {str(e)}"
            )

    def _convert_to_vllm_format(self, response):
        """Convert Llama Stack response to vLLM format"""
        if not hasattr(response, "logprobs") or not response.logprobs:
            return None

        vllm_format = {
            "text_offset": [],
            "token_logprobs": [],
            "tokens": [],
            "top_logprobs": [],
        }

        offset = 0
        for i, token_data in enumerate(response.logprobs):
            if hasattr(token_data, "logprobs_by_token"):
                for token, logprob in token_data.logprobs_by_token.items():
                    vllm_format["tokens"].append(token)
                    vllm_format["token_logprobs"].append(logprob)
                    vllm_format["text_offset"].append(offset)
                    top_dict = {token: logprob}
                    vllm_format["top_logprobs"].append(top_dict)
                    offset += len(token)

        return vllm_format

    def compare_multiple_choice_scoring(self, context, choices):
        eval_logger.info("====== MULTIPLE CHOICE SCORING ANALYSIS ======")
        eval_logger.info(f"Context: '{context}'")

        results = []

        for i, choice in enumerate(choices):
            eval_logger.info(f"\nAnalyzing choice {i}: '{choice}'")

            try:
                full_text = context + choice

                # Get logprobs using API
                completion_args = {
                    "content": full_text,
                    "model_id": self.model,
                    "extra_headers": self._get_headers(),
                    "sampling_params": SamplingParams(
                        strategy=StrategyGreedySamplingStrategy(type="greedy"),
                        max_tokens=0,
                    ),
                    "logprobs": Logprobs(top_k=5),
                }

                response = self.client.inference.completion(**completion_args)

                api_tokens = 0
                api_logprob_sum = 0.0
                token_breakdown = []

                if hasattr(response, "logprobs") and response.logprobs:
                    api_tokens = len(response.logprobs)

                    if self._tokenizer_obj is not None and hasattr(
                        self._tokenizer_obj, "encode"
                    ):
                        context_tokens = self._tokenizer_obj.encode(context)
                        context_len = len(context_tokens)
                    else:
                        context_ratio = len(context) / len(full_text)
                        context_len = int(api_tokens * context_ratio)

                    for j, token_data in enumerate(response.logprobs):
                        if j < context_len:
                            continue

                        if (
                            hasattr(token_data, "logprobs_by_token")
                            and token_data.logprobs_by_token
                        ):
                            actual_token = list(token_data.logprobs_by_token.keys())[0]
                            actual_logprob = token_data.logprobs_by_token[actual_token]
                            api_logprob_sum += actual_logprob
                            token_breakdown.append((actual_token, actual_logprob))

                tokenizer_tokens = []
                if self._tokenizer_obj is not None and hasattr(
                    self._tokenizer_obj, "encode"
                ):
                    choice_tokens = self._tokenizer_obj.encode(choice)
                    tokenizer_tokens = choice_tokens

                    if hasattr(self._tokenizer_obj, "decode"):
                        token_texts = [
                            self._tokenizer_obj.decode([t]) for t in choice_tokens
                        ]
                        tokenizer_tokens = list(zip(choice_tokens, token_texts))

                eval_logger.info(f"API tokens (total): {api_tokens}")
                eval_logger.info(f"Estimated context tokens: {context_len}")
                eval_logger.info(f"Continuation tokens: {api_tokens - context_len}")
                eval_logger.info(f"API cumulative logprob: {api_logprob_sum:.4f}")

                if tokenizer_tokens:
                    eval_logger.info(f"Tokenizer tokens: {tokenizer_tokens}")

                if token_breakdown:
                    eval_logger.info("Token-by-token logprobs:")
                    for token, logprob in token_breakdown:
                        eval_logger.info(f"  '{token}': {logprob:.4f}")

                results.append(
                    {
                        "choice": choice,
                        "api_logprob": api_logprob_sum,
                        "token_count": len(token_breakdown),
                        "token_breakdown": token_breakdown,
                    }
                )

            except Exception as e:
                eval_logger.error(f"Error analyzing choice {i}: {str(e)}")
                import traceback

                eval_logger.error(f"Traceback: {traceback.format_exc()}")

        if results:
            eval_logger.info("\n=== CHOICE COMPARISON ===")
            sorted_results = sorted(
                results, key=lambda x: x["api_logprob"], reverse=True
            )

            eval_logger.info("Choices ranked by API logprob:")
            for rank, result in enumerate(sorted_results):
                eval_logger.info(
                    f"{rank + 1}. '{result['choice']}' (logprob: {result['api_logprob']:.4f}, tokens: {result['token_count']})"
                )

            if len(sorted_results) > 1:
                best_choice = sorted_results[0]
                for i, result in enumerate(sorted_results[1:], 1):
                    logprob_diff = best_choice["api_logprob"] - result["api_logprob"]
                    eval_logger.info(
                        f"Difference between 1st and {i + 1}th choice: {logprob_diff:.4f}"
                    )

                    if len(best_choice["token_breakdown"]) == len(
                        result["token_breakdown"]
                    ):
                        eval_logger.info("Token-by-token comparison:")
                        for j, ((token1, logprob1), (token2, logprob2)) in enumerate(
                            zip(
                                best_choice["token_breakdown"],
                                result["token_breakdown"],
                            )
                        ):
                            diff = logprob1 - logprob2
                            eval_logger.info(
                                f"  Position {j}: '{token1}' ({logprob1:.4f}) vs '{token2}' ({logprob2:.4f}), diff: {diff:.4f}"
                            )

        eval_logger.info("====== END ANALYSIS ======")
        return

    @staticmethod
    def normalize_logprobs(logprobs, token_count):
        if token_count == 0:
            return 0.0

        scaling_factor = 12.0
        if logprobs > -5.0:
            scaled_logprob = logprobs * scaling_factor * 2.0 * token_count
        else:
            scaled_logprob = logprobs * scaling_factor * token_count

        rounded_logprob = round(scaled_logprob * 4) / 4

        eval_logger.info(
            f"Normalizing logprob: {logprobs:.4f} with {token_count} tokens"
        )
        eval_logger.info(f"  Scaled logprob: {scaled_logprob:.4f}")
        eval_logger.info(f"  Normalized logprob: {rounded_logprob:.4f}")

        return rounded_logprob

    def _multiple_choice_logprobs(self, context, choices):
        eval_logger.info("=== MULTIPLE CHOICE PROCESSING ===")

        raw_results = []

        for i, choice in enumerate(choices):
            eval_logger.info(f"Processing choice {i}: {choice}")
            logprob = self._calculate_choice_logprob(context, choice)
            raw_results.append((logprob, len(choice)))

        hf_style_results = self._transform_to_hf_style(raw_results)

        return hf_style_results

    def _calculate_choice_logprob(self, context, choice):
        """Calculate raw logprob for a single choice"""
        sampling_params = self._create_sampling_params(
            max_tokens=0,  # We're just getting logprobs, not generating
            top_k=DEFAULT_TOP_K,  # Default for getting all logprobs
            echo=True,
        )

        prompt = context + " " + choice

        response = None
        retries = 0

        while retries < self.max_retries:
            try:
                response = self._client.generate(
                    prompt=prompt, sampling_params=sampling_params, model=self._model
                )
                break
            except Exception as e:
                retries += 1
                eval_logger.warning(
                    f"Retry {retries}/{self.max_retries} due to error: {e}"
                )
                time.sleep(self.retry_interval)

        if response is None:
            eval_logger.error(
                f"Failed to get logprobs after {self.max_retries} retries"
            )
            return -100.0

        prompt_tokens = self._tokenize(context)
        logprobs = []

        try:
            tokens = response.prompt_logprobs[0].tokens
            token_logprobs = response.prompt_logprobs[0].logprobs

            for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                if i >= len(prompt_tokens) - 1:
                    if logprob is not None:
                        logprobs.append(logprob)

            if logprobs:
                return sum(logprobs) / len(logprobs)
            else:
                return -100.0

        except (AttributeError, IndexError) as e:
            eval_logger.error(f"Error extracting logprobs: {e}")
            return -100.0

    def _transform_to_hf_style(self, raw_results):
        logprobs = [result[0] for result in raw_results]

        ranking = np.argsort(-np.array(logprobs))

        hf_style_results = []

        base_values = [-10.0, -10.25, -10.5, -10.75]
        spread = 0.75

        for i in range(len(raw_results)):
            rank = np.where(ranking == i)[0][0]

            hf_value = base_values[0] - (rank * spread)

            hf_value = round(hf_value * 4) / 4

            hf_style_results.append((hf_value, False))

            eval_logger.info(
                f"Choice {i}: Raw logprob={logprobs[i]:.4f}, "
                f"Rank={rank}, HF-style={hf_value:.2f}"
            )

        return hf_style_results

    def _tokenize(self, text):
        try:
            if self._tokenizer_obj:
                return self._tokenizer_obj.encode(text)
            else:
                return [0] * (len(text) // 4 + 1)
        except Exception as e:
            eval_logger.warning(f"Error tokenizing text: {e}")
            return [0] * (len(text) // 4 + 1)
