from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from utils import get_empd_dataset, get_dataset, make_logdir
import logging

# logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tokenizer_class = OpenAIGPTTokenizer 
    tokenizer = tokenizer_class.from_pretrained("openai-gpt")
    logger.info("Start persona dataset") 
    # pers_dataset = get_dataset(tokenizer, "", "./dataset_cache")

    # logger.info("Start emp dataset")
    emp_dataset = get_empd_dataset(tokenizer, "", "./test")
    # print(emp_dataset.keys())
    i = 0
    # for u in pers_dataset["train"]:
    #     if i == 10:
    #         break
    #     else:
    #         i += 1
    #         print(u)
