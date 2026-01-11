import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.read_config import config


class Reranker:
    def __init__(self,model_path=config.get("model_path").get('bge_rerank')):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def rerank(self,query,passages):
        """
        重排序
        :param query:查询字符串
        :param passages:待排序的文档列表
        :return:
        """
        scores=[]
        for passage in passages:
            inputs=self.tokenizer(
                query,
                passage,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            # 移动到GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs=self.model(**inputs)
                score=outputs.logits[0].item()
                scores.append(score)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_passages = [passages[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        return sorted_passages,sorted_scores



if __name__ == '__main__':
    reranker=Reranker()
    passages=["机器学习是人工智能的一个分支。",
    "深度学习是机器学习的一种方法。",
    "人工智能涉及多个领域。"]

    results,scores=reranker.rerank("什么是机器学习？",passages)
    print(results)
    print(scores)