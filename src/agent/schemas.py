from pydantic import BaseModel,Field





class QueryAnalysisResult(BaseModel):
    """查询分析结果"""
    optimized_prompt:str=Field(description="优化后的清晰、完整的、无歧义提示词")



class CorrelationAnalysisResult(BaseModel):
    """相关性分析结果"""
    rate:int=Field(description="相关性分析得分,分数越高越相关",ge=0,le=10)
























class SubTaskAnalysis(BaseModel):
    """子任务分析"""
    task_name:str=Field(description="子任务名")
    use_local_database:bool=Field(description="该任务是否需要用到用户本地的多模态数据库")



class VectorRetrievalRequest(BaseModel):
    """向量检索请求数据模型"""
    collection_name:str=Field(...,description="向量集的名字，确定使用哪个向量集进行检索。")
    query_text:str=Field(...,description="用于检索某个确定的向量集的查询语句。对于文档检索是文本查询，对于图片检索是描述性文本")
    top_k:int=Field(3,description="返回最相关的结果数量",ge=1,le=10)
    # similarity_threshold:float=Field(0.3,description="相似度阈值，低于此值的结果将被过滤",ge=0.0,le=1.0)






