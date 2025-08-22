from typing import Dict, List, Optional, Any
import logging
from langchain.chains.router import MultiRetrievalQAChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_retrieval_prompt import MULTI_RETRIEVAL_ROUTER_TEMPLATE
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.llms.base import BaseLLM
# from langchain.chat_models import ChatOpenAI  # removed
from sqlalchemy.orm import Session

from app.services.vector_service import VectorService
from app.services.document_service import DocumentService
from app.core.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class MultiRetrievalService:
    """
    多知识库检索服务
    基于LangChain的RouterChain实现根据问题自动选择知识库
    """
    
    def __init__(self):
        # 使用单例VectorService
        self.vector_service = VectorService.get_instance()
        self.document_service = DocumentService()
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()  # 创建LLM实例
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.router_chain = None
        self._initialize_retrievers()
        self._initialize_router_chain()
    
    def _create_llm(self):
        """
        创建LLM实例
        """
        return self.llm_service.get_llm()
    
    def _initialize_retrievers(self):
        """
        为每个文档类别初始化检索器
        """
        logger.info("🔧 开始初始化类别检索器...")
        # 尝试为所有文档类别创建检索器
        categories = ["manual", "procedure", "development", "record", "other"]
        
        for category in categories:
            try:
                logger.info(f"🔍 正在为类别 '{category}' 创建检索器...")
                # 为每个类别创建专门的检索器
                retriever = self.vector_service.get_category_retriever(category)
                if retriever:
                    self.retrievers[category] = retriever
                    logger.info(f"✅ 已为类别 '{category}' 初始化检索器，类型: {type(retriever).__name__}")
                else:
                    logger.warning(f"❌ 类别 '{category}' 的检索器为None，可能该类别没有文档")
            except Exception as e:
                logger.warning(f"❌ 为类别 '{category}' 初始化检索器失败: {e}")
        
        # 始终尝试创建一个通用检索器，作为兜底与合并来源
        try:
            logger.info("🔍 创建/刷新通用检索器作为兜底...")
            general_retriever = self.vector_service.get_retriever()
            if general_retriever:
                self.retrievers["general"] = general_retriever
                logger.info(f"✅ 已创建通用检索器，类型: {type(general_retriever).__name__}")
        except Exception as e:
            logger.error(f"❌ 创建通用检索器失败: {e}")
        
        logger.info(f"🎯 检索器初始化完成，可用类别: {list(self.retrievers.keys())}")
        for category, retriever in self.retrievers.items():
            logger.info(f"  - {category}: {type(retriever).__name__}")
    
    def _get_retriever_infos(self) -> List[Dict[str, str]]:
        """
        获取检索器信息，用于路由器链
        """
        retriever_infos = [
            {
                "name": "manual",
                "description": "包含质量手册、管理制度、规范标准等文档。适用于质量管理体系、管理制度、标准规范等问题。"
            },
            {
                "name": "procedure",
                "description": "包含操作程序、工作流程、作业指导书等文档。适用于具体操作步骤、工作流程、执行方法等问题。"
            },
            {
                "name": "development",
                "description": "包含开发程序、技术文档、开发指南等文档。适用于技术开发、编程、系统设计等问题。"
            },
            {
                "name": "record",
                "description": "包含记录表单、检查清单、报告模板等文档。适用于记录填写、表单使用、报告格式等问题。"
            },
            {
                "name": "other",
                "description": "包含商务/供应链/采购/合同/报价等通用或未归档的文档。适用于价格、供应商、采购条款等问题。"
            },
            {
                "name": "general",
                "description": "通用文档检索器，包含所有类型的文档。适用于无法明确分类的问题与兜底合并。"
            }
        ]
        
        # 只返回有对应检索器的类别
        return [info for info in retriever_infos if info["name"] in self.retrievers]
    
    def _initialize_router_chain(self):
        """
        初始化路由器链
        """
        if not self.retrievers:
            logger.warning("没有可用的检索器，无法初始化路由器链")
            return
        
        try:
            # 获取检索器信息
            retriever_infos = self._get_retriever_infos()
            
            # 创建目标链字典
            destination_chains = {}
            for info in retriever_infos:
                category = info["name"]
                if category in self.retrievers:
                    # 为每个类别创建RetrievalQA链
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=self.retrievers[category],
                        return_source_documents=True,
                        input_key="query"
                    )
                    destination_chains[category] = qa_chain
            
            # 创建路由器提示模板
            destinations = [f"{info['name']}: {info['description']}" for info in retriever_infos]
            destinations_str = "\n".join(destinations)
            
            router_template = MULTI_RETRIEVAL_ROUTER_TEMPLATE.format(
                destinations=destinations_str
            )
            
            router_prompt = PromptTemplate(
                template=router_template,
                input_variables=["input"],
                output_parser=RouterOutputParser()
            )
            
            # 创建路由器链
            router_chain = LLMRouterChain.from_llm(
                llm=self.llm,
                prompt=router_prompt
            )
            
            # 创建默认链（使用通用文档检索器）
            default_chain = None
            if "general" in destination_chains:
                default_chain = destination_chains["general"]
            elif destination_chains:
                default_chain = list(destination_chains.values())[0]
            
            # 创建多检索QA链
            if default_chain:
                self.router_chain = MultiRetrievalQAChain(
                    router_chain=router_chain,
                    destination_chains=destination_chains,
                    default_chain=default_chain,
                    verbose=True
                )
                logger.info(f"已初始化多知识库路由器，包含 {len(destination_chains)} 个知识库")
            else:
                logger.error("无法创建默认链，路由器初始化失败")
                
        except Exception as e:
            logger.error(f"初始化路由器链失败: {e}")
            self.router_chain = None
    
    def ask_question_with_routing(
        self,
        question: str,
        session_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        使用路由链回答问题
        """
        try:
            logger.info(f"🚀 开始多知识库路由问答: {question[:50]}...")
            
            # 使用智能路由选择最合适的知识库
            preferred_category = self._classify_question(question)
            
            # 如果首选类别的检索器不存在，降级到可用的检索器
            if preferred_category in self.retrievers:
                selected_category = preferred_category
            else:
                # 优先使用通用检索器作为降级
                if "general" in self.retrievers:
                    selected_category = "general"
                    logger.warning(f"首选类别 '{preferred_category}' 不可用，降级到 'general'")
                else:
                    # 降级到第一个可用的检索器
                    available_categories = list(self.retrievers.keys())
                    if available_categories:
                        selected_category = available_categories[0]
                        logger.warning(f"首选类别 '{preferred_category}' 不可用，降级到 '{selected_category}'")
                    else:
                        raise Exception("没有可用的检索器")
            
            if selected_category in self.retrievers:
                retriever = self.retrievers[selected_category]
                logger.info(f"🔍 MultiRetrievalService调用检索器: 类别={selected_category}, 检索器类型={type(retriever).__name__}")
                
                # 直接使用检索器获取相关文档
                docs_primary = []
                try:
                    if hasattr(retriever, "invoke"):
                        docs_primary = retriever.invoke(question)
                    elif hasattr(retriever, "get_relevant_documents"):
                        docs_primary = retriever.get_relevant_documents(question)
                except Exception as re:
                    logger.warning(f"主检索器检索失败，将尝试通用检索器兜底: {re}")
                    docs_primary = []
                logger.info(f"📋 主检索结果: {len(docs_primary)} 个文档")
                
                # 当结果稀疏时，使用通用检索器兜底并合并
                docs_merged = list(docs_primary)
                fallback_merge_used = False
                if (len(docs_primary) < 3) and ("general" in self.retrievers) and (selected_category != "general"):
                    try:
                        general_retriever = self.retrievers["general"]
                        docs_general = []
                        if hasattr(general_retriever, "invoke"):
                            docs_general = general_retriever.invoke(question)
                        elif hasattr(general_retriever, "get_relevant_documents"):
                            docs_general = general_retriever.get_relevant_documents(question)
                        logger.info(f"📎 通用兜底检索结果: {len(docs_general)} 个文档，将进行合并去重")
                        
                        # 合并去重（按 source + title + 前64字符）
                        seen = set()
                        def _sig(d):
                            meta = getattr(d, "metadata", {}) or {}
                            src = meta.get("source", "")
                            title = meta.get("title", "")
                            head = (getattr(d, "page_content", "") or "")[:64]
                            return f"{src}|{title}|{head}"
                        for d in docs_primary:
                            seen.add(_sig(d))
                        for d in docs_general:
                            sig = _sig(d)
                            if sig not in seen:
                                docs_merged.append(d)
                                seen.add(sig)
                        fallback_merge_used = True
                    except Exception as ge:
                        logger.warning(f"通用检索器兜底合并失败: {ge}")
                
                # 构建上下文（限制前3-5段）
                top_docs = docs_merged[:5]
                context = "\n\n".join([doc.page_content for doc in top_docs[:3]])
                
                # 使用LLM生成答案
                prompt = f"基于以下文档内容回答问题：\n\n文档内容：\n{context}\n\n问题：{question}\n\n答案："
                answer = self.llm.predict(prompt)
                
                # 提取源文档信息
                source_docs = [{
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知来源"),
                    "title": doc.metadata.get("title", "未知标题"),
                    "metadata": doc.metadata
                } for doc in top_docs]
                
                # 计算实际置信度
                result_for_confidence = {
                    "answer": answer,
                    "source_documents": source_docs
                }
                confidence = self._calculate_confidence(result_for_confidence)
                
                route_info = {
                    "selected_retriever": selected_category,
                    "preferred_retriever": preferred_category,
                    "routing_confidence": 0.9 if preferred_category == selected_category else 0.7,
                    "available_retrievers": list(self.retrievers.keys()),
                    "question_classification": "intelligent_routing",
                    "fallback_used": preferred_category != selected_category,
                    "fallback_merge_used": fallback_merge_used,
                    "primary_doc_count": len(docs_primary),
                    "merged_doc_count": len(docs_merged)
                }
                
                logger.info(f"🎯 智能路由选择: {selected_category}")
                logger.info(f"📊 计算置信度: {confidence:.2f}")
                
                return {
                    "answer": answer,
                    "source_documents": source_docs,
                    "route_info": route_info,
                    "confidence": confidence
                }
            else:
                raise ValueError(f"未找到类别 '{selected_category}' 的检索器")
            
        except Exception as e:
            logger.error(f"❌ 多知识库路由问答失败: {e}")
            # 降级到通用检索器
            try:
                logger.info("🔄 降级到通用文档检索器")
                general_retriever = self.retrievers.get("general")
                if general_retriever:
                    docs = general_retriever.get_relevant_documents(question)
                    context = "\n\n".join([doc.page_content for doc in docs[:3]])
                    
                    # 简单的基于上下文的回答
                    answer = f"基于通用文档库的搜索结果：\n\n{context}"
                    
                    source_docs = [{
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "未知来源"),
                        "title": doc.metadata.get("title", "未知标题"),
                        "metadata": doc.metadata
                    } for doc in docs]
                    
                    return {
                        "answer": answer,
                        "source_documents": source_docs,
                        "route_info": {
                            "selected_retriever": "general",
                            "routing_confidence": 0.5,
                            "available_retrievers": list(self.retrievers.keys()),
                            "fallback_reason": str(e)
                        },
                        "confidence": 0.5
                    }
                else:
                    raise e
            except Exception as fallback_error:
                logger.error(f"❌ 降级处理也失败: {fallback_error}")
                raise e
    
    def _extract_source_documents(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从结果中提取源文档
        """
        source_docs = []
        
        # 尝试从不同的结果格式中提取源文档
        if isinstance(result, dict):
            docs = result.get("source_documents", [])
            for doc in docs:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    source_docs.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "未知来源"),
                        "title": doc.metadata.get("title", "未知标题"),
                        "metadata": doc.metadata
                    })
        
        return source_docs
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        计算回答的置信度
        """
        # 简单的置信度计算逻辑
        # 可以根据实际需求进行优化
        source_count = len(result.get("source_documents", []))
        if source_count >= 3:
            return 0.9
        elif source_count >= 2:
            return 0.8
        elif source_count >= 1:
            return 0.7
        else:
            return 0.5
    
    def _classify_question(self, question: str) -> str:
        """
        根据问题内容智能分类，选择最合适的知识库
        """
        try:
            # 构建分类提示词
            classification_prompt = f"""
你是一个智能文档分类器。请根据用户问题的内容，判断应该从哪个知识库中检索信息。

可用的知识库类别：
- manual: 质量手册、管理制度、规范标准
- procedure: 操作程序、工作流程、作业指导书
- development: 开发程序、技术文档、开发指南
- record: 记录表单、检查清单、报告模板
- other: 商务/供应链/采购/合同/报价等通用或未归档的文档

用户问题：{question}

请仅返回最合适的类别名称（manual/procedure/development/record/other），不要包含其他内容。
"""
            
            # 使用LLM进行分类
            classification_result = self.llm.predict(classification_prompt).strip().lower()
            
            # 验证分类结果
            valid_categories = ["manual", "procedure", "development", "record", "other"]
            if classification_result in valid_categories:
                logger.info(f"🤖 问题分类结果: {question[:30]}... -> {classification_result}")
                return classification_result
            else:
                # 如果分类结果无效，使用关键词匹配作为备选方案
                return self._fallback_classification(question)
                
        except Exception as e:
            logger.error(f"问题分类失败: {e}")
            return self._fallback_classification(question)
    
    def _fallback_classification(self, question: str) -> str:
        """
        基于关键词的备选分类方法
        """
        question_lower = question.lower()
        
        # 开发相关关键词
        if any(keyword in question_lower for keyword in [
            "开发", "程序", "代码", "编程", "技术", "软件", "系统", "接口", "api", 
            "数据库", "算法", "架构", "设计", "测试", "调试", "部署", "版本"
        ]):
            return "development"
        
        # 程序相关关键词
        elif any(keyword in question_lower for keyword in [
            "程序", "流程", "步骤", "操作", "作业", "指导", "执行", "实施", 
            "如何", "怎么", "方法", "过程", "工序"
        ]):
            return "procedure"
        
        # 记录相关关键词
        elif any(keyword in question_lower for keyword in [
            "记录", "表单", "清单", "报告", "模板", "格式", "填写", "登记", 
            "统计", "汇总", "检查", "审核"
        ]):
            return "record"
        
        # 商务/报价/采购/合同/供应链 等归为 other
        elif any(keyword in question_lower for keyword in [
            "报价", "价格", "价目", "费用", "成本", "预算", "付款", "发票", "税率", "税额",
            "供应商", "采购", "招标", "投标", "比价", "合同", "协议", "条款", "商务", "商议",
            "供应链", "物料", "采购单", "询价", "订货", "供货", "对账", "结算"
        ]):
            return "other"
        
        # 手册/制度/规范/标准 归为 manual
        elif any(keyword in question_lower for keyword in [
            "手册", "制度", "规范", "标准", "方针", "政策", "章程", "守则", "要求", "原则"
        ]):
            return "manual"
        
        # 默认选择第一个可用的检索器
        else:
            available_categories = list(self.retrievers.keys())
            if available_categories:
                return available_categories[0]
            else:
                return "manual"  # 最后的备选方案
    
    def _log_routing_decision(
        self, 
        question: str, 
        route_info: Dict[str, Any], 
        db: Optional[Session] = None
    ):
        """
        记录路由决策日志
        """
        try:
            logger.info(
                f"路由决策 - 问题: {question[:50]}..., "
                f"选择的检索器: {route_info.get('selected_retriever')}"
            )
            
            # 这里可以添加更详细的日志记录到数据库
            # 例如保存到专门的路由日志表
            
        except Exception as e:
            logger.error(f"记录路由决策日志失败: {e}")
    
    def get_available_categories(self) -> List[str]:
        """
        获取可用的文档类别
        """
        return list(self.retrievers.keys())
    
    def refresh_retrievers(self):
        """
        刷新检索器（当向量数据更新时调用）
        """
        logger.info("开始刷新检索器...")
        self.retrievers.clear()
        self._initialize_retrievers()
        self._initialize_router_chain()
        logger.info("检索器刷新完成")