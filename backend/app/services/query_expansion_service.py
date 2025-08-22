from typing import List, Dict, Any, Optional, Set
import logging
import re
import jieba
import jieba.posseg as pseg
from collections import defaultdict

from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.core.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class QueryExpansionService:
    """
    查询扩展服务
    """
    
    def __init__(self):
        self.llm_service = None
        self.synonym_dict = {}
        self.domain_keywords = {}
        
        # 初始化LLM服务
        try:
            self.llm_service = LLMService()
            logger.info("查询扩展服务的LLM初始化成功")
        except Exception as e:
            logger.warning(f"LLM服务初始化失败: {e}")
        
        # 初始化同义词词典
        self._initialize_synonym_dict()
        
        # 初始化领域关键词
        self._initialize_domain_keywords()
        
        logger.info("查询扩展服务初始化完成")
    
    def _initialize_synonym_dict(self):
        """初始化同义词词典"""
        try:
            # 基础同义词映射
            self.synonym_dict = {
                # 开发相关
                '开发': ['研发', '编程', '程序设计', '软件开发', '代码开发'],
                '测试': ['检测', '验证', '校验', '质量保证', 'QA'],
                '部署': ['发布', '上线', '投产', '交付'],
                '需求': ['要求', '规格', '功能', '特性'],
                '设计': ['架构', '方案', '规划', '蓝图'],
                '文档': ['资料', '说明', '手册', '指南'],
                '流程': ['过程', '步骤', '程序', '工序'],
                '规范': ['标准', '准则', '规定', '要求'],
                '管理': ['治理', '控制', '监督', '运营'],
                '质量': ['品质', '水平', '标准', '等级'],
                
                # 技术相关
                '系统': ['平台', '框架', '架构', '环境'],
                '接口': ['API', '服务', '端点', '连接'],
                '数据库': ['DB', '存储', '数据源', '仓库'],
                '服务器': ['主机', '节点', '机器', '设备'],
                '网络': ['连接', '通信', '传输', '链路'],
                '安全': ['防护', '保护', '加密', '认证'],
                '性能': ['效率', '速度', '响应', '吞吐'],
                '监控': ['监测', '观察', '跟踪', '检查'],
                
                # 业务相关
                '用户': ['客户', '使用者', '终端用户', '操作员'],
                '产品': ['系统', '应用', '软件', '工具'],
                '项目': ['工程', '任务', '计划', '方案'],
                '团队': ['小组', '组织', '部门', '人员'],
                '会议': ['讨论', '沟通', '交流', '评审'],
                '报告': ['汇报', '总结', '分析', '说明'],
                '计划': ['规划', '安排', '方案', '策略'],
                '问题': ['故障', '缺陷', '错误', 'bug'],
            }
            
            logger.info(f"同义词词典初始化完成，包含 {len(self.synonym_dict)} 个词条")
            
        except Exception as e:
            logger.error(f"同义词词典初始化失败: {e}")
    
    def _initialize_domain_keywords(self):
        """初始化领域关键词"""
        try:
            self.domain_keywords = {
                'development': {
                    'keywords': ['开发', '编程', '代码', '程序', '软件', '系统', '架构', '设计', '测试', '部署'],
                    'related_terms': ['需求分析', '技术选型', '代码审查', '单元测试', '集成测试', '持续集成', '版本控制']
                },
                'procedure': {
                    'keywords': ['流程', '步骤', '程序', '操作', '执行', '处理', '审批', '规程'],
                    'related_terms': ['工作流', '业务流程', '操作指南', '执行标准', '审批流程', '处理流程']
                },
                'manual': {
                    'keywords': ['手册', '指南', '说明', '文档', '教程', '帮助', '使用', '操作'],
                    'related_terms': ['用户手册', '操作指南', '技术文档', '使用说明', '配置手册', '维护手册']
                },
                'policy': {
                    'keywords': ['政策', '制度', '规定', '条例', '办法', '准则', '标准'],
                    'related_terms': ['管理制度', '规章制度', '政策文件', '管理办法', '实施细则', '操作规范']
                },
                'guideline': {
                    'keywords': ['指导', '准则', '原则', '规范', '标准', '要求', '建议'],
                    'related_terms': ['技术规范', '开发规范', '编码规范', '设计准则', '最佳实践', '行业标准']
                }
            }
            
            logger.info(f"领域关键词初始化完成，包含 {len(self.domain_keywords)} 个领域")
            
        except Exception as e:
            logger.error(f"领域关键词初始化失败: {e}")
    
    def expand_query(self, 
                    query: str, 
                    expansion_type: str = 'comprehensive',
                    max_expansions: int = 5) -> List[str]:
        """扩展查询"""
        try:
            logger.info(f"开始查询扩展: {query[:50]}... (类型: {expansion_type})")
            
            expanded_queries = [query]  # 包含原始查询
            
            if expansion_type == 'synonym':
                # 仅使用同义词扩展
                expanded_queries.extend(self._synonym_expansion(query))
            elif expansion_type == 'llm':
                # 仅使用LLM扩展
                expanded_queries.extend(self._llm_expansion(query))
            elif expansion_type == 'comprehensive':
                # 综合扩展
                expanded_queries.extend(self._synonym_expansion(query))
                expanded_queries.extend(self._llm_expansion(query))
                expanded_queries.extend(self._domain_expansion(query))
            
            # 去重并限制数量
            unique_queries = list(dict.fromkeys(expanded_queries))  # 保持顺序去重
            result_queries = unique_queries[:max_expansions + 1]  # +1 包含原始查询
            
            logger.info(f"查询扩展完成: 原始1个 -> 扩展{len(result_queries)}个")
            return result_queries
            
        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [query]  # 返回原始查询
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """基于同义词的查询扩展"""
        try:
            expanded_queries = []
            
            # 分词
            words = list(jieba.cut(query))
            
            # 为每个词查找同义词
            for i, word in enumerate(words):
                if word in self.synonym_dict:
                    synonyms = self.synonym_dict[word]
                    for synonym in synonyms[:2]:  # 限制每个词的同义词数量
                        # 替换当前词生成新查询
                        new_words = words.copy()
                        new_words[i] = synonym
                        new_query = ''.join(new_words)
                        if new_query != query:
                            expanded_queries.append(new_query)
            
            logger.debug(f"同义词扩展生成 {len(expanded_queries)} 个查询")
            return expanded_queries[:3]  # 限制数量
            
        except Exception as e:
            logger.error(f"同义词扩展失败: {e}")
            return []
    
    def _llm_expansion(self, query: str) -> List[str]:
        """基于LLM的查询扩展"""
        try:
            if not self.llm_service:
                return []
            
            # 构建提示词
            prompt_template = PromptTemplate(
                input_variables=["query"],
                template="""
请为以下查询生成2-3个相关的查询变体，这些变体应该能够帮助检索到相同或相关的信息。
要求：
1. 保持查询的核心意图不变
2. 使用不同的表达方式或同义词
3. 可以适当扩展或细化查询内容
4. 每个变体用换行符分隔
5. 不要包含原始查询

原始查询：{query}

查询变体：
"""
            )
            
            # 生成扩展查询
            prompt = prompt_template.format(query=query)
            response = self.llm_service.generate_response(prompt)
            
            # 解析响应
            if response and response.strip():
                expanded_queries = [
                    line.strip() 
                    for line in response.strip().split('\n') 
                    if line.strip() and line.strip() != query
                ]
                
                logger.debug(f"LLM扩展生成 {len(expanded_queries)} 个查询")
                return expanded_queries[:3]  # 限制数量
            
            return []
            
        except Exception as e:
            logger.error(f"LLM查询扩展失败: {e}")
            return []
    
    def _domain_expansion(self, query: str) -> List[str]:
        """基于领域知识的查询扩展"""
        try:
            expanded_queries = []
            
            # 检测查询所属领域
            detected_domains = self._detect_query_domain(query)
            
            for domain in detected_domains:
                if domain in self.domain_keywords:
                    domain_info = self.domain_keywords[domain]
                    related_terms = domain_info.get('related_terms', [])
                    
                    # 添加相关术语到查询中
                    for term in related_terms[:2]:  # 限制数量
                        if term not in query:
                            expanded_query = f"{query} {term}"
                            expanded_queries.append(expanded_query)
            
            logger.debug(f"领域扩展生成 {len(expanded_queries)} 个查询")
            return expanded_queries[:2]  # 限制数量
            
        except Exception as e:
            logger.error(f"领域扩展失败: {e}")
            return []
    
    def _detect_query_domain(self, query: str) -> List[str]:
        """检测查询所属领域"""
        try:
            detected_domains = []
            
            for domain, domain_info in self.domain_keywords.items():
                keywords = domain_info.get('keywords', [])
                
                # 计算匹配分数
                matches = sum(1 for keyword in keywords if keyword in query)
                if matches > 0:
                    detected_domains.append((domain, matches))
            
            # 按匹配分数排序
            detected_domains.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前2个最匹配的领域
            return [domain for domain, _ in detected_domains[:2]]
            
        except Exception as e:
            logger.error(f"领域检测失败: {e}")
            return []
    
    def expand_query_with_context(self, 
                                 query: str, 
                                 context_docs: List[Document],
                                 max_expansions: int = 3) -> List[str]:
        """基于上下文文档的查询扩展"""
        try:
            logger.info(f"基于上下文扩展查询: {len(context_docs)} 个文档")
            
            # 提取上下文关键词
            context_keywords = self._extract_context_keywords(context_docs)
            
            # 生成扩展查询
            expanded_queries = [query]
            
            # 添加高频关键词到查询
            for keyword, freq in context_keywords[:max_expansions]:
                if keyword not in query and len(keyword) > 1:
                    expanded_query = f"{query} {keyword}"
                    expanded_queries.append(expanded_query)
            
            logger.info(f"上下文扩展完成: {len(expanded_queries)} 个查询")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"上下文查询扩展失败: {e}")
            return [query]
    
    def _extract_context_keywords(self, docs: List[Document]) -> List[tuple]:
        """从上下文文档中提取关键词"""
        try:
            keyword_freq = defaultdict(int)
            
            for doc in docs:
                # 分词并统计词频
                words = pseg.cut(doc.page_content)
                for word, flag in words:
                    # 过滤停用词和标点
                    if (len(word) > 1 and 
                        flag in ['n', 'v', 'a', 'nr', 'ns', 'nt', 'nz'] and
                        not re.match(r'[\d\W]+', word)):
                        keyword_freq[word] += 1
            
            # 按频率排序
            sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_keywords[:10]  # 返回前10个高频词
            
        except Exception as e:
            logger.error(f"提取上下文关键词失败: {e}")
            return []
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """使用同义词扩展查询"""
        return self._synonym_expansion(query)
    
    def expand_with_domain_knowledge(self, query: str) -> List[str]:
        """使用领域知识扩展查询"""
        return self._domain_expansion(query)
    
    def comprehensive_expand(self, query: str) -> List[str]:
        """综合扩展查询"""
        return self.expand_query(query, expansion_type='comprehensive')
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """获取扩展统计信息"""
        return {
            'synonym_dict_size': len(self.synonym_dict),
            'domain_count': len(self.domain_keywords),
            'llm_available': self.llm_service is not None,
            'supported_domains': list(self.domain_keywords.keys())
        }
    
    def add_custom_synonyms(self, word: str, synonyms: List[str]):
        """添加自定义同义词"""
        try:
            if word not in self.synonym_dict:
                self.synonym_dict[word] = []
            
            for synonym in synonyms:
                if synonym not in self.synonym_dict[word]:
                    self.synonym_dict[word].append(synonym)
            
            logger.info(f"添加自定义同义词: {word} -> {synonyms}")
            
        except Exception as e:
            logger.error(f"添加自定义同义词失败: {e}")
    
    def update_domain_keywords(self, domain: str, keywords: List[str], related_terms: List[str]):
        """更新领域关键词"""
        try:
            if domain not in self.domain_keywords:
                self.domain_keywords[domain] = {}
            
            self.domain_keywords[domain]['keywords'] = keywords
            self.domain_keywords[domain]['related_terms'] = related_terms
            
            logger.info(f"更新领域关键词: {domain}")
            
        except Exception as e:
            logger.error(f"更新领域关键词失败: {e}")