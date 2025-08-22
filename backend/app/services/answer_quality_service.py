from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass
from enum import Enum

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

from app.core.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """答案质量等级"""
    EXCELLENT = "excellent"  # 优秀 (>= 0.8)
    GOOD = "good"           # 良好 (>= 0.6)
    FAIR = "fair"           # 一般 (>= 0.4)
    POOR = "poor"           # 较差 (< 0.4)

@dataclass
class QualityScore:
    """质量评分"""
    overall_score: float
    relevance_score: float
    completeness_score: float
    coherence_score: float
    factuality_score: float
    level: QualityLevel
    details: Dict[str, Any]

class AnswerQualityService:
    """
    答案质量评估服务
    """
    
    def __init__(self):
        self.llm_service = None
        self.embedding_model = None
        
        # 初始化LLM服务
        try:
            self.llm_service = LLMService()
            logger.info("答案质量评估的LLM服务初始化成功")
        except Exception as e:
            logger.warning(f"LLM服务初始化失败: {e}")
        
        # 初始化嵌入模型
        self._initialize_embedding_model()
        
        # 质量评估权重
        self.weights = {
            'relevance': 0.35,      # 相关性
            'completeness': 0.25,   # 完整性
            'coherence': 0.20,      # 连贯性
            'factuality': 0.20      # 事实性
        }
        
        logger.info("答案质量评估服务初始化完成")
    
    def _initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            # 使用TF-IDF向量化器替代sentence_transformers
            self.embedding_model = TfidfVectorizer(
                max_features=1000,
                stop_words=None,
                ngram_range=(1, 2)
            )
            logger.info("成功初始化TF-IDF向量化器")
                
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            self.embedding_model = None
    
    def evaluate_answer_quality(self, 
                               query: str, 
                               answer: str, 
                               source_docs: List[Document],
                               evaluation_method: str = 'comprehensive') -> QualityScore:
        """评估答案质量"""
        try:
            logger.info(f"开始评估答案质量: {len(answer)} 字符")
            
            # 计算各项评分
            relevance_score = self._evaluate_relevance(query, answer, source_docs)
            completeness_score = self._evaluate_completeness(query, answer, source_docs)
            coherence_score = self._evaluate_coherence(answer)
            factuality_score = self._evaluate_factuality(answer, source_docs)
            
            # 计算综合评分
            overall_score = (
                relevance_score * self.weights['relevance'] +
                completeness_score * self.weights['completeness'] +
                coherence_score * self.weights['coherence'] +
                factuality_score * self.weights['factuality']
            )
            
            # 确定质量等级
            level = self._determine_quality_level(overall_score)
            
            # 构建详细信息
            details = {
                'query_length': len(query),
                'answer_length': len(answer),
                'source_docs_count': len(source_docs),
                'evaluation_method': evaluation_method,
                'weights': self.weights.copy()
            }
            
            quality_score = QualityScore(
                overall_score=overall_score,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                coherence_score=coherence_score,
                factuality_score=factuality_score,
                level=level,
                details=details
            )
            
            logger.info(f"答案质量评估完成: {level.value} (总分: {overall_score:.3f})")
            return quality_score
            
        except Exception as e:
            logger.error(f"答案质量评估失败: {e}")
            # 返回默认评分
            return QualityScore(
                overall_score=0.5,
                relevance_score=0.5,
                completeness_score=0.5,
                coherence_score=0.5,
                factuality_score=0.5,
                level=QualityLevel.FAIR,
                details={'error': str(e)}
            )
    
    def _evaluate_relevance(self, query: str, answer: str, source_docs: List[Document]) -> float:
        """评估相关性"""
        try:
            # 方法1: 语义相似度
            semantic_score = self._calculate_semantic_similarity(query, answer)
            
            # 方法2: 关键词匹配
            keyword_score = self._calculate_keyword_overlap(query, answer)
            
            # 方法3: 源文档相关性
            source_relevance = self._calculate_source_relevance(query, source_docs)
            
            # 综合相关性评分
            relevance_score = (
                semantic_score * 0.5 +
                keyword_score * 0.3 +
                source_relevance * 0.2
            )
            
            logger.debug(f"相关性评分: 语义{semantic_score:.3f}, 关键词{keyword_score:.3f}, 源文档{source_relevance:.3f} -> {relevance_score:.3f}")
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            logger.error(f"相关性评估失败: {e}")
            return 0.5
    
    def _evaluate_completeness(self, query: str, answer: str, source_docs: List[Document]) -> float:
        """评估完整性"""
        try:
            # 方法1: 答案长度评估
            length_score = self._evaluate_answer_length(answer)
            
            # 方法2: 信息覆盖度
            coverage_score = self._evaluate_information_coverage(query, answer, source_docs)
            
            # 方法3: 结构完整性
            structure_score = self._evaluate_answer_structure(answer)
            
            # 综合完整性评分
            completeness_score = (
                length_score * 0.3 +
                coverage_score * 0.5 +
                structure_score * 0.2
            )
            
            logger.debug(f"完整性评分: 长度{length_score:.3f}, 覆盖{coverage_score:.3f}, 结构{structure_score:.3f} -> {completeness_score:.3f}")
            return min(1.0, max(0.0, completeness_score))
            
        except Exception as e:
            logger.error(f"完整性评估失败: {e}")
            return 0.5
    
    def _evaluate_coherence(self, answer: str) -> float:
        """评估连贯性"""
        try:
            # 方法1: 句子连贯性
            sentence_coherence = self._evaluate_sentence_coherence(answer)
            
            # 方法2: 逻辑结构
            logical_structure = self._evaluate_logical_structure(answer)
            
            # 方法3: 语言流畅性
            fluency_score = self._evaluate_language_fluency(answer)
            
            # 综合连贯性评分
            coherence_score = (
                sentence_coherence * 0.4 +
                logical_structure * 0.3 +
                fluency_score * 0.3
            )
            
            logger.debug(f"连贯性评分: 句子{sentence_coherence:.3f}, 逻辑{logical_structure:.3f}, 流畅{fluency_score:.3f} -> {coherence_score:.3f}")
            return min(1.0, max(0.0, coherence_score))
            
        except Exception as e:
            logger.error(f"连贯性评估失败: {e}")
            return 0.5
    
    def _evaluate_factuality(self, answer: str, source_docs: List[Document]) -> float:
        """评估事实性"""
        try:
            # 方法1: 源文档支持度
            source_support = self._calculate_source_support(answer, source_docs)
            
            # 方法2: 事实陈述检测
            factual_statements = self._detect_factual_statements(answer)
            
            # 方法3: 不确定性表达
            uncertainty_handling = self._evaluate_uncertainty_handling(answer)
            
            # 综合事实性评分
            factuality_score = (
                source_support * 0.6 +
                factual_statements * 0.2 +
                uncertainty_handling * 0.2
            )
            
            logger.debug(f"事实性评分: 源支持{source_support:.3f}, 事实陈述{factual_statements:.3f}, 不确定性{uncertainty_handling:.3f} -> {factuality_score:.3f}")
            return min(1.0, max(0.0, factuality_score))
            
        except Exception as e:
            logger.error(f"事实性评估失败: {e}")
            return 0.5
    
    def _calculate_semantic_similarity(self, query: str, answer: str) -> float:
        """计算语义相似度"""
        try:
            if self.embedding_model:
                # 使用TF-IDF计算相似度
                texts = [query, answer]
                tfidf_matrix = self.embedding_model.fit_transform(texts)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            else:
                # 降级到简单的词汇重叠
                return self._calculate_keyword_overlap(query, answer)
                
        except Exception as e:
            logger.error(f"语义相似度计算失败: {e}")
            return 0.5
    
    def _calculate_keyword_overlap(self, query: str, answer: str) -> float:
        """计算关键词重叠度"""
        try:
            # 分词
            query_words = set(jieba.cut(query))
            answer_words = set(jieba.cut(answer))
            
            # 过滤停用词和标点
            query_words = {w for w in query_words if len(w) > 1 and not re.match(r'[\d\W]+', w)}
            answer_words = {w for w in answer_words if len(w) > 1 and not re.match(r'[\d\W]+', w)}
            
            if not query_words:
                return 0.0
            
            # 计算重叠度
            overlap = len(query_words & answer_words)
            overlap_ratio = overlap / len(query_words)
            
            return min(1.0, overlap_ratio * 2)  # 放大系数
            
        except Exception as e:
            logger.error(f"关键词重叠计算失败: {e}")
            return 0.5
    
    def _calculate_source_relevance(self, query: str, source_docs: List[Document]) -> float:
        """计算源文档相关性"""
        try:
            if not source_docs:
                return 0.0
            
            relevance_scores = []
            for doc in source_docs:
                # 计算查询与文档的相似度
                doc_relevance = self._calculate_semantic_similarity(query, doc.page_content)
                relevance_scores.append(doc_relevance)
            
            # 返回平均相关性
            return np.mean(relevance_scores) if relevance_scores else 0.0
            
        except Exception as e:
            logger.error(f"源文档相关性计算失败: {e}")
            return 0.5
    
    def _evaluate_answer_length(self, answer: str) -> float:
        """评估答案长度"""
        try:
            length = len(answer)
            
            # 理想长度范围: 50-500字符
            if length < 20:
                return 0.2  # 太短
            elif length < 50:
                return 0.6  # 较短
            elif length <= 500:
                return 1.0  # 理想
            elif length <= 1000:
                return 0.8  # 较长但可接受
            else:
                return 0.6  # 太长
                
        except Exception as e:
            logger.error(f"答案长度评估失败: {e}")
            return 0.5
    
    def _evaluate_information_coverage(self, query: str, answer: str, source_docs: List[Document]) -> float:
        """评估信息覆盖度"""
        try:
            if not source_docs:
                return 0.5
            
            # 提取查询中的关键信息点
            query_keywords = set(jieba.cut(query))
            query_keywords = {w for w in query_keywords if len(w) > 1}
            
            # 统计答案中覆盖的信息点
            answer_keywords = set(jieba.cut(answer))
            covered_keywords = query_keywords & answer_keywords
            
            if not query_keywords:
                return 0.5
            
            coverage_ratio = len(covered_keywords) / len(query_keywords)
            return min(1.0, coverage_ratio * 1.5)  # 放大系数
            
        except Exception as e:
            logger.error(f"信息覆盖度评估失败: {e}")
            return 0.5
    
    def _evaluate_answer_structure(self, answer: str) -> float:
        """评估答案结构"""
        try:
            # 检查结构化元素
            structure_score = 0.5  # 基础分
            
            # 有分段
            if '\n' in answer or '。' in answer:
                structure_score += 0.2
            
            # 有列表或编号
            if re.search(r'[1-9]\.|[一二三四五]、|\*|•', answer):
                structure_score += 0.2
            
            # 有总结性语句
            if any(keyword in answer for keyword in ['总之', '综上', '因此', '所以']):
                structure_score += 0.1
            
            return min(1.0, structure_score)
            
        except Exception as e:
            logger.error(f"答案结构评估失败: {e}")
            return 0.5
    
    def _evaluate_sentence_coherence(self, answer: str) -> float:
        """评估句子连贯性"""
        try:
            sentences = re.split(r'[。！？]', answer)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                return 0.8  # 单句默认连贯
            
            # 简单的连贯性检查
            coherence_score = 0.7  # 基础分
            
            # 检查连接词
            connectors = ['但是', '然而', '因此', '所以', '而且', '另外', '此外', '同时']
            has_connectors = any(conn in answer for conn in connectors)
            if has_connectors:
                coherence_score += 0.2
            
            # 检查代词使用
            pronouns = ['这', '那', '它', '其', '该']
            has_pronouns = any(pron in answer for pron in pronouns)
            if has_pronouns:
                coherence_score += 0.1
            
            return min(1.0, coherence_score)
            
        except Exception as e:
            logger.error(f"句子连贯性评估失败: {e}")
            return 0.5
    
    def _evaluate_logical_structure(self, answer: str) -> float:
        """评估逻辑结构"""
        try:
            # 检查逻辑结构指示词
            logical_indicators = {
                'sequence': ['首先', '其次', '然后', '最后', '第一', '第二'],
                'causality': ['因为', '由于', '导致', '造成', '结果'],
                'contrast': ['但是', '然而', '相反', '不过'],
                'summary': ['总之', '综上', '总的来说']
            }
            
            structure_score = 0.5  # 基础分
            
            for category, indicators in logical_indicators.items():
                if any(indicator in answer for indicator in indicators):
                    structure_score += 0.125  # 每类加0.125分
            
            return min(1.0, structure_score)
            
        except Exception as e:
            logger.error(f"逻辑结构评估失败: {e}")
            return 0.5
    
    def _evaluate_language_fluency(self, answer: str) -> float:
        """评估语言流畅性"""
        try:
            # 简单的流畅性检查
            fluency_score = 0.8  # 基础分
            
            # 检查重复词汇
            words = list(jieba.cut(answer))
            unique_words = set(words)
            if len(words) > 0:
                repetition_ratio = 1 - (len(unique_words) / len(words))
                if repetition_ratio > 0.3:  # 重复率过高
                    fluency_score -= 0.2
            
            # 检查句子长度变化
            sentences = re.split(r'[。！？]', answer)
            sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
            if sentence_lengths:
                length_variance = np.var(sentence_lengths)
                if length_variance > 100:  # 句子长度变化合理
                    fluency_score += 0.1
            
            return min(1.0, max(0.0, fluency_score))
            
        except Exception as e:
            logger.error(f"语言流畅性评估失败: {e}")
            return 0.5
    
    def _calculate_source_support(self, answer: str, source_docs: List[Document]) -> float:
        """计算源文档支持度"""
        try:
            if not source_docs:
                return 0.3  # 无源文档时的默认分数
            
            # 计算答案与源文档的相似度
            support_scores = []
            for doc in source_docs:
                similarity = self._calculate_semantic_similarity(answer, doc.page_content)
                support_scores.append(similarity)
            
            # 返回最高支持度
            return max(support_scores) if support_scores else 0.3
            
        except Exception as e:
            logger.error(f"源文档支持度计算失败: {e}")
            return 0.5
    
    def _detect_factual_statements(self, answer: str) -> float:
        """检测事实陈述"""
        try:
            # 检查确定性表达
            certain_expressions = ['是', '为', '有', '包括', '含有', '具有']
            uncertain_expressions = ['可能', '也许', '大概', '似乎', '据说']
            
            certain_count = sum(1 for expr in certain_expressions if expr in answer)
            uncertain_count = sum(1 for expr in uncertain_expressions if expr in answer)
            
            total_expressions = certain_count + uncertain_count
            if total_expressions == 0:
                return 0.5
            
            # 适度的确定性是好的
            certainty_ratio = certain_count / total_expressions
            if 0.3 <= certainty_ratio <= 0.8:
                return 0.8
            elif certainty_ratio > 0.8:
                return 0.6  # 过于确定
            else:
                return 0.7  # 过于不确定
                
        except Exception as e:
            logger.error(f"事实陈述检测失败: {e}")
            return 0.5
    
    def _evaluate_uncertainty_handling(self, answer: str) -> float:
        """评估不确定性处理"""
        try:
            # 检查不确定性表达
            uncertainty_phrases = [
                '无法确定', '不确定', '无法从文档中找到', '文档中没有提及',
                '需要更多信息', '可能需要', '建议咨询'
            ]
            
            has_uncertainty = any(phrase in answer for phrase in uncertainty_phrases)
            
            # 适当的不确定性表达是好的
            if has_uncertainty:
                return 0.8
            else:
                return 0.6  # 没有不确定性表达，可能过于绝对
                
        except Exception as e:
            logger.error(f"不确定性处理评估失败: {e}")
            return 0.5
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """确定质量等级"""
        if overall_score >= 0.8:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.6:
            return QualityLevel.GOOD
        elif overall_score >= 0.4:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def filter_by_quality(self, 
                         answers: List[Tuple[str, List[Document]]], 
                         queries: List[str],
                         min_quality: QualityLevel = QualityLevel.FAIR) -> List[Tuple[str, List[Document], QualityScore]]:
        """根据质量过滤答案"""
        try:
            filtered_results = []
            min_score = self._quality_level_to_score(min_quality)
            
            for i, (answer, source_docs) in enumerate(answers):
                query = queries[i] if i < len(queries) else ""
                
                # 评估答案质量
                quality_score = self.evaluate_answer_quality(query, answer, source_docs)
                
                # 过滤低质量答案
                if quality_score.overall_score >= min_score:
                    filtered_results.append((answer, source_docs, quality_score))
                else:
                    logger.info(f"过滤低质量答案: {quality_score.level.value} (分数: {quality_score.overall_score:.3f})")
            
            logger.info(f"质量过滤完成: {len(answers)} -> {len(filtered_results)} 个答案")
            return filtered_results
            
        except Exception as e:
            logger.error(f"质量过滤失败: {e}")
            return [(answer, docs, None) for answer, docs in answers]
    
    def _quality_level_to_score(self, level: QualityLevel) -> float:
        """质量等级转换为分数"""
        mapping = {
            QualityLevel.EXCELLENT: 0.8,
            QualityLevel.GOOD: 0.6,
            QualityLevel.FAIR: 0.4,
            QualityLevel.POOR: 0.0
        }
        return mapping.get(level, 0.4)
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """获取质量评估统计信息"""
        return {
            'embedding_model_available': self.embedding_model is not None,
            'llm_service_available': self.llm_service is not None,
            'evaluation_weights': self.weights.copy(),
            'quality_levels': [level.value for level in QualityLevel],
            'supported_methods': ['comprehensive', 'fast', 'detailed']
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新评估权重"""
        try:
            # 验证权重
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"权重总和不为1: {total_weight}，将进行归一化")
                # 归一化权重
                for key in new_weights:
                    new_weights[key] /= total_weight
            
            # 更新权重
            self.weights.update(new_weights)
            logger.info(f"更新评估权重: {self.weights}")
            
        except Exception as e:
            logger.error(f"更新权重失败: {e}")