import { useState, useEffect, useRef } from 'react'
import {
  Card,
  Input,
  Button,
  List,
  Typography,
  Space,
  Tag,
  Rate,
  Divider,
  Avatar,
  Spin,
  Empty,
  message,
  Row,
  Col,
  Tooltip,
  Modal,
  Form,
  Select,
  Switch,
  Popover
} from 'antd'
import {
  SendOutlined,
  QuestionCircleOutlined,
  RobotOutlined,
  LikeOutlined,
  DislikeOutlined,
  CopyOutlined,
  HistoryOutlined,
  DeleteOutlined,
  ExportOutlined,
  DatabaseOutlined,
  SettingOutlined
} from '@ant-design/icons'
import { QAItem, QAHistory, QuestionRequest, AnswerResponse } from '../types/index'
import { qaApi } from '../services/qaApi'

const { TextArea } = Input
const { Title, Text, Paragraph } = Typography
const { Option } = Select

interface QAProps {
  user?: any
}

const QA = ({ user }: QAProps) => {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [qaHistory, setQaHistory] = useState<QAItem[]>([])
  const [historyVisible, setHistoryVisible] = useState(false)
  const [allHistory, setAllHistory] = useState<QAHistory[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [suggestionsLoading, setSuggestionsLoading] = useState(false)
  const [vectorLogs, setVectorLogs] = useState<Array<{
    timestamp: string
    level: string
    message: string
    details: any
  }>>([])  
  const [vectorLogsVisible, setVectorLogsVisible] = useState(false)
  const [vectorLogsLoading, setVectorLogsLoading] = useState(false)
  const [useMultiRetrieval, setUseMultiRetrieval] = useState(true)
  const [advancedRetrievalSettings, setAdvancedRetrievalSettings] = useState({
    useEnsemble: false, // 默认关闭，需要手动开启
    useParentDocument: false, // 默认关闭
    useMultiQuery: false, // 默认关闭，根据配置auto_prefer_multi_query_for_complex: false
    useContextualCompression: true, // 默认开启，通常有助于提升质量
    retrievalMode: 'auto' // auto, vector, hybrid, multi_query, ensemble
  })

  // 加载问题建议
  const loadSuggestions = async () => {
    setSuggestionsLoading(true)
    try {
      const suggestionList = await qaApi.getQuestionSuggestions()
      setSuggestions(suggestionList)
    } catch (error) {
      console.error('加载问题建议失败:', error)
      // 使用默认建议作为后备
      setSuggestions([
        '如何查找相关文档？',
        '质量控制的要求是什么？',
        '安全操作规范有哪些？',
        '文档管理流程是怎样的？',
        '异常处理步骤有哪些？'
      ])
    } finally {
      setSuggestionsLoading(false)
    }
  }

  // 组件加载时获取建议
  useEffect(() => {
    loadSuggestions()
  }, [])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [qaHistory])

  // 问答API调用
  const handleAskQuestion = async () => {
    if (!question.trim()) {
      message.warning('请输入问题')
      return
    }

    setLoading(true)
    
    // 添加用户问题到历史记录
    const userQuestion: QAItem = {
      id: Date.now().toString(),
      question: question.trim(),
      answer: '',
      confidence_score: 0,
      created_at: new Date().toISOString(),
      user_id: user?.id || 'anonymous',
      feedback: null,
      sources: [],
      processing_time: 0,
      type: 'question'
    }
    
    setQaHistory(prev => [...prev, userQuestion])
    const currentQuestion = question.trim()
    setQuestion('')

    try {
      // 获取向量搜索日志
      setVectorLogsLoading(true)
      try {
        const vectorLogsResponse = await qaApi.getVectorSearchLogs(currentQuestion)
        setVectorLogs(vectorLogsResponse.logs)
        setVectorLogsVisible(true)
      } catch (vectorError) {
        console.error('获取向量搜索日志失败:', vectorError)
      } finally {
        setVectorLogsLoading(false)
      }
      
      // 调用真实API
      const overrides: Record<string, any> = {}
      
      // 添加高级检索设置到 overrides
      if (advancedRetrievalSettings.retrievalMode !== 'auto') {
        overrides.mode = advancedRetrievalSettings.retrievalMode
      }
      
      if (advancedRetrievalSettings.useEnsemble) {
        overrides.use_ensemble = true
      }
      
      if (advancedRetrievalSettings.useParentDocument) {
        overrides.use_parent_document = true
      }
      
      if (advancedRetrievalSettings.useMultiQuery) {
        overrides.use_multi_query = true
      }
      
      if (advancedRetrievalSettings.useContextualCompression) {
        overrides.use_contextual_compression = true
      }
      
      const request: QuestionRequest = {
        question: currentQuestion,
        session_id: undefined, // 可以根据需要添加会话管理
        use_multi_retrieval: useMultiRetrieval,
        overrides: Object.keys(overrides).length > 0 ? overrides : undefined
      }
      
      const response: AnswerResponse = await qaApi.askQuestion(request)
      
      // 兼容后端返回字段差异
      const answerText =
        typeof response.answer === 'string'
          ? response.answer
          : (response.answer as any)?.text || ''

      const confidenceScore = response.confidence || 0.7

      const sourceDocs = (response as any).source_documents || (response as any).sources || []

      const aiAnswer: QAItem = {
        id: (response as any).qa_log_id?.toString() || '',
        question: currentQuestion,
        answer: answerText,
        confidence_score: confidenceScore,
        created_at: new Date().toISOString(),
        user_id: 'system',
        feedback: null,
        sources: sourceDocs.map((doc: any) => ({
          document_id: doc.document_id?.toString() || '',
          document_title: doc.metadata?.title || doc.metadata?.filename || '未知文档',
          chunk_id: `chunk_${doc.document_id}`,
          relevance_score: doc.similarity_score,
          content: doc.content || doc.chunk_content
        })),
        processing_time: response.processing_time || 0,
        type: 'answer'
      }
      
      setQaHistory(prev => [...prev, aiAnswer])
      
      // 重新加载问题建议，因为新问题已添加到历史记录
      loadSuggestions()
      
      // 重新加载历史记录，确保新的问答记录被保存到数据库
      loadHistory()
      
    } catch (error) {
      console.error('问答失败:', error)
      message.error('问答失败，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  // 处理反馈
  const handleFeedback = async (qaId: string, feedback: number) => {
    try {
      await qaApi.submitFeedback({
        qa_log_id: qaId,
        satisfaction_score: feedback,
        is_helpful: feedback >= 3 // 3分及以上认为有帮助
      })
      
      setQaHistory(prev => 
        prev.map(item => 
          item.id === qaId ? { ...item, feedback } : item
        )
      )
      message.success('反馈已提交')
    } catch (error) {
      console.error('反馈提交失败:', error)
      message.error('反馈提交失败，请稍后重试')
    }
  }

  // 复制答案
  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text)
    message.success('已复制到剪贴板')
  }

  // 加载历史记录
  const loadHistory = async () => {
    setHistoryLoading(true)
    try {
      const response = await qaApi.getQAHistory()
      
      // 处理后端返回的分页响应格式
      const qaLogs = Array.isArray(response) ? response : (response as any).items || []
      
      const history: QAHistory[] = qaLogs.map((log: any) => ({
        id: log.id.toString(),
        question: log.question,
        answer: log.answer,
        confidence_score: 0.85, // QALog中没有confidence_score，使用默认值
        created_at: log.created_at,
        user_id: log.user_id.toString(),
        feedback: log.feedback_score || null,
        processing_time: log.processing_time
      }))
      
      setAllHistory(history)
    } catch (error) {
      console.error('加载历史记录失败:', error)
      message.error('加载历史记录失败，请稍后重试')
    } finally {
      setHistoryLoading(false)
    }
  }

  // 删除历史记录
  const handleDeleteHistory = (id: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这条问答记录吗？',
      onOk: () => {
        setAllHistory(prev => prev.filter(item => item.id !== id))
        message.success('删除成功')
      }
    })
  }

  // 导出历史记录
  const handleExportHistory = () => {
    const data = allHistory.map(item => ({
      问题: item.question,
      回答: item.answer,
      置信度: `${Math.round(item.confidence_score * 100)}%`,
      创建时间: new Date(item.created_at).toLocaleString('zh-CN'),
      反馈评分: item.feedback || '无'
    }))
    
    const jsonStr = JSON.stringify(data, null, 2)
    const blob = new Blob([jsonStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `qa_history_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    message.success('导出成功')
  }

  // 格式化时间
  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN')
  }

  return (
    <div style={{ padding: '0 8px' }}>
      <div style={{ marginBottom: 16 }}>
        <Title level={3} style={{ marginBottom: 8 }}>智能问答</Title>
        <Text type="secondary" style={{ fontSize: '13px' }}>
          基于知识库的智能问答系统，为您提供准确的文档信息和专业解答。
        </Text>
      </div>

      <Row gutter={[12, 12]}>
        {/* 问答区域 */}
        <Col xs={24} lg={16}>
          <Card size="small">
            {/* 对话历史 */}
            <div style={{ height: 'calc(100vh - 480px)', minHeight: '300px', maxHeight: '500px', overflowY: 'auto', marginBottom: 16 }}>
              {qaHistory.length === 0 ? (
                <Empty 
                  description="开始您的第一个问题吧！"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                />
              ) : (
                <List
                  dataSource={qaHistory}
                  renderItem={(item) => (
                    <div style={{ marginBottom: 16 }}>
                      {item.type === 'question' ? (
                        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                          <div style={{ 
                            maxWidth: '80%', 
                            backgroundColor: '#1890ff', 
                            color: 'white',
                            padding: '8px 12px',
                            borderRadius: '12px',
                            marginLeft: '20%'
                          }}>
                            <Text style={{ color: 'white' }}>{item.question}</Text>
                          </div>
                          <Avatar 
                            icon={<QuestionCircleOutlined />} 
                            style={{ marginLeft: 8, backgroundColor: '#1890ff' }}
                          />
                        </div>
                      ) : (
                        <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                          <Avatar 
                            icon={<RobotOutlined />} 
                            style={{ marginRight: 8, backgroundColor: '#52c41a' }}
                          />
                          <div style={{ 
                            maxWidth: '80%', 
                            backgroundColor: '#f6f6f6',
                            padding: '12px',
                            borderRadius: '12px',
                            marginRight: '20%'
                          }}>
                            <Paragraph style={{ margin: 0, whiteSpace: 'pre-line' }}>
                              {item.answer}
                            </Paragraph>
                            
                            {item.sources && item.sources.length > 0 && (
                              <div style={{ marginTop: 8 }}>
                                <Text type="secondary" style={{ fontSize: '12px' }}>来源文档：</Text>
                                <div style={{ marginTop: 4 }}>
                                  {item.sources.map((source, index) => (
                                    <Tag key={index} style={{ fontSize: '11px' }}>
                                      {source.document_title}
                                    </Tag>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            <div style={{ 
                              marginTop: 8, 
                              display: 'flex', 
                              justifyContent: 'space-between',
                              alignItems: 'center'
                            }}>
                              <div>
                                <Tag color="blue">
                                  置信度: {Math.round(item.confidence_score * 100)}%
                                </Tag>
                                <Text type="secondary" style={{ fontSize: '11px' }}>
                                  {item.processing_time}s
                                </Text>
                              </div>
                              
                              <Space>
                                <Tooltip title="复制答案">
                                  <Button 
                                    type="text" 
                                    size="small"
                                    icon={<CopyOutlined />}
                                    onClick={() => handleCopy(item.answer)}
                                  />
                                </Tooltip>
                                
                                <Tooltip title="有用">
                                  <Button 
                                    type="text" 
                                    size="small"
                                    icon={<LikeOutlined />}
                                    style={{ color: item.feedback === 1 ? '#52c41a' : undefined }}
                                    onClick={() => handleFeedback(item.id, 1)}
                                  />
                                </Tooltip>
                                
                                <Tooltip title="无用">
                                  <Button 
                                    type="text" 
                                    size="small"
                                    icon={<DislikeOutlined />}
                                    style={{ color: item.feedback === 0 ? '#ff4d4f' : undefined }}
                                    onClick={() => handleFeedback(item.id, 0)}
                                  />
                                </Tooltip>
                              </Space>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                />
              )}
              
              {loading && (
                <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 16 }}>
                  <Avatar 
                    icon={<RobotOutlined />} 
                    style={{ marginRight: 8, backgroundColor: '#52c41a' }}
                  />
                  <div style={{ 
                    backgroundColor: '#f6f6f6',
                    padding: '12px',
                    borderRadius: '12px'
                  }}>
                    <Spin size="small" />
                    <Text style={{ marginLeft: 8 }}>正在思考中...</Text>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
            
            {/* 设置区域 - 紧凑布局 */}
            <Card size="small" style={{ marginBottom: 12 }}>
              <Row align="middle" justify="space-between">
                <Col>
                  <Space size="small">
                    <SettingOutlined style={{ color: '#1890ff' }} />
                    <Text strong style={{ fontSize: 13 }}>检索设置</Text>
                  </Space>
                </Col>
                <Col>
                  <Space size="small">
                    <Tag color={useMultiRetrieval ? 'blue' : 'default'} style={{ fontSize: 11 }}>
                      {useMultiRetrieval ? '智能路由' : '单一检索'}
                    </Tag>
                    <Switch
                      size="small"
                      checked={useMultiRetrieval}
                      onChange={setUseMultiRetrieval}
                    />
                    <Popover
                      title="高级检索设置"
                      trigger="click"
                      placement="topRight"
                      content={
                        <div style={{ width: 280 }}>
                          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                            <div>
                              <Text type="secondary" style={{ fontSize: 12 }}>检索模式:</Text>
                              <Select
                                value={advancedRetrievalSettings.retrievalMode}
                                onChange={(value) => setAdvancedRetrievalSettings(prev => ({ ...prev, retrievalMode: value }))}
                                style={{ width: '100%', marginTop: 4 }}
                                size="small"
                              >
                                <Option value="auto">自动</Option>
                                <Option value="vector">向量</Option>
                                <Option value="hybrid">混合</Option>
                                <Option value="multi_query">多查询</Option>
                                <Option value="ensemble">集成</Option>
                              </Select>
                            </div>
                            <Divider style={{ margin: '8px 0' }} />
                            <div>
                              <Text type="secondary" style={{ fontSize: 12 }}>检索增强选项:</Text>
                              <div style={{ marginTop: 8 }}>
                                <Row gutter={[0, 8]}>
                                  <Col span={24}>
                                    <Space size="small" align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
                                      <Text style={{ fontSize: 12 }}>集成检索</Text>
                                      <Switch
                                        size="small"
                                        checked={advancedRetrievalSettings.useEnsemble}
                                        onChange={(checked) => setAdvancedRetrievalSettings(prev => ({ ...prev, useEnsemble: checked }))}
                                      />
                                    </Space>
                                  </Col>
                                  <Col span={24}>
                                    <Space size="small" align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
                                      <Text style={{ fontSize: 12 }}>父文档检索</Text>
                                      <Switch
                                        size="small"
                                        checked={advancedRetrievalSettings.useParentDocument}
                                        onChange={(checked) => setAdvancedRetrievalSettings(prev => ({ ...prev, useParentDocument: checked }))}
                                      />
                                    </Space>
                                  </Col>
                                  <Col span={24}>
                                    <Space size="small" align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
                                      <Text style={{ fontSize: 12 }}>多查询检索</Text>
                                      <Switch
                                        size="small"
                                        checked={advancedRetrievalSettings.useMultiQuery}
                                        onChange={(checked) => setAdvancedRetrievalSettings(prev => ({ ...prev, useMultiQuery: checked }))}
                                      />
                                    </Space>
                                  </Col>
                                  <Col span={24}>
                                    <Space size="small" align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
                                      <Text style={{ fontSize: 12 }}>上下文压缩</Text>
                                      <Switch
                                        size="small"
                                        checked={advancedRetrievalSettings.useContextualCompression}
                                        onChange={(checked) => setAdvancedRetrievalSettings(prev => ({ ...prev, useContextualCompression: checked }))}
                                      />
                                    </Space>
                                  </Col>
                                </Row>
                              </div>
                            </div>
                          </Space>
                        </div>
                      }
                    >
                      <Button size="small" type="text" icon={<SettingOutlined />}>
                        高级
                      </Button>
                    </Popover>
                  </Space>
                </Col>
              </Row>
            </Card>
            
            {/* 输入区域 */}
            <div style={{ marginTop: 8 }}>
              <TextArea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="请输入您的问题..."
                autoSize={{ minRows: 2, maxRows: 3 }}
                onPressEnter={(e) => {
                  if (!e.shiftKey) {
                    e.preventDefault()
                    handleAskQuestion()
                  }
                }}
              />
              <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  按 Enter 发送，Shift + Enter 换行
                </Text>
                <Button 
                    type="primary" 
                    size="small"
                    icon={<SendOutlined />}
                    loading={loading}
                    onClick={handleAskQuestion}
                    disabled={!question.trim()}
                  >
                    发送
                  </Button>
              </div>
            </div>
          </Card>
        </Col>
        
        {/* 侧边栏 */}
        <Col xs={24} lg={8}>
          {/* 问题建议 */}
          <Card 
            title="问题建议" 
            size="small" 
            style={{ marginBottom: 12 }}
            extra={
              <Button 
                type="link" 
                size="small"
                loading={suggestionsLoading}
                onClick={loadSuggestions}
                style={{ fontSize: '11px' }}
              >
                刷新
              </Button>
            }
          >
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {suggestionsLoading ? (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <Spin size="small" />
                </div>
              ) : (
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  {suggestions.map((suggestion, index) => (
                    <Button 
                      key={index}
                      type="link" 
                      style={{ 
                        textAlign: 'left', 
                        height: 'auto', 
                        padding: '2px 0',
                        whiteSpace: 'normal',
                        fontSize: '12px'
                      }}
                      onClick={() => setQuestion(suggestion)}
                    >
                      {suggestion}
                    </Button>
                  ))}
                </Space>
              )}
            </div>
          </Card>
          
          {/* 历史记录 */}
          <Card 
            title="历史记录"
            size="small"
            style={{ marginBottom: 12 }}
            extra={
              <Space size="small">
                <Button 
                  type="link" 
                  icon={<ExportOutlined />}
                  size="small"
                  onClick={handleExportHistory}
                  style={{ fontSize: '11px' }}
                >
                  导出
                </Button>
                <Button 
                  type="link" 
                  icon={<HistoryOutlined />}
                  size="small"
                  onClick={() => {
                    setHistoryVisible(true)
                    loadHistory()
                  }}
                  style={{ fontSize: '11px' }}
                >
                  查看全部
                </Button>
              </Space>
            }
          >
            <Text type="secondary" style={{ fontSize: '12px' }}>点击查看全部历史记录</Text>
          </Card>
          
          {/* 向量日志 */}
          <Card 
            title="向量搜索日志"
            size="small"
            extra={
              <Button 
                type="link" 
                icon={<DatabaseOutlined />}
                size="small"
                onClick={() => setVectorLogsVisible(true)}
                disabled={vectorLogs.length === 0}
                style={{ fontSize: '11px' }}
              >
                查看日志
              </Button>
            }
          >
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {vectorLogs.length > 0 
                ? `已记录 ${vectorLogs.length} 条向量搜索日志` 
                : '暂无向量搜索日志'}
            </Text>
          </Card>
        </Col>
      </Row>
      
      {/* 历史记录模态框 */}
      <Modal
        title="问答历史记录"
        open={historyVisible}
        onCancel={() => setHistoryVisible(false)}
        footer={null}
        width={800}
      >
        <Spin spinning={historyLoading}>
          <List
            dataSource={allHistory}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Rate 
                    key="rate"
                    disabled 
                    defaultValue={item.feedback || 0} 
                    count={5}
                    style={{ fontSize: '14px' }}
                  />,
                  <Button 
                    key="delete"
                    type="link" 
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => handleDeleteHistory(item.id)}
                  >
                    删除
                  </Button>
                ]}
              >
                <List.Item.Meta
                  title={item.question}
                  description={
                    <div>
                      <Paragraph 
                        ellipsis={{ rows: 2, expandable: true }}
                        style={{ margin: '8px 0' }}
                      >
                        {item.answer}
                      </Paragraph>
                      <Space>
                        <Tag color="blue">
                          置信度: {Math.round(item.confidence_score * 100)}%
                        </Tag>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {formatTime(item.created_at)}
                        </Text>
                      </Space>
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Spin>
      </Modal>
      
      {/* 向量搜索日志模态框 */}
      <Modal
        title="向量搜索日志"
        open={vectorLogsVisible}
        onCancel={() => setVectorLogsVisible(false)}
        footer={null}
        width={900}
      >
        <Spin spinning={vectorLogsLoading}>
          <List
            dataSource={vectorLogs}
            renderItem={(log, index) => {
              const getLogColor = (level: string) => {
                switch (level) {
                  case 'ERROR': return 'red'
                  case 'WARNING': return 'orange'
                  case 'INFO': return 'blue'
                  case 'DEBUG': return 'green'
                  default: return 'default'
                }
              }
              
              return (
                <List.Item>
                  <List.Item.Meta
                    title={
                      <Space>
                        <Tag color={getLogColor(log.level)}>{log.level}</Tag>
                        <Text>{log.message}</Text>
                      </Space>
                    }
                    description={
                      <div>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(log.timestamp).toLocaleString()}
                        </Text>
                        {Object.keys(log.details).length > 0 && (
                          <div style={{ marginTop: 8 }}>
                            <Text strong>详细信息:</Text>
                            <pre style={{ 
                              backgroundColor: '#f5f5f5', 
                              padding: '8px', 
                              borderRadius: '4px',
                              fontSize: '12px',
                              marginTop: '4px',
                              whiteSpace: 'pre-wrap'
                            }}>
                              {JSON.stringify(log.details, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )
            }}
          />
        </Spin>
      </Modal>
    </div>
  )
}

export default QA