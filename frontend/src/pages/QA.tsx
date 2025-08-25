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
  Popover,
  Upload
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
  SettingOutlined,
  UploadOutlined,
  FileOutlined,
  SwapOutlined,
  ApiOutlined,
  FilterOutlined,
  PaperClipOutlined,
  DownOutlined
} from '@ant-design/icons'
import { QAItem, QAHistory, QuestionRequest, AnswerResponse } from '../types/index'
import { qaApi } from '../services/qaApi'
import KnowledgeBaseSelector from '../components/KnowledgeBaseSelector'

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
  const [advancedRetrievalSettings, setAdvancedRetrievalSettings] = useState({
    useEnsemble: false, // 默认关闭，需要手动开启
    useParentDocument: false, // 默认关闭
    useMultiQuery: false, // 默认关闭，根据配置auto_prefer_multi_query_for_complex: false
    useContextualCompression: true, // 默认开启，通常有助于提升质量
    retrievalMode: 'auto' // auto, vector, hybrid, multi_query, ensemble
  })
  
  // 模型切换相关状态
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [currentModel, setCurrentModel] = useState<string>('')
  const [modelLoading, setModelLoading] = useState(false)
  
  // Kimi文件上传相关状态
  const [kimiFiles, setKimiFiles] = useState<any[]>([])
  const [kimiFileLoading, setKimiFileLoading] = useState(false)
  
  // 知识库选择器相关状态
  const [selectedKbIds, setSelectedKbIds] = useState<string[]>([])
  const [showKbSelector, setShowKbSelector] = useState(false)

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

  // 加载可用模型
  const loadAvailableModels = async () => {
    try {
      const models = await qaApi.getAvailableModels()
      setAvailableModels(models)
    } catch (error) {
      console.error('加载可用模型失败:', error)
      message.error('加载可用模型失败')
    }
  }

  // 加载当前模型
  const loadCurrentModel = async () => {
    try {
      const model = await qaApi.getCurrentModel()
      setCurrentModel(model)
      // 如果是Kimi模型，加载文件列表
      if (model.startsWith('kimi')) {
        loadKimiFiles()
      }
    } catch (error) {
      console.error('加载当前模型失败:', error)
      message.error('加载当前模型失败')
    }
  }

  // 切换模型
  const handleSwitchModel = async (modelName: string) => {
    setModelLoading(true)
    try {
      await qaApi.switchModel(modelName)
      setCurrentModel(modelName)
      message.success(`已切换到 ${modelName} 模型`)
      
      // 如果切换到Kimi模型，加载文件列表
      if (modelName.startsWith('kimi')) {
        loadKimiFiles()
      } else {
        setKimiFiles([])
      }
    } catch (error) {
      console.error('切换模型失败:', error)
      message.error('切换模型失败，请稍后重试')
    } finally {
      setModelLoading(false)
    }
  }

  // 加载Kimi文件列表
  const loadKimiFiles = async () => {
    setKimiFileLoading(true)
    try {
      const response = await qaApi.getKimiFiles()
      // 处理返回的数据结构，确保数据有效性
      const files = response.files || response || []
      // 过滤掉无效的文件数据
      const validFiles = files.filter((file: any) => 
        file && 
        (file.id || file.file_id) && 
        (file.name || file.filename)
      )
      setKimiFiles(validFiles)
      console.log('加载Kimi文件列表成功:', validFiles)
    } catch (error) {
      console.error('加载Kimi文件列表失败:', error)
      // 发生错误时清空文件列表，不显示错误消息（避免干扰用户）
      setKimiFiles([])
    } finally {
      setKimiFileLoading(false)
    }
  }

  // 处理Kimi文件上传
  const handleKimiFileUpload = async (file: File) => {
    setKimiFileLoading(true)
    try {
      const result = await qaApi.uploadFileToKimi(file)
      message.success(`文件 ${file.name} 上传成功`)
      loadKimiFiles() // 重新加载文件列表
      return result
    } catch (error) {
      console.error('文件上传失败:', error)
      message.error('文件上传失败，请稍后重试')
      throw error
    } finally {
      setKimiFileLoading(false)
    }
  }

  // 删除Kimi文件
  const handleDeleteKimiFile = async (fileId: string) => {
    try {
      await qaApi.deleteKimiFile(fileId)
      message.success('文件删除成功')
      loadKimiFiles() // 重新加载文件列表
    } catch (error) {
      console.error('文件删除失败:', error)
      message.error('文件删除失败，请稍后重试')
    }
  }

  // 组件加载时获取建议和模型信息
  useEffect(() => {
    loadSuggestions()
    loadAvailableModels()
    loadCurrentModel()
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
        active_kb_ids: selectedKbIds.length > 0 ? selectedKbIds : undefined,
        overrides: Object.keys(overrides).length > 0 ? overrides : undefined,
        // 如果使用Kimi模型且有上传文件，添加文件信息
        ...(currentModel?.includes('kimi') && kimiFiles.length > 0 && {
          kimi_files: kimiFiles.map(file => file.id)
        })
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
          document_title: doc.metadata?.title || doc.metadata?.filename || doc.title || '未知文档',
          chunk_id: `chunk_${doc.document_id}`,
          relevance_score: doc.similarity_score,
          content: doc.content || doc.chunk_content,
          kb_id: doc.kb_id,
          kb_name: doc.kb_name,
          page_number: doc.page_number
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
                                  {(() => {
                                    // 按知识库分组来源文档
                                    const groupedSources = item.sources.reduce((groups: Record<string, typeof item.sources>, source) => {
                                      const kbKey = source.kb_name || '未分类知识库'
                                      if (!groups[kbKey]) {
                                        groups[kbKey] = []
                                      }
                                      groups[kbKey].push(source)
                                      return groups
                                    }, {})
                                    
                                    return Object.entries(groupedSources).map(([kbName, sources]) => (
                                      <div key={kbName} style={{ marginBottom: 4 }}>
                                        {Object.keys(groupedSources).length > 1 && (
                                          <div style={{ marginBottom: 2 }}>
                                            <Tag color="geekblue" style={{ fontSize: '10px', marginBottom: 2 }}>
                                              {kbName}
                                            </Tag>
                                          </div>
                                        )}
                                        <div>
                                          {sources.map((source, index) => (
                                            <Tag 
                                              key={`${kbName}-${index}`} 
                                              style={{ 
                                                fontSize: '11px',
                                                marginBottom: 2,
                                                backgroundColor: Object.keys(groupedSources).length > 1 ? '#f0f0f0' : undefined
                                              }}
                                            >
                                              {source.document_title}
                                              {source.page_number && (
                                                <span style={{ color: '#666', marginLeft: 4 }}>
                                                  (第{source.page_number}页)
                                                </span>
                                              )}
                                            </Tag>
                                          ))}
                                        </div>
                                      </div>
                                    ))
                                  })()
                                  }
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
            
            {/* Icon工具栏 - 紧凑设计 */}
            <div style={{ 
              marginBottom: 16, 
              padding: '8px 12px', 
              backgroundColor: '#fafafa', 
              borderRadius: '8px',
              border: '1px solid #f0f0f0'
            }}>
              <Row align="middle" justify="space-between">
                <Col>
                  <Space size="large">
                    {/* 模型选择 */}
                    <Popover
                      title="选择模型"
                      trigger="click"
                      placement="bottomLeft"
                      content={
                        <div style={{ width: 200 }}>
                          <Select
                            value={currentModel}
                            onChange={handleSwitchModel}
                            loading={modelLoading}
                            style={{ width: '100%' }}
                            size="small"
                            placeholder="选择模型"
                          >
                            {availableModels.map(model => (
                              <Option key={model} value={model}>
                                {model}
                              </Option>
                            ))}
                          </Select>
                          <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                             当前: <Tag color="blue" style={{ fontSize: 11 }}>{currentModel}</Tag>
                           </div>
                        </div>
                      }
                    >
                      <Tooltip title="模型选择">
                        <Button 
                          type="text" 
                          icon={<ApiOutlined />} 
                          size="small"
                          style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            color: '#1890ff',
                            fontWeight: 500
                          }}
                        >
                          {currentModel}
                          <DownOutlined style={{ fontSize: 10, marginLeft: 4 }} />
                        </Button>
                      </Tooltip>
                    </Popover>

                    {/* 检索设置 */}
                    <Popover
                      title="检索设置"
                      trigger="click"
                      placement="bottom"
                      content={
                        <div style={{ width: 300 }}>
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
                              <Text type="secondary" style={{ fontSize: 12 }}>高级检索选项:</Text>
                              <Row gutter={[8, 8]} style={{ marginTop: 4 }}>
                                <Col span={24}>
                                  <Space size="small" align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
                                    <Text style={{ fontSize: 12 }}>集成检索(Ensemble)</Text>
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
                          </Space>
                        </div>
                      }
                    >
                      <Tooltip title="检索设置">
                        <Button 
                          type="text" 
                          icon={<FilterOutlined />} 
                          size="small"
                        >
                          检索设置
                        </Button>
                      </Tooltip>
                    </Popover>

                    {/* 知识库选择器 */}
                    <Popover
                      title="知识库选择"
                      trigger="click"
                      placement="bottomRight"
                      content={
                        <div style={{ width: 320 }}>
                          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                            <div>
                              <Text type="secondary" style={{ fontSize: 12 }}>选择要检索的知识库:</Text>
                              <div style={{ marginTop: 8 }}>
                                <KnowledgeBaseSelector
                                  selectedKbIds={selectedKbIds}
                                  onSelectionChange={setSelectedKbIds}
                                  placeholder="选择知识库（留空表示全部）"
                                  maxCount={5}
                                  size="small"
                                  showSearch={true}
                                  allowClear={true}
                                  style={{ width: '100%' }}
                                />
                              </div>
                            </div>
                            
                            {selectedKbIds.length > 0 && (
                              <div>
                                <Text type="secondary" style={{ fontSize: 12 }}>已选择 {selectedKbIds.length} 个知识库</Text>
                                <div style={{ marginTop: 4 }}>
                                  <Text style={{ fontSize: 11, color: '#666' }}>
                                    问答将仅在选定的知识库中检索相关内容
                                  </Text>
                                </div>
                              </div>
                            )}
                            
                            {selectedKbIds.length === 0 && (
                              <div>
                                <Text style={{ fontSize: 11, color: '#666' }}>
                                  未选择知识库时，将在所有可用知识库中检索
                                </Text>
                              </div>
                            )}
                          </Space>
                        </div>
                      }
                    >
                      <Tooltip title="知识库选择">
                        <Button 
                          type="text" 
                          icon={<DatabaseOutlined />} 
                          size="small"
                          style={{ color: selectedKbIds.length > 0 ? '#52c41a' : '#8c8c8c' }}
                        >
                          {selectedKbIds.length > 0 ? `已选${selectedKbIds.length}个` : '知识库'}
                        </Button>
                      </Tooltip>
                    </Popover>

                    {/* 附件管理 - 仅Kimi模型显示 */}
                    {currentModel.startsWith('kimi') && (
                      <Popover
                        title="附件管理"
                        trigger="click"
                        placement="bottomRight"
                        content={
                          <div style={{ width: 280 }}>
                            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                              <div>
                                <Upload
                                  beforeUpload={(file) => {
                                    handleKimiFileUpload(file)
                                    return false
                                  }}
                                  showUploadList={false}
                                  accept=".txt,.pdf,.doc,.docx,.md"
                                >
                                  <Button 
                                    size="small" 
                                    icon={<UploadOutlined />}
                                    loading={kimiFileLoading}
                                    block
                                  >
                                    上传文件
                                  </Button>
                                </Upload>
                              </div>
                              
                              {kimiFiles.length > 0 && (
                                <div>
                                  <Text type="secondary" style={{ fontSize: 12 }}>已上传文件:</Text>
                                  <div style={{ marginTop: 8, maxHeight: 120, overflowY: 'auto' }}>
                                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                                      {kimiFiles.map((file: any) => (
                                        <div key={file.id} style={{ 
                                          display: 'flex', 
                                          justifyContent: 'space-between', 
                                          alignItems: 'center',
                                          padding: '4px 8px',
                                          backgroundColor: '#f8f9fa',
                                          borderRadius: '4px'
                                        }}>
                                          <Space size="small">
                                            <FileOutlined style={{ fontSize: 12, color: '#52c41a' }} />
                                            <Text style={{ fontSize: 12 }}>
                                              {file.name && file.name.length > 20 ? `${file.name.substring(0, 20)}...` : (file.name || '未知文件')}
                                            </Text>
                                          </Space>
                                          <Button 
                                            type="text" 
                                            size="small"
                                            icon={<DeleteOutlined />}
                                            onClick={() => handleDeleteKimiFile(file.id)}
                                            style={{ fontSize: 10, color: '#ff4d4f' }}
                                          />
                                        </div>
                                      ))}
                                    </Space>
                                  </div>
                                </div>
                              )}
                              
                              {kimiFiles.length === 0 && (
                                <div style={{ textAlign: 'center', color: '#8c8c8c', fontSize: 12 }}>
                                  暂无上传文件
                                </div>
                              )}
                            </Space>
                          </div>
                        }
                      >
                        <Tooltip title="附件管理">
                          <Button 
                            type="text" 
                            icon={<PaperClipOutlined />} 
                            size="small"
                            style={{ color: kimiFiles.length > 0 ? '#52c41a' : '#8c8c8c' }}
                          >
                            {kimiFiles.length > 0 ? kimiFiles.length : '附件'}
                          </Button>
                        </Tooltip>
                      </Popover>
                    )}
                  </Space>
                </Col>
                
                <Col>
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    智能问答助手
                  </Text>
                </Col>
              </Row>
            </div>
            
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
              {/* 文件上传区域已移至工具栏 */}
              
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