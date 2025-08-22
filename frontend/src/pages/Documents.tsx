import { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Typography,
  Upload,
  Modal,
  Form,
  Input,
  Select,
  message,
  Popconfirm,
  Progress,
  Tooltip,
  Row,
  Col,
  Statistic,
  Divider
} from 'antd'
import {
  UploadOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  EditOutlined,
  FileTextOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  ReloadOutlined,
  SearchOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'
import type { UploadProps, TableColumnsType } from 'antd'
import { Document, DocumentStatistics } from '../types/index'
import { documentApi } from '../services/documentApi'
import { StatisticsApi } from '../services/statisticsApi'

const { Title, Text } = Typography
const { Option } = Select
const { Search } = Input

interface DocumentsProps {
  user?: any
}

const Documents = ({ user }: DocumentsProps) => {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(false)
  const [uploadVisible, setUploadVisible] = useState(false)
  const [editVisible, setEditVisible] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)
  const [stats, setStats] = useState<DocumentStatistics | null>(null)
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [form] = Form.useForm()
  const [editForm] = Form.useForm()
  
  // 预览相关状态
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewContent, setPreviewContent] = useState<{
    document_id: string
    total_chunks: number
    chunks: Array<{
      chunk_id: string
      chunk_index: number
      content: string
      page_number?: number
      metadata?: any
    }>
  } | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [vectorizingDocs, setVectorizingDocs] = useState<Set<number>>(new Set())
  const [vectorizationProgress, setVectorizationProgress] = useState<Map<number, {
    progress: number;
    current_step: string;
    message: string;
    status: string;
    error?: string;
  }>>(new Map())
  const [rebuildingIndex, setRebuildingIndex] = useState(false)

  // 加载文档列表
  const loadDocuments = async () => {
    setLoading(true)
    try {
      // 构建搜索参数
      const searchParams = {
        query: searchText || undefined,
        category: categoryFilter !== 'all' ? categoryFilter : undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined
      }
      
      // 获取文档列表和统计数据
      const [documentsResponse, statsData] = await Promise.all([
        documentApi.getDocuments(searchParams),
        StatisticsApi.getDocumentStatistics()
      ])
      
      setDocuments(documentsResponse.items || [])
      setStats(statsData)
    } catch (error) {
      console.error('加载文档失败:', error)
      message.error('加载文档失败，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  // 文件上传配置
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: true,
    accept: '.pdf,.doc,.docx,.txt,.md,.xlsx,.xls',
    showUploadList: {
      showPreviewIcon: false,
      showRemoveIcon: true,
      showDownloadIcon: false
    },
    beforeUpload: (file: File) => {
      const isValidType = ['pdf', 'doc', 'docx', 'txt', 'md', 'xlsx', 'xls'].some(
        type => file.name.toLowerCase().endsWith(type)
      )
      if (!isValidType) {
        message.error('只支持 PDF、Word、Excel、文本文件格式')
        return false
      }
      const isLt1G = file.size / 1024 / 1024 < 1024
      if (!isLt1G) {
        message.error('文件大小不能超过 1GB')
        return false
      }
      
      // 自动填充文件名到标题字段（去除扩展名）
      const fileName = file.name.replace(/\.[^/.]+$/, '')
      const currentTitle = form.getFieldValue('title')
      if (!currentTitle) {
        form.setFieldsValue({ title: fileName })
      }
      
      return true
    },
    customRequest: async (options) => {
      const { file, onSuccess, onError, onProgress } = options
      try {
        // 获取表单值，不进行验证（因为标题可以自动生成）
        const formValues = form.getFieldsValue()
        
        // 准备上传参数，如果标题为空则使用文件名
        const fileName = (file as File).name.replace(/\.[^/.]+$/, '')
        const metadata = {
          title: formValues.title || fileName,
          category: formValues.category || 'other',
          tags: formValues.description || '',
          version: '1.0',
          auto_vectorize: true  // 改为自动向量化
        }
        
        const response = await documentApi.uploadDocument(
          file as File,
          metadata,
          (progress: number) => {
            onProgress?.({ percent: progress })
          }
        )
        
        onSuccess?.(response)
        message.success(`${(file as File).name} 上传成功，正在开始向量化...`)
        
        // 上传成功后自动开始监控向量化进度
        const documentId = response?.id
        const numericId = Number(documentId)
        if (numericId) {
          // 添加到向量化中的文档列表（统一使用数字ID）
          setVectorizingDocs(prev => new Set([...prev, numericId]))
          
          // 初始化进度状态
          setVectorizationProgress(prev => {
            const newMap = new Map(prev)
            newMap.set(numericId, {
              progress: 0,
              current_step: '准备开始',
              message: '正在启动向量化任务...',
              status: 'starting'
            })
            return newMap
          })
          
          // 延迟2秒后开始轮询（给后端一些时间开始处理）
          setTimeout(() => {
            pollVectorizationStatus(numericId)
          }, 2000)
        } else {
          console.warn('上传响应中未找到文档ID:', response)
          message.warning('上传成功，但无法监控向量化进度')
        }
        
        form.resetFields() // 重置表单
        setUploadVisible(false) // 关闭模态框
        loadDocuments() // 重新加载文档列表
      } catch (error: any) {
        console.error('文件上传失败:', error)
        onError?.(error as Error)
        message.error(`${(file as File).name} 上传失败: ${error.message || '未知错误'}`)
      }
    }
  }

  // 删除文档
  const handleDelete = async (id: number) => {
    try {
      await documentApi.deleteDocument(id)
      message.success('删除成功')
      loadDocuments() // 重新加载文档列表
    } catch (error) {
      console.error('删除文档失败:', error)
      message.error('删除失败，请稍后重试')
    }
  }

  // 重新处理文档
  const handleReprocess = async (id: number) => {
    try {
      await documentApi.vectorizeDocument(id)
      message.success('重新处理已开始')
      loadDocuments() // 重新加载文档列表
    } catch (error) {
      console.error('重新处理文档失败:', error)
      message.error('重新处理失败，请稍后重试')
    }
  }

  // 向量化文档
  const handleVectorize = async (id: number) => {
    try {
      // 添加到向量化中的文档列表
      setVectorizingDocs(prev => new Set([...prev, id]))
      
      // 初始化进度状态
      setVectorizationProgress(prev => {
        const newMap = new Map(prev)
        newMap.set(id, {
          progress: 0,
          current_step: '准备开始',
          message: '正在启动向量化任务...',
          status: 'starting'
        })
        return newMap
      })
      
      const response = await documentApi.vectorizeDocument(id)
      message.success('向量化已开始，请稍后查看状态')
      
      // 延迟1秒后开始轮询检查向量化状态
      setTimeout(() => {
        pollVectorizationStatus(id)
      }, 1000)
      
    } catch (error: any) {
      console.error('向量化文档失败:', error)
      message.error(`向量化失败: ${error.message || '请稍后重试'}`)
      
      // 从向量化列表中移除
      setVectorizingDocs(prev => {
        const newSet = new Set(prev)
        newSet.delete(id)
        return newSet
      })
      
      // 更新进度状态为错误
      setVectorizationProgress(prev => {
        const newMap = new Map(prev)
        newMap.set(id, {
          progress: 0,
          current_step: '启动失败',
          message: `向量化启动失败: ${error.message || '未知错误'}`,
          status: 'error',
          error: error.message || '未知错误'
        })
        return newMap
      })
      
      // 3秒后清除错误状态
      setTimeout(() => {
        setVectorizationProgress(prev => {
          const newMap = new Map(prev)
          newMap.delete(id)
          return newMap
        })
      }, 3000)
    }
  }

  // 重建向量数据库
  const handleRebuildIndex = async () => {
    try {
      setRebuildingIndex(true)
      message.info('正在重建向量数据库，请稍候...')
      
      await documentApi.rebuildVectorIndex()
      message.success('向量数据库重建成功')
      
      // 重新加载文档列表以更新状态
      loadDocuments()
    } catch (error) {
      console.error('重建向量数据库失败:', error)
      message.error('重建向量数据库失败，请稍后重试')
    } finally {
      setRebuildingIndex(false)
    }
  }

  // 轮询向量化状态
  const pollVectorizationStatus = (id: number) => {
    let pollCount = 0
    const maxPolls = 150 // 最多轮询150次 (5分钟)
    
    const pollInterval = setInterval(async () => {
      pollCount++
      
      try {
        // 获取详细的向量化进度
        console.log(`正在检查文档 ${id} 的向量化进度...`)
        const progressData = await documentApi.getVectorizationProgress(id)
        console.log(`文档 ${id} 向量化进度:`, progressData)
        
        // 更新进度状态
        setVectorizationProgress(prev => {
          const newMap = new Map(prev)
          newMap.set(id, {
            progress: progressData.progress,
            current_step: progressData.current_step,
            message: progressData.message,
            status: progressData.status,
            error: progressData.error || undefined
          })
          return newMap
        })
        
        // 如果向量化完成或失败，停止轮询
        if (progressData.status === 'completed' || progressData.status === 'error') {
          clearInterval(pollInterval)
          
          // 从向量化列表中移除
          setVectorizingDocs(prev => {
            const newSet = new Set(prev)
            newSet.delete(id)
            return newSet
          })
          
          // 清除进度状态（延迟3秒以便用户看到完成状态）
          setTimeout(() => {
            setVectorizationProgress(prev => {
              const newMap = new Map(prev)
              newMap.delete(id)
              return newMap
            })
          }, 3000)
          
          if (progressData.status === 'completed') {
            message.success('文档向量化完成')
          } else if (progressData.status === 'error') {
            message.error(`文档向量化失败: ${progressData.error || '未知错误'}`)
          }
          
          // 重新加载文档列表
          loadDocuments()
          return
        }
        
        // 检查是否超过最大轮询次数
        if (pollCount >= maxPolls) {
          clearInterval(pollInterval)
          setVectorizingDocs(prev => {
            const newSet = new Set(prev)
            newSet.delete(id)
            return newSet
          })
          setVectorizationProgress(prev => {
            const newMap = new Map(prev)
            newMap.set(id, {
              progress: 0,
              current_step: '轮询超时',
              message: '向量化状态检查超时，请手动刷新页面查看结果',
              status: 'timeout',
              error: '轮询超时'
            })
            return newMap
          })
          message.warning('向量化状态检查超时，请刷新页面查看最新状态')
          setTimeout(() => {
            setVectorizationProgress(prev => {
              const newMap = new Map(prev)
              newMap.delete(id)
              return newMap
            })
          }, 5000)
        }
        
      } catch (error: any) {
        console.error(`检查文档 ${id} 向量化状态失败 (第${pollCount}次):`, error)
        
        // 如果连续失败多次，尝试回退检查
        if (pollCount % 5 === 0) { // 每5次失败尝试一次回退检查
          console.log(`尝试回退检查文档 ${id} 状态...`)
          try {
            const response = await documentApi.getDocuments()
            const document = response.items.find((doc: Document) => doc.id === id)
            
            if (document) {
              if (document.is_vectorized || document.status === 'vectorized') {
                clearInterval(pollInterval)
                setVectorizingDocs(prev => {
                  const newSet = new Set(prev)
                  newSet.delete(id)
                  return newSet
                })
                setVectorizationProgress(prev => {
                  const newMap = new Map(prev)
                  newMap.delete(id)
                  return newMap
                })
                message.success('文档向量化完成')
                loadDocuments()
                return
              } else if (document.status === 'failed' || document.processing_status === 'failed') {
                clearInterval(pollInterval)
                setVectorizingDocs(prev => {
                  const newSet = new Set(prev)
                  newSet.delete(id)
                  return newSet
                })
                setVectorizationProgress(prev => {
                  const newMap = new Map(prev)
                  newMap.delete(id)
                  return newMap
                })
                message.error('文档向量化失败')
                loadDocuments()
                return
              }
            }
          } catch (fallbackError) {
            console.error('回退检查也失败:', fallbackError)
          }
        }
        
        // 如果超过最大轮询次数，停止轮询
        if (pollCount >= maxPolls) {
          clearInterval(pollInterval)
          setVectorizingDocs(prev => {
            const newSet = new Set(prev)
            newSet.delete(id)
            return newSet
          })
          setVectorizationProgress(prev => {
            const newMap = new Map(prev)
            newMap.delete(id)
            return newMap
          })
          message.error('向量化状态检查失败，请刷新页面查看最新状态')
        }
      }
    }, 2000) // 每2秒检查一次
  }

  // 编辑文档信息
  const handleEdit = (doc: Document) => {
    setSelectedDoc(doc)
    editForm.setFieldsValue({
      title: doc.title,
      category: doc.category,
      description: doc.description
    })
    setEditVisible(true)
  }

  // 保存编辑
  const handleSaveEdit = async () => {
    try {
      const values = await editForm.validateFields()
      if (selectedDoc) {
        await documentApi.updateDocument(selectedDoc.id, values)
        message.success('更新成功')
        loadDocuments() // 重新加载文档列表
      }
      
      setEditVisible(false)
      setSelectedDoc(null)
    } catch (error) {
      console.error('更新文档失败:', error)
      message.error('更新失败，请稍后重试')
    }
  }

  // 预览文档
  const handlePreview = async (doc: Document) => {
    setSelectedDoc(doc)
    setPreviewVisible(true)
    setPreviewLoading(true)
    
    try {
      const content = await documentApi.getDocumentContent(doc.id)
      setPreviewContent(content)
    } catch (error) {
      console.error('获取文档内容失败:', error)
      message.error('获取文档内容失败，请稍后重试')
    } finally {
      setPreviewLoading(false)
    }
  }

  // 获取文件图标
  const getFileIcon = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return <FilePdfOutlined style={{ color: '#ff4d4f' }} />
      case 'doc':
      case 'docx':
        return <FileWordOutlined style={{ color: '#1890ff' }} />
      case 'xls':
      case 'xlsx':
        return <FileExcelOutlined style={{ color: '#52c41a' }} />
      default:
        return <FileTextOutlined style={{ color: '#666' }} />
    }
  }

  // 获取状态标签
  const getStatusTag = (record: any) => {
    const { status, processing_status } = record
    
    // 根据状态组合确定显示
    let color = 'default'
    let text = status
    
    // 判断是否已向量化：status为'vectorized'
    if (status === 'vectorized') {
      color = 'green'
      text = '已向量化'
    } else if (status === 'completed' && processing_status === 'completed') {
      color = 'blue'
      text = '已处理'
    } else if (processing_status === 'vectorizing' || vectorizingDocs.has(record.id)) {
      color = 'orange'
      text = '向量化中'
    } else {
      const statusMap = {
        uploaded: { color: 'cyan', text: '已上传' },
        processing: { color: 'blue', text: '处理中' },
        completed: { color: 'blue', text: '已处理' },
        failed: { color: 'red', text: '失败' },
        pending: { color: 'orange', text: '待处理' },
        error: { color: 'red', text: '错误' },
        active: { color: 'green', text: '活跃' }
      }
      const config = statusMap[status as keyof typeof statusMap] || statusMap[processing_status as keyof typeof statusMap]
      if (config) {
        color = config.color
        text = config.text
      }
    }
    
    return <Tag color={color}>{text}</Tag>
  }

  // 获取分类标签
  const getCategoryTag = (category: string) => {
    const categoryMap = {
      manual: { color: 'blue', text: '手册' },
      policy: { color: 'green', text: '政策' },
      procedure: { color: 'orange', text: '流程' },
      guideline: { color: 'purple', text: '指导' },
      development: { color: 'cyan', text: '开发' },
      other: { color: 'default', text: '其他' }
    }
    const config = categoryMap[category as keyof typeof categoryMap] || { color: 'default', text: category }
    return <Tag color={config.color}>{config.text}</Tag>
  }

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 格式化时间
  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN')
  }

  // 过滤文档
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.title.toLowerCase().includes(searchText.toLowerCase()) ||
                         doc.filename.toLowerCase().includes(searchText.toLowerCase())
    const matchesStatus = statusFilter === 'all' || doc.processing_status === statusFilter
    const matchesCategory = categoryFilter === 'all' || doc.category === categoryFilter
    return matchesSearch && matchesStatus && matchesCategory
  })

  // 表格列定义
  const columns: TableColumnsType<Document> = [
    {
      title: '文档',
      dataIndex: 'title',
      key: 'title',
      render: (text: string, record: Document) => (
        <Space>
          {getFileIcon(record.file_type)}
          <div>
            <div style={{ fontWeight: 500 }}>{text}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.filename}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => getCategoryTag(category),
      filters: [
        { text: '手册', value: 'manual' },
        { text: '政策', value: 'policy' },
        { text: '流程', value: 'procedure' },
        { text: '指导', value: 'guideline' },
        { text: '开发', value: 'development' },
        { text: '其他', value: 'other' }
      ],
      onFilter: (value, record) => record.category === value
    },
    {
      title: '状态',
      dataIndex: 'processing_status',
      key: 'status',
      render: (status: string, record: Document) => getStatusTag(record),
      filters: [
        { text: '已向量化', value: 'vectorized' },
        { text: '已处理', value: 'completed' },
        { text: '向量化中', value: 'vectorizing' },
        { text: '处理中', value: 'processing' },
        { text: '失败', value: 'failed' },
        { text: '待处理', value: 'pending' }
      ],
      onFilter: (value, record) => {
        if (value === 'vectorized') {
          return record.status === 'vectorized' || (record.status === 'completed' && record.is_vectorized)
        }
        if (value === 'completed') {
          return record.status === 'completed' && !record.is_vectorized
        }
        return record.processing_status === value || record.status === value
      }
    },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'size',
      render: (size: number) => formatFileSize(size),
      sorter: (a: Document, b: Document) => a.file_size - b.file_size
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => formatTime(time),
      sorter: (a: Document, b: Document) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: string, record: Document) => (
        <Space>
          <Tooltip title="预览">
            <Button 
              type="link" 
              icon={<EyeOutlined />} 
              size="small"
              onClick={() => handlePreview(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button 
              type="link" 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => handleEdit(record)}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button type="link" icon={<DownloadOutlined />} size="small" />
          </Tooltip>
          {/* 显示向量化相关操作和进度 */}
          {(record.status !== 'vectorized' && record.processing_status === 'completed') || 
           vectorizingDocs.has(record.id) || 
           record.processing_status === 'vectorizing' ||
           record.status === 'vectorizing' ? (
            <div>
              {(vectorizingDocs.has(record.id) || record.processing_status === 'vectorizing' || record.status === 'vectorizing') ? (
                <div style={{ minWidth: 120 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Progress 
                      type="circle" 
                      size={20} 
                      percent={vectorizationProgress.get(record.id)?.progress || 0}
                      showInfo={false}
                      strokeColor="#1890ff"
                    />
                    <Tooltip title={vectorizationProgress.get(record.id)?.message || '向量化中...'}>
                      <Text style={{ fontSize: '12px', color: '#666' }}>
                        {vectorizationProgress.get(record.id)?.current_step || '处理中'}
                      </Text>
                    </Tooltip>
+                   <Text style={{ fontSize: '12px', color: '#666' }}>
+                     {(vectorizationProgress.get(record.id)?.progress ?? 0)}%
+                   </Text>
                  </div>
                  {vectorizationProgress.get(record.id)?.error && (
                    <Text type="danger" style={{ fontSize: '11px' }}>
                      {vectorizationProgress.get(record.id)?.error}
                    </Text>
                  )}
                </div>
              ) : (
                <Tooltip title="向量化">
                  <Button 
                    type="link" 
                    icon={<ThunderboltOutlined />} 
                    size="small"
                    onClick={() => handleVectorize(record.id)}
                    style={{ color: '#1890ff' }}
                  />
                </Tooltip>
              )}
            </div>
          ) : null}
          {record.processing_status === 'failed' && (
            <Tooltip title="重新处理">
              <Button 
                type="link" 
                icon={<ReloadOutlined />} 
                size="small"
                onClick={() => handleReprocess(record.id)}
              />
            </Tooltip>
          )}
          <Popconfirm
            title="确定要删除这个文档吗？"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Tooltip title="删除">
              <Button type="link" icon={<DeleteOutlined />} size="small" danger />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ]

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>文档管理</Title>
        <Text type="secondary">
          管理知识库文档，支持上传、编辑、删除和重新处理文档。
        </Text>
      </div>

      {/* 统计卡片 */}
      {stats && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="总文档数"
                value={stats.total_documents}
                prefix={<FileTextOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="已处理"
                value={stats.status_distribution?.vectorized || 0}
                suffix={`/ ${stats.total_documents}`}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="文档块数"
                value={stats.vector_store_stats?.total_chunks || 0}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card>
              <Statistic
                title="存储大小"
                value={formatFileSize(stats.vector_store_stats?.total_size || 0)}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 操作栏 */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={8}>
            <Search
              placeholder="搜索文档标题或文件名"
              value={searchText}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchText(e.target.value)}
              prefix={<SearchOutlined />}
            />
          </Col>
          <Col xs={12} sm={4}>
            <Select
              value={statusFilter}
              onChange={setStatusFilter}
              style={{ width: '100%' }}
            >
              <Option value="all">全部状态</Option>
              <Option value="completed">已完成</Option>
              <Option value="processing">处理中</Option>
              <Option value="failed">失败</Option>
              <Option value="pending">待处理</Option>
            </Select>
          </Col>
          <Col xs={12} sm={4}>
            <Select
              value={categoryFilter}
              onChange={setCategoryFilter}
              style={{ width: '100%' }}
            >
              <Option value="all">全部分类</Option>
              <Option value="manual">手册</Option>
              <Option value="policy">政策</Option>
              <Option value="procedure">流程</Option>
              <Option value="guideline">指导</Option>
              <Option value="other">其他</Option>
            </Select>
          </Col>
          <Col xs={24} sm={8} style={{ textAlign: 'right' }}>
            <Space>
              <Button 
                icon={<ReloadOutlined />}
                onClick={loadDocuments}
                loading={loading}
              >
                刷新
              </Button>
              <Tooltip title="重建向量数据库以同步已删除的文档">
                <Button 
                  icon={<DatabaseOutlined />}
                  onClick={handleRebuildIndex}
                  loading={rebuildingIndex}
                  type="default"
                >
                  重建索引
                </Button>
              </Tooltip>
              <Button 
                type="primary" 
                icon={<UploadOutlined />}
                onClick={() => setUploadVisible(true)}
              >
                上传文档
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 文档表格 */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredDocuments}
          rowKey="id"
          loading={loading}
          pagination={{
            total: filteredDocuments.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total: number, range: [number, number]) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
          }}
        />
      </Card>

      {/* 上传文档模态框 */}
      <Modal
        title="上传文档"
        open={uploadVisible}
        onCancel={() => setUploadVisible(false)}
        footer={null}
        width={600}
      >
        {/* 上传成功后将自动关闭弹窗，进度将在文档列表中展示 */}
        <Form form={form} layout="vertical">
          <Form.Item
            name="title"
            label="文档标题"
            help="留空将自动使用文件名作为标题"
          >
            <Input placeholder="留空将自动使用文件名" />
          </Form.Item>
          
          <Form.Item
            name="category"
            label="文档分类"
            help="系统将根据文档内容自动分类，也可手动指定"
          >
            <Select placeholder="自动分类（可手动选择）" allowClear>
              <Option value="manual">手册</Option>
              <Option value="policy">政策</Option>
              <Option value="procedure">流程</Option>
              <Option value="guideline">指导</Option>
              <Option value="development">开发</Option>
              <Option value="other">其他</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="description"
            label="文档描述"
          >
            <Input.TextArea 
              rows={3} 
              placeholder="请输入文档描述（可选）"
            />
          </Form.Item>
          
          <Form.Item label="选择文件">
            <Upload.Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持 PDF、Word、Excel、文本文件，单个文件不超过 1GB
              </p>
            </Upload.Dragger>
          </Form.Item>
        </Form>
      </Modal>

      {/* 编辑文档模态框 */}
      <Modal
        title="编辑文档信息"
        open={editVisible}
        onOk={handleSaveEdit}
        onCancel={() => {
          setEditVisible(false)
          setSelectedDoc(null)
        }}
        okText="保存"
        cancelText="取消"
      >
        <Form form={editForm} layout="vertical">
          <Form.Item
            name="title"
            label="文档标题"
            rules={[{ required: true, message: '请输入文档标题' }]}
          >
            <Input placeholder="请输入文档标题" />
          </Form.Item>
          
          <Form.Item
            name="category"
            label="文档分类"
            help="系统将根据文档内容自动分类，也可手动指定"
          >
            <Select placeholder="自动分类（可手动选择）" allowClear>
              <Option value="manual">手册</Option>
              <Option value="policy">政策</Option>
              <Option value="procedure">流程</Option>
              <Option value="guideline">指导</Option>
              <Option value="development">开发</Option>
              <Option value="other">其他</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="description"
            label="文档描述"
          >
            <Input.TextArea 
              rows={3} 
              placeholder="请输入文档描述（可选）"
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* 文档预览模态框 */}
      <Modal
        title={`文档预览 - ${selectedDoc?.title || ''}`}
        open={previewVisible}
        onCancel={() => {
          setPreviewVisible(false)
          setSelectedDoc(null)
          setPreviewContent(null)
        }}
        footer={[
          <Button key="close" onClick={() => {
            setPreviewVisible(false)
            setSelectedDoc(null)
            setPreviewContent(null)
          }}>
            关闭
          </Button>
        ]}
        width={800}
        style={{ top: 20 }}
      >
        {previewLoading ? (
          <div style={{ textAlign: 'center', padding: '50px 0' }}>
            <Text>正在加载文档内容...</Text>
          </div>
        ) : previewContent ? (
          <div>
            <div style={{ marginBottom: 16, padding: '8px 12px', backgroundColor: '#f5f5f5', borderRadius: 4 }}>
              <Text type="secondary">
                共 {previewContent.total_chunks} 个文档片段
              </Text>
            </div>
            <div style={{ maxHeight: '60vh', overflow: 'auto' }}>
              {previewContent.chunks.map((chunk, index) => (
                <div key={chunk.chunk_id} style={{ marginBottom: 16 }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: 8,
                    padding: '4px 8px',
                    backgroundColor: '#fafafa',
                    borderRadius: 4
                  }}>
                    <Text strong>片段 {chunk.chunk_index + 1}</Text>
                    {chunk.page_number && (
                      <Tag color="blue">第 {chunk.page_number} 页</Tag>
                    )}
                  </div>
                  <div style={{ 
                    padding: '12px 16px',
                    border: '1px solid #e8e8e8',
                    borderRadius: 4,
                    backgroundColor: '#fff',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap'
                  }}>
                    {chunk.content}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '50px 0' }}>
            <Text type="secondary">暂无内容</Text>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default Documents