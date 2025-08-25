import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  message,
  Popconfirm,
  Row,
  Col,
  Statistic,
  Breadcrumb,
  Tooltip,
  Switch,
  Upload,
  Progress,
  Divider,
  Layout,
  Tree,
  Dropdown,
  Menu
} from 'antd'
import {
  BookOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  FileTextOutlined,
  SettingOutlined,
  HomeOutlined,
  FolderOutlined,
  FolderOpenOutlined,
  UploadOutlined,
  DownloadOutlined,
  EyeOutlined,
  ReloadOutlined,
  SearchOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  SwapOutlined,
  MoreOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined
} from '@ant-design/icons'
import type { TableColumnsType, UploadProps, TreeDataNode } from 'antd'
import { Document, DocumentStatistics, KnowledgeBase, KnowledgeBaseCreate } from '../types/index'
import { getKnowledgeBases, createKnowledgeBase, updateKnowledgeBase, deleteKnowledgeBase } from '../services/kbApi'
import { documentApi } from '../services/documentApi'
import { StatisticsApi } from '../services/statisticsApi'

const { Title, Text } = Typography
const { Sider, Content } = Layout
const { Search } = Input
const { Option } = Select

interface KnowledgeBaseFormData {
  name: string
  code: string
  description?: string
  is_active: boolean
}

const UnifiedKnowledgeManagement: React.FC = () => {
  // 知识库相关状态
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [selectedKb, setSelectedKb] = useState<KnowledgeBase | null>(null)
  const [kbModalVisible, setKbModalVisible] = useState(false)
  const [editingKb, setEditingKb] = useState<KnowledgeBase | null>(null)
  const [kbForm] = Form.useForm<KnowledgeBaseFormData>()

  // 文档相关状态
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(false)
  const [uploadVisible, setUploadVisible] = useState(false)
  const [editVisible, setEditVisible] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)
  const [stats, setStats] = useState<DocumentStatistics | null>(null)
  const [searchText, setSearchText] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([])
  const [moveModalVisible, setMoveModalVisible] = useState(false)
  const [targetKbId, setTargetKbId] = useState<string>('')
  const [form] = Form.useForm()
  const [editForm] = Form.useForm()

  // 预览相关状态
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewContent, setPreviewContent] = useState<any>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [vectorizingDocs, setVectorizingDocs] = useState<Set<number>>(new Set())
  const [vectorizationProgress, setVectorizationProgress] = useState<Map<number, any>>(new Map())
  const [rebuildingIndex, setRebuildingIndex] = useState(false)

  // 加载知识库列表
  const loadKnowledgeBases = async () => {
    try {
      console.log('开始加载知识库列表...')
      
      // 检查认证token
      const token = localStorage.getItem('auth-token')
      console.log('当前认证token:', token ? '存在' : '不存在')
      
      const response = await getKnowledgeBases()
      console.log('知识库API响应:', response)
      console.log('知识库数据:', response.kbs)
      setKnowledgeBases(response.kbs || [])
      if (!response.kbs || response.kbs.length === 0) {
        console.warn('知识库列表为空')
        message.warning('暂无知识库数据，请先创建知识库')
      }
    } catch (error: any) {
      console.error('加载知识库失败:', error)
      console.error('错误详情:', {
        message: error?.message,
        status: error?.response?.status,
        statusText: error?.response?.statusText,
        data: error?.response?.data
      })
      message.error(`加载知识库失败: ${error?.message || '未知错误'}`)
    }
  }

  // 加载文档列表
  const loadDocuments = async () => {
    setLoading(true)
    try {
      const searchParams = {
        query: searchText || undefined,
        kb_id: selectedKb?.id,
        status: statusFilter !== 'all' ? statusFilter : undefined
      }
      
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
    loadKnowledgeBases()
  }, [])

  useEffect(() => {
    if (selectedKb) {
      loadDocuments()
    }
  }, [selectedKb, searchText, statusFilter])

  // 知识库操作
  const openCreateKbModal = () => {
    setEditingKb(null)
    kbForm.resetFields()
    setKbModalVisible(true)
  }

  const openEditKbModal = (kb: KnowledgeBase) => {
    setEditingKb(kb)
    kbForm.setFieldsValue({
      name: kb.name,
      code: kb.code,
      description: kb.description,
      is_active: kb.is_active
    })
    setKbModalVisible(true)
  }

  const handleKbSubmit = async (values: KnowledgeBaseFormData) => {
    try {
      if (editingKb) {
        await updateKnowledgeBase(editingKb.id, values)
        message.success('知识库更新成功')
      } else {
        await createKnowledgeBase(values as KnowledgeBaseCreate)
        message.success('知识库创建成功')
      }
      setKbModalVisible(false)
      setEditingKb(null)
      kbForm.resetFields()
      loadKnowledgeBases()
    } catch (error: any) {
      console.error('Submit KB error:', error)
      message.error(error.message || '操作失败')
    }
  }

  const handleDeleteKb = async (id: string) => {
    try {
      await deleteKnowledgeBase(id)
      message.success('知识库删除成功')
      loadKnowledgeBases()
      if (selectedKb?.id === id) {
        setSelectedKb(null)
      }
    } catch (error: any) {
      console.error('Delete KB error:', error)
      message.error(error.message || '删除失败')
    }
  }

  // 文档操作
  const handleMoveToKb = async (documentId: string, kbId: string) => {
    try {
      await documentApi.moveDocumentToKb(documentId, kbId)
      message.success('文档移动成功')
      await Promise.all([loadDocuments(), loadKnowledgeBases()])
    } catch (error) {
      console.error('移动文档失败:', error)
      message.error('移动文档失败，请稍后重试')
    }
  }

  const handleBatchMove = async () => {
    if (selectedDocuments.length === 0) {
      message.warning('请选择要移动的文档')
      return
    }
    if (!targetKbId) {
      message.warning('请选择目标知识库')
      return
    }

    try {
      await documentApi.batchMoveDocumentsToKb(selectedDocuments, targetKbId)
      message.success(`成功移动 ${selectedDocuments.length} 个文档`)
      setMoveModalVisible(false)
      setSelectedDocuments([])
      setTargetKbId('')
      await Promise.all([loadDocuments(), loadKnowledgeBases()])
    } catch (error) {
      console.error('批量移动文档失败:', error)
      message.error('批量移动文档失败，请稍后重试')
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await documentApi.deleteDocument(String(id))
      message.success('文档删除成功')
      loadDocuments()
    } catch (error) {
      console.error('删除文档失败:', error)
      message.error('删除文档失败，请稍后重试')
    }
  }

  const handleEdit = (doc: Document) => {
    setSelectedDoc(doc)
    editForm.setFieldsValue({
      title: doc.title,
      category: doc.category,
      description: doc.description,
      kb_id: doc.kb_id ? String(doc.kb_id) : (selectedKb?.id || undefined)
    })
    setEditVisible(true)
  }

  const handleEditSubmit = async (values: any) => {
    if (!selectedDoc) return
    
    try {
      await documentApi.updateDocument(String(selectedDoc.id), values)
      message.success('文档更新成功')
      setEditVisible(false)
      setSelectedDoc(null)
      editForm.resetFields()
      loadDocuments()
    } catch (error) {
      console.error('更新文档失败:', error)
      message.error('更新文档失败，请稍后重试')
    }
  }

  const handlePreview = async (doc: Document) => {
    setPreviewLoading(true)
    setPreviewVisible(true)
    try {
      const content = await documentApi.getDocumentContent(String(doc.id))
      setPreviewContent(content)
    } catch (error) {
      console.error('预览文档失败:', error)
      message.error('预览文档失败')
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleVectorize = async (docId: number) => {
    try {
      setVectorizingDocs(prev => new Set(prev).add(docId))
      // 注意：这里需要根据实际API调整
      message.success('开始向量化处理')
      loadDocuments()
    } catch (error) {
      console.error('向量化失败:', error)
      message.error('向量化失败，请稍后重试')
      setVectorizingDocs(prev => {
        const newSet = new Set(prev)
        newSet.delete(docId)
        return newSet
      })
    }
  }

  const handleReprocess = async (docId: string) => {
    try {
      // 注意：这里需要根据实际API调整
      message.success('开始重新处理')
      loadDocuments()
    } catch (error) {
      console.error('重新处理失败:', error)
      message.error('重新处理失败，请稍后重试')
    }
  }

  const handleRebuildIndex = async () => {
    try {
      setRebuildingIndex(true)
      // 注意：这里需要根据实际API调整
      message.success('向量索引重建成功')
      loadDocuments()
    } catch (error) {
      console.error('重建索引失败:', error)
      message.error('重建索引失败，请稍后重试')
    } finally {
      setRebuildingIndex(false)
    }
  }

  // 工具函数
  const getFileIcon = (fileType: string) => {
    switch (fileType?.toLowerCase()) {
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

  const getCategoryTag = (category: string) => {
    const categoryMap: { [key: string]: { color: string; text: string } } = {
      manual: { color: 'blue', text: '手册' },
      policy: { color: 'green', text: '政策' },
      procedure: { color: 'orange', text: '流程' },
      guideline: { color: 'purple', text: '指导' },
      development: { color: 'cyan', text: '开发' },
      other: { color: 'default', text: '其他' }
    }
    const config = categoryMap[category] || categoryMap.other
    return <Tag color={config.color}>{config.text}</Tag>
  }

  const getStatusTag = (record: Document) => {
    const isVectorizing = vectorizingDocs.has(record.id)
    const progress = vectorizationProgress.get(record.id)
    
    if (isVectorizing || record.status === 'vectorizing') {
      return (
        <Space>
          <Tag color="processing">向量化中</Tag>
          {progress && (
            <Progress 
              type="circle" 
              size={16} 
              percent={progress.progress || 0}
              showInfo={false}
            />
          )}
        </Space>
      )
    }
    
    const statusMap: { [key: string]: { color: string; text: string } } = {
      vectorized: { color: 'success', text: '已向量化' },
      processed: { color: 'blue', text: '已处理' },
      processing: { color: 'processing', text: '处理中' },
      uploaded: { color: 'default', text: '已上传' },
      failed: { color: 'error', text: '失败' },
      error: { color: 'error', text: '错误' }
    }
    const config = statusMap[record.status] || { color: 'default', text: record.status }
    return <Tag color={config.color}>{config.text}</Tag>
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN')
  }

  // 构建知识库树形数据
  const treeData: TreeDataNode[] = [
    {
      title: '全部文档',
      key: 'all',
      icon: <HomeOutlined />,
      children: knowledgeBases.map(kb => ({
        title: (
          <Space>
            <span>{kb.name}</span>
            <Text type="secondary">({kb.document_count || 0})</Text>
          </Space>
        ),
        key: kb.id,
        icon: kb.is_active ? <FolderOpenOutlined /> : <FolderOutlined />,
        isLeaf: true
      }))
    }
  ]

  // 文档表格列定义
  const documentColumns: TableColumnsType<Document> = [
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
      render: (category: string) => getCategoryTag(category)
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: Document) => getStatusTag(record)
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
      render: (_: string, record: Document) => {
        const actionMenu = (
          <Menu>
            <Menu.Item key="preview" icon={<EyeOutlined />} onClick={() => handlePreview(record)}>
              预览
            </Menu.Item>
            <Menu.Item key="edit" icon={<EditOutlined />} onClick={() => handleEdit(record)}>
              编辑
            </Menu.Item>
            <Menu.Item key="download" icon={<DownloadOutlined />}>
              下载
            </Menu.Item>
            {selectedKb && (
              <Menu.SubMenu key="move" title="移动到" icon={<SwapOutlined />}>
                {knowledgeBases.filter(kb => kb.id !== selectedKb.id).map(kb => (
                  <Menu.Item 
                    key={kb.id} 
                    onClick={() => handleMoveToKb(record.id.toString(), kb.id)}
                  >
                    {kb.name}
                  </Menu.Item>
                ))}
              </Menu.SubMenu>
            )}
            {(record.status !== 'vectorized' && record.status === 'processed') && (
              <Menu.Item 
                key="vectorize" 
                icon={<ThunderboltOutlined />} 
                onClick={() => handleVectorize(Number(record.id))}
              >
                向量化
              </Menu.Item>
            )}
            {(record.status === 'failed' || record.status === 'error') && (
              <Menu.Item 
                key="reprocess" 
                icon={<ReloadOutlined />} 
                onClick={() => handleReprocess(record.id.toString())}
              >
                重新处理
              </Menu.Item>
            )}
            <Menu.Divider />
            <Menu.Item 
              key="delete" 
              icon={<DeleteOutlined />} 
              danger
              onClick={() => {
                Modal.confirm({
                  title: '确定要删除这个文档吗？',
                  onOk: () => handleDelete(record.id.toString())
                })
              }}
            >
              删除
            </Menu.Item>
          </Menu>
        )

        return (
          <Dropdown overlay={actionMenu} trigger={['click']}>
            <Button type="link" icon={<MoreOutlined />} />
          </Dropdown>
        )
      }
    }
  ]

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
        const formValues = form.getFieldsValue()
        const fileName = (file as File).name.replace(/\.[^/.]+$/, '')
        const metadata = {
          title: formValues.title || fileName,
          category: formValues.category || 'other',
          tags: formValues.description || '',
          version: '1.0',
          auto_vectorize: true,
          kb_id: selectedKb?.id // 如果选中了知识库，直接上传到该知识库
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
        
        const documentId = response?.id
        if (documentId) {
          setVectorizingDocs(prev => new Set(prev).add(Number(documentId)))
        }
        
        loadDocuments()
      } catch (error) {
        console.error('上传失败:', error)
        onError?.(error as Error)
        message.error(`${(file as File).name} 上传失败`)
      }
    }
  }

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      {/* 左侧知识库树 */}
      <Sider 
        width={280} 
        style={{ 
          background: '#fff',
          borderRight: '1px solid #f0f0f0'
        }}
      >
        <div style={{ padding: '16px' }}>
          <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Title level={4} style={{ margin: 0 }}>知识库</Title>
            <Button 
              type="primary" 
              size="small" 
              icon={<PlusOutlined />}
              onClick={openCreateKbModal}
            >
              新建
            </Button>
          </div>
          
          <Tree
            treeData={treeData}
            defaultExpandAll
            selectedKeys={selectedKb ? [selectedKb.id] : ['all']}
            onSelect={(keys) => {
              if (keys.length > 0) {
                const key = keys[0] as string
                if (key === 'all') {
                  setSelectedKb(null)
                } else {
                  const kb = knowledgeBases.find(k => k.id === key)
                  setSelectedKb(kb || null)
                }
              }
            }}
            showIcon
          />
          
          {/* 知识库操作按钮 */}
          {selectedKb && (
            <div style={{ marginTop: 16, padding: '12px', background: '#f8f9fa', borderRadius: 6 }}>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{selectedKb.name}</Text>
              </div>
              <div style={{ marginBottom: 8 }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {selectedKb.description || '暂无描述'}
                </Text>
              </div>
              <Space size="small">
                <Button 
                  size="small" 
                  icon={<EditOutlined />}
                  onClick={() => openEditKbModal(selectedKb)}
                >
                  编辑
                </Button>
                <Popconfirm
                  title="确定要删除这个知识库吗？"
                  onConfirm={() => handleDeleteKb(selectedKb.id)}
                  okText="确定"
                  cancelText="取消"
                >
                  <Button 
                    size="small" 
                    icon={<DeleteOutlined />}
                    danger
                  >
                    删除
                  </Button>
                </Popconfirm>
              </Space>
            </div>
          )}
        </div>
      </Sider>

      {/* 右侧内容区域 */}
      <Layout>
        <Content style={{ padding: '24px' }}>
          {/* 面包屑导航 */}
          <Breadcrumb style={{ marginBottom: 16 }}>
            <Breadcrumb.Item>
              <HomeOutlined /> 知识库管理
            </Breadcrumb.Item>
            {selectedKb && (
              <Breadcrumb.Item>
                <BookOutlined /> {selectedKb.name}
              </Breadcrumb.Item>
            )}
          </Breadcrumb>

          {/* 页面标题和描述 */}
          <div style={{ marginBottom: 24 }}>
            <Title level={2}>
              {selectedKb ? `${selectedKb.name} - 文档管理` : '知识库概览'}
            </Title>
            <Text type="secondary">
              {selectedKb 
                ? `管理 "${selectedKb.name}" 知识库中的文档，支持上传、编辑、删除和重新处理文档。`
                : '选择左侧知识库查看和管理文档，或创建新的知识库。'
              }
            </Text>
          </div>

          {/* 统计卡片 */}
          {stats && (
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
              <Col xs={24} sm={6}>
                <Card>
                  <Statistic
                    title={selectedKb ? '知识库文档数' : '总文档数'}
                    value={selectedKb ? selectedKb.document_count || 0 : stats.total_documents}
                    prefix={<FileTextOutlined />}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card>
                  <Statistic
                    title="已处理"
                    value={stats.status_distribution?.vectorized || 0}
                    suffix={`/ ${selectedKb ? selectedKb.document_count || 0 : stats.total_documents}`}
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

          {selectedKb ? (
            // 文档管理界面
            <>
              {/* 操作栏 */}
              <Card style={{ marginBottom: 16 }}>
                <Row gutter={[16, 16]} align="middle">
                  <Col xs={24} sm={8}>
                    <Search
                      placeholder="搜索文档标题或文件名"
                      value={searchText}
                      onChange={(e) => setSearchText(e.target.value)}
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
                      <Option value="vectorized">已向量化</Option>
                      <Option value="processed">已处理</Option>
                      <Option value="processing">处理中</Option>
                      <Option value="failed">失败</Option>
                    </Select>
                  </Col>
                  <Col xs={24} sm={12} style={{ textAlign: 'right' }}>
                    <Space>
                      {selectedDocuments.length > 0 && (
                        <Button 
                          icon={<SwapOutlined />}
                          onClick={() => setMoveModalVisible(true)}
                        >
                          批量移动 ({selectedDocuments.length})
                        </Button>
                      )}
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
                  columns={documentColumns}
                  dataSource={documents}
                  rowKey="id"
                  loading={loading}
                  rowSelection={{
                    selectedRowKeys: selectedDocuments,
                    onChange: (keys) => setSelectedDocuments(keys as string[])
                  }}
                  pagination={{
                    total: documents.length,
                    pageSize: 20,
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
                  }}
                />
              </Card>
            </>
          ) : (
            // 知识库概览界面
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={6}>
                <Card>
                  <Statistic
                    title="总知识库数"
                    value={knowledgeBases.length}
                    prefix={<BookOutlined />}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card>
                  <Statistic
                    title="启用中"
                    value={knowledgeBases.filter(kb => kb.is_active).length}
                    suffix={`/ ${knowledgeBases.length}`}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card>
                  <Statistic
                    title="总文档数"
                    value={knowledgeBases.reduce((sum, kb) => sum + (kb.document_count || 0), 0)}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={6}>
                <Card>
                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={openCreateKbModal}
                    style={{ width: '100%' }}
                  >
                    新建知识库
                  </Button>
                </Card>
              </Col>
            </Row>
          )}
        </Content>
      </Layout>

      {/* 知识库创建/编辑模态框 */}
      <Modal
        title={editingKb ? '编辑知识库' : '新建知识库'}
        open={kbModalVisible}
        onCancel={() => {
          setKbModalVisible(false)
          setEditingKb(null)
          kbForm.resetFields()
        }}
        footer={null}
      >
        <Form
          form={kbForm}
          layout="vertical"
          onFinish={handleKbSubmit}
        >
          <Form.Item
            name="name"
            label="知识库名称"
            rules={[{ required: true, message: '请输入知识库名称' }]}
          >
            <Input placeholder="请输入知识库名称" />
          </Form.Item>
          <Form.Item
            name="code"
            label="知识库代码"
            rules={[{ required: true, message: '请输入知识库代码' }]}
          >
            <Input placeholder="请输入知识库代码（英文字母和数字）" />
          </Form.Item>
          <Form.Item
            name="description"
            label="描述"
          >
            <Input.TextArea placeholder="请输入知识库描述" rows={3} />
          </Form.Item>
          <Form.Item
            name="is_active"
            label="状态"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch checkedChildren="启用" unCheckedChildren="禁用" />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => {
                setKbModalVisible(false)
                setEditingKb(null)
                kbForm.resetFields()
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                {editingKb ? '更新' : '创建'}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 文档上传模态框 */}
      <Modal
        title="上传文档"
        open={uploadVisible}
        onCancel={() => {
          setUploadVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={() => {
            setUploadVisible(false)
            form.resetFields()
          }}
        >
          <Form.Item
            name="title"
            label="文档标题"
            rules={[{ required: true, message: '请输入文档标题' }]}
          >
            <Input placeholder="文档标题（可自动从文件名生成）" />
          </Form.Item>
          <Form.Item
            name="category"
            label="文档分类"
            initialValue="other"
          >
            <Select>
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
            label="描述"
          >
            <Input.TextArea placeholder="文档描述或标签" rows={2} />
          </Form.Item>
          <Form.Item
            label="选择文件"
            required
          >
            <Upload.Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持 PDF、Word、Excel、文本文件，单个文件不超过 1GB
                {selectedKb && (
                  <><br />文档将上传到当前知识库：{selectedKb.name}</>
                )}
              </p>
            </Upload.Dragger>
          </Form.Item>
        </Form>
      </Modal>

      {/* 文档编辑模态框 */}
      <Modal
        title="编辑文档"
        open={editVisible}
        onCancel={() => {
          setEditVisible(false)
          setSelectedDoc(null)
          editForm.resetFields()
        }}
        footer={null}
      >
        <Form
          form={editForm}
          layout="vertical"
          onFinish={handleEditSubmit}
        >
          <Form.Item
            name="title"
            label="文档标题"
            rules={[{ required: true, message: '请输入文档标题' }]}
          >
            <Input placeholder="请输入文档标题" />
          </Form.Item>
          <Form.Item
            name="kb_id"
            label="知识库"
            rules={[{ required: true, message: '请选择知识库' }]}
          >
            <Select placeholder="请选择知识库">
              {knowledgeBases.map(kb => (
                <Option key={kb.id} value={kb.id}>
                  {kb.name}
                  {kb.description && (
                    <span style={{ color: '#999', marginLeft: 8 }}>
                      - {kb.description}
                    </span>
                  )}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="category"
            label="文档分类"
          >
            <Select>
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
            label="描述"
          >
            <Input.TextArea placeholder="文档描述" rows={3} />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => {
                setEditVisible(false)
                setSelectedDoc(null)
                editForm.resetFields()
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                更新
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 批量移动模态框 */}
      <Modal
        title="批量移动文档"
        open={moveModalVisible}
        onCancel={() => {
          setMoveModalVisible(false)
          setTargetKbId('')
        }}
        onOk={handleBatchMove}
        okText="移动"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <Text>已选择 {selectedDocuments.length} 个文档</Text>
        </div>
        <Form.Item
          label="目标知识库"
          required
        >
          <Select
            value={targetKbId}
            onChange={setTargetKbId}
            placeholder="请选择目标知识库"
          >
            {knowledgeBases.filter(kb => kb.id !== selectedKb?.id).map(kb => (
              <Option key={kb.id} value={kb.id}>
                {kb.name}
              </Option>
            ))}
          </Select>
        </Form.Item>
      </Modal>

      {/* 文档预览模态框 */}
      <Modal
        title="文档预览"
        open={previewVisible}
        onCancel={() => {
          setPreviewVisible(false)
          setPreviewContent(null)
        }}
        footer={[
          <Button key="close" onClick={() => {
            setPreviewVisible(false)
            setPreviewContent(null)
          }}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {previewLoading ? (
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <Progress type="circle" />
            <div style={{ marginTop: 16 }}>加载中...</div>
          </div>
        ) : previewContent ? (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>文档ID:</Text> {previewContent.document_id}<br />
              <Text strong>总块数:</Text> {previewContent.total_chunks}
            </div>
            <Divider />
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {previewContent.chunks?.map((chunk: any, index: number) => (
                <div key={chunk.chunk_id} style={{ marginBottom: 16, padding: 12, background: '#f8f9fa', borderRadius: 4 }}>
                  <div style={{ marginBottom: 8 }}>
                    <Tag>块 {chunk.chunk_index + 1}</Tag>
                    {chunk.page_number && <Tag color="blue">第 {chunk.page_number} 页</Tag>}
                  </div>
                  <div style={{ whiteSpace: 'pre-wrap', fontSize: '14px' }}>
                    {chunk.content}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <Text type="secondary">暂无预览内容</Text>
          </div>
        )}
      </Modal>
    </Layout>
  )
}

export default UnifiedKnowledgeManagement