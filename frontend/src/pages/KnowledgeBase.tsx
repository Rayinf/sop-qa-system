import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Switch,
  message,
  Popconfirm,
  Tag,
  Typography,
  Row,
  Col,
  Statistic,
  Tooltip
} from 'antd'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  BookOutlined,
  FileTextOutlined,
  EyeOutlined
} from '@ant-design/icons'
import { ColumnsType } from 'antd/es/table'
import {
  getKnowledgeBases,
  createKnowledgeBase,
  updateKnowledgeBase,
  deleteKnowledgeBase,
  getKnowledgeBaseDocuments
} from '../services/kbApi'
import { KnowledgeBase as KBType, Document } from '../types/index'

const { Title, Text } = Typography
const { TextArea } = Input

interface KnowledgeBaseFormData {
  name: string
  code: string
  description?: string
  is_active: boolean
}

const KnowledgeBase: React.FC = () => {
  const [knowledgeBases, setKnowledgeBases] = useState<KBType[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [editingKB, setEditingKB] = useState<KBType | null>(null)
  const [documentsModalVisible, setDocumentsModalVisible] = useState(false)
  const [selectedKBDocuments, setSelectedKBDocuments] = useState<Document[]>([])
  const [selectedKBName, setSelectedKBName] = useState('')
  const [form] = Form.useForm<KnowledgeBaseFormData>()

  // 加载知识库列表
  const loadKnowledgeBases = async () => {
    try {
      setLoading(true)
      const data = await getKnowledgeBases()
      setKnowledgeBases(data.kbs || [])
    } catch (error) {
      message.error('加载知识库列表失败')
      console.error('Load knowledge bases error:', error)
    } finally {
      setLoading(false)
    }
  }

  // 查看知识库文档
  const viewKBDocuments = async (kb: KBType) => {
    try {
      const response = await getKnowledgeBaseDocuments(kb.id)
      setSelectedKBDocuments(response?.items || [])
      setSelectedKBName(kb.name)
      setDocumentsModalVisible(true)
    } catch (error) {
      message.error('加载知识库文档失败')
      console.error('Load KB documents error:', error)
    }
  }

  // 创建或更新知识库
  const handleSubmit = async (values: KnowledgeBaseFormData) => {
    try {
      if (editingKB) {
        await updateKnowledgeBase(editingKB.id, values)
        message.success('知识库更新成功')
      } else {
        await createKnowledgeBase(values)
        message.success('知识库创建成功')
      }
      setModalVisible(false)
      setEditingKB(null)
      form.resetFields()
      loadKnowledgeBases()
    } catch (error) {
      message.error(editingKB ? '更新知识库失败' : '创建知识库失败')
      console.error('Submit KB error:', error)
    }
  }

  // 删除知识库
  const handleDelete = async (id: string) => {
    try {
      await deleteKnowledgeBase(id)
      message.success('知识库删除成功')
      loadKnowledgeBases()
    } catch (error) {
      message.error('删除知识库失败')
      console.error('Delete KB error:', error)
    }
  }

  // 打开编辑模态框
  const openEditModal = (kb: KBType) => {
    setEditingKB(kb)
    form.setFieldsValue({
      name: kb.name,
      code: kb.code,
      description: kb.description,
      is_active: kb.is_active
    })
    setModalVisible(true)
  }

  // 打开创建模态框
  const openCreateModal = () => {
    setEditingKB(null)
    form.resetFields()
    form.setFieldsValue({ is_active: true })
    setModalVisible(true)
  }

  // 表格列配置
  const columns: ColumnsType<KBType> = [
    {
      title: '知识库名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: KBType) => (
        <Space>
          <BookOutlined style={{ color: record.is_active ? '#1890ff' : '#d9d9d9' }} />
          <Text strong={record.is_active}>{text}</Text>
        </Space>
      )
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (text: string) => text || '-'
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 100,
      render: (isActive: boolean) => (
        <Tag color={isActive ? 'green' : 'default'}>
          {isActive ? '启用' : '禁用'}
        </Tag>
      )
    },
    {
      title: '文档数量',
      dataIndex: 'document_count',
      key: 'document_count',
      width: 120,
      render: (count: number = 0) => (
        <Text type={count > 0 ? 'success' : 'secondary'}>
          {count} 个文档
        </Text>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_, record: KBType) => (
        <Space>
          <Tooltip title="查看文档">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => viewKBDocuments(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => openEditModal(record)}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个知识库吗？"
            description="删除后将无法恢复，请谨慎操作。"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Tooltip title="删除">
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ]

  // 文档表格列配置
  const documentColumns: ColumnsType<Document> = [
    {
      title: '文档名称',
      dataIndex: 'title',
      key: 'title',
      render: (text: string) => (
        <Space>
          <FileTextOutlined />
          <Text>{text}</Text>
        </Space>
      )
    },
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      ellipsis: true
    },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      width: 120,
      render: (size: number) => {
        if (!size) return '-'
        const kb = size / 1024
        return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`
      }
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString()
    }
  ]

  useEffect(() => {
    loadKnowledgeBases()
  }, [])

  // 统计数据
  const stats = {
    total: knowledgeBases.length,
    active: knowledgeBases.filter(kb => kb.is_active).length,
    totalDocuments: knowledgeBases.reduce((sum, kb) => sum + (kb.document_count || 0), 0)
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <Title level={2}>知识库管理</Title>
        <Text type="secondary">管理系统中的知识库，包括创建、编辑、删除和查看文档</Text>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} className="mb-6">
        <Col span={8}>
          <Card>
            <Statistic
              title="总知识库数"
              value={stats.total}
              prefix={<BookOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="启用知识库"
              value={stats.active}
              prefix={<BookOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="总文档数"
              value={stats.totalDocuments}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 知识库表格 */}
      <Card>
        <div className="flex justify-between items-center mb-4">
          <Title level={4} className="mb-0">知识库列表</Title>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={openCreateModal}
          >
            新建知识库
          </Button>
        </div>

        <Table
          columns={columns}
          dataSource={knowledgeBases}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`
          }}
        />
      </Card>

      {/* 创建/编辑知识库模态框 */}
      <Modal
        title={editingKB ? '编辑知识库' : '新建知识库'}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false)
          setEditingKB(null)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{ is_active: true }}
        >
          <Form.Item
            name="name"
            label="知识库名称"
            rules={[
              { required: true, message: '请输入知识库名称' },
              { max: 100, message: '名称长度不能超过100个字符' }
            ]}
          >
            <Input placeholder="请输入知识库名称" />
          </Form.Item>

          <Form.Item
            name="code"
            label="知识库代码"
            rules={[
              { required: true, message: '请输入知识库代码' },
              { max: 50, message: '代码长度不能超过50个字符' },
              { pattern: /^[A-Z0-9_]+$/, message: '代码只能包含大写字母、数字和下划线' }
            ]}
          >
            <Input placeholder="请输入知识库代码，如：QMS_MANUAL" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
            rules={[
              { max: 500, message: '描述长度不能超过500个字符' }
            ]}
          >
            <TextArea
              rows={4}
              placeholder="请输入知识库描述（可选）"
            />
          </Form.Item>

          <Form.Item
            name="is_active"
            label="状态"
            valuePropName="checked"
          >
            <Switch
              checkedChildren="启用"
              unCheckedChildren="禁用"
            />
          </Form.Item>

          <Form.Item className="mb-0">
            <Space className="w-full justify-end">
              <Button
                onClick={() => {
                  setModalVisible(false)
                  setEditingKB(null)
                  form.resetFields()
                }}
              >
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                {editingKB ? '更新' : '创建'}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 查看知识库文档模态框 */}
      <Modal
        title={`知识库文档 - ${selectedKBName}`}
        open={documentsModalVisible}
        onCancel={() => setDocumentsModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDocumentsModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        <Table
          columns={documentColumns}
          dataSource={selectedKBDocuments}
          rowKey="id"
          pagination={{
            pageSize: 5,
            showTotal: (total) => `共 ${total} 个文档`
          }}
        />
      </Modal>
    </div>
  )
}

export default KnowledgeBase