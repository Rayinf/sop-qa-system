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
  Tabs,
  Breadcrumb,
  Tooltip,
  Switch
} from 'antd'
import {
  BookOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  FileTextOutlined,
  SettingOutlined,
  HomeOutlined
} from '@ant-design/icons'
import type { TableColumnsType } from 'antd'
import { KnowledgeBase, KnowledgeBaseCreate } from '../types/index'
import { getKnowledgeBases, createKnowledgeBase, updateKnowledgeBase, deleteKnowledgeBase } from '../services/kbApi'
import UnifiedKnowledgeManagement from './UnifiedKnowledgeManagement'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface KnowledgeBaseFormData {
  name: string
  code: string
  description?: string
  is_active: boolean
}

const KnowledgeManagement: React.FC = () => {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [editingKb, setEditingKb] = useState<KnowledgeBase | null>(null)
  const [selectedKb, setSelectedKb] = useState<KnowledgeBase | null>(null)
  const [form] = Form.useForm<KnowledgeBaseFormData>()

  // 加载知识库列表
  const loadKnowledgeBases = async () => {
    setLoading(true)
    try {
      const response = await getKnowledgeBases()
      setKnowledgeBases(response.kbs || [])
    } catch (error) {
      console.error('加载知识库失败:', error)
      message.error('加载知识库失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadKnowledgeBases()
  }, [])

  // 创建/编辑知识库
  const handleSubmit = async (values: KnowledgeBaseFormData) => {
    try {
      if (editingKb) {
        await updateKnowledgeBase(editingKb.id, values)
        message.success('知识库更新成功')
      } else {
        await createKnowledgeBase(values as KnowledgeBaseCreate)
        message.success('知识库创建成功')
      }
      setModalVisible(false)
      setEditingKb(null)
      form.resetFields()
      loadKnowledgeBases()
    } catch (error: any) {
      console.error('Submit KB error:', error)
      message.error(error.message || '操作失败')
    }
  }

  // 删除知识库
  const handleDelete = async (id: string) => {
    try {
      await deleteKnowledgeBase(id)
      message.success('知识库删除成功')
      loadKnowledgeBases()
      // 如果删除的是当前选中的知识库，清空选择
      if (selectedKb?.id === id) {
        setSelectedKb(null)
      }
    } catch (error: any) {
      message.error(error.message || '删除失败')
    }
  }

  // 打开创建模态框
  const openCreateModal = () => {
    setEditingKb(null)
    form.resetFields()
    form.setFieldsValue({ is_active: true })
    setModalVisible(true)
  }

  // 打开编辑模态框
  const openEditModal = (kb: KnowledgeBase) => {
    setEditingKb(kb)
    form.setFieldsValue({
      name: kb.name,
      code: kb.code,
      description: kb.description,
      is_active: kb.is_active
    })
    setModalVisible(true)
  }

  // 知识库表格列定义
  const columns: TableColumnsType<KnowledgeBase> = [
    {
      title: '知识库名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: KnowledgeBase) => (
        <Space>
          <BookOutlined />
          <div>
            <div style={{ fontWeight: 500 }}>{text}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.code}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      render: (text: string) => text || '-'
    },
    {
      title: '文档数量',
      dataIndex: 'document_count',
      key: 'document_count',
      render: (count: number) => (
        <Tag color="blue">{count} 个文档</Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      render: (isActive: boolean) => (
        <Tag color={isActive ? 'green' : 'red'}>
          {isActive ? '启用' : '禁用'}
        </Tag>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString('zh-CN')
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: string, record: KnowledgeBase) => (
        <Space>
          <Tooltip title="管理文档">
            <Button
              type="link"
              icon={<FileTextOutlined />}
              onClick={() => setSelectedKb(record)}
            >
              管理文档
            </Button>
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              type="link"
              icon={<EditOutlined />}
              onClick={() => openEditModal(record)}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个知识库吗？"
            description="删除后将无法恢复，且会删除所有相关文档。"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Tooltip title="删除">
              <Button
                type="link"
                icon={<DeleteOutlined />}
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ]

  // 如果选中了知识库，显示文档管理页面
  if (selectedKb) {
    return (
      <div>
        <div style={{ marginBottom: 16 }}>
          <Breadcrumb>
            <Breadcrumb.Item>
              <Button
                type="link"
                icon={<HomeOutlined />}
                onClick={() => setSelectedKb(null)}
                style={{ padding: 0 }}
              >
                知识库管理
              </Button>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <BookOutlined /> {selectedKb.name}
            </Breadcrumb.Item>
          </Breadcrumb>
        </div>
        <UnifiedKnowledgeManagement />
      </div>
    )
  }

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>知识库管理</Title>
        <Text type="secondary">
          管理知识库，每个知识库可以包含多个文档，用于分类组织和检索。
        </Text>
      </div>

      {/* 统计信息 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
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
              value={knowledgeBases.reduce((sum, kb) => sum + kb.document_count, 0)}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={openCreateModal}
              style={{ width: '100%' }}
            >
              新建知识库
            </Button>
          </Card>
        </Col>
      </Row>

      {/* 知识库列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={knowledgeBases}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total: number, range: [number, number]) => 
              `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
          }}
        />
      </Card>

      {/* 创建/编辑知识库模态框 */}
      <Modal
        title={editingKb ? '编辑知识库' : '新建知识库'}
        open={modalVisible}
        onOk={() => form.submit()}
        onCancel={() => {
          setModalVisible(false)
          setEditingKb(null)
          form.resetFields()
        }}
        okText="确定"
        cancelText="取消"
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          <Form.Item
            name="name"
            label="知识库名称"
            rules={[
              { required: true, message: '请输入知识库名称' },
              { max: 100, message: '名称不能超过100个字符' }
            ]}
          >
            <Input placeholder="请输入知识库名称" />
          </Form.Item>

          <Form.Item
            name="code"
            label="知识库代码"
            rules={[
              { required: true, message: '请输入知识库代码' },
              { max: 50, message: '代码不能超过50个字符' },
              { pattern: /^[A-Z0-9_]+$/, message: '代码只能包含大写字母、数字和下划线' }
            ]}
          >
            <Input 
              placeholder="请输入知识库代码（如：MANUAL_2024）" 
              style={{ textTransform: 'uppercase' }}
            />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <Input.TextArea
              rows={3}
              placeholder="请输入知识库描述（可选）"
              maxLength={500}
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
        </Form>
      </Modal>
    </div>
  )
}

export default KnowledgeManagement